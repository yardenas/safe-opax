from typing import NamedTuple
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import distrax as dtx
from optax import OptState, l2_loss

from safe_opax.common.learner import Learner
from safe_opax.la_mbda.rssm import RSSM, Features, ShiftScale, State
from safe_opax.la_mbda.types import Prediction
from safe_opax.rl.types import Policy

_EMBEDDING_SIZE = 1024

class Encoder(eqx.Module):
    cnn_layers: list[eqx.nn.Conv2d]

    def __init__(
        self,
        *,
        key: jax.Array,
    ):
        kernels = [4, 4, 4, 4]
        depth = 32
        keys = jax.random.split(key, len(kernels))
        in_channels = 3
        self.cnn_layers = []
        for i, (key, kernel) in enumerate(zip(keys, kernels)):
            out_channels = 2**i * depth
            self.cnn_layers.append(
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=2,
                    key=key,
                )
            )
            in_channels = out_channels

    def __call__(self, observation: jax.Array) -> jax.Array:
        x = observation
        for layer in self.cnn_layers:
            x = jnn.relu(layer(x))
        x = x.ravel()
        return x


class Decoder(eqx.Module):
    linear: eqx.nn.Linear
    cnn_layers: list[eqx.nn.ConvTranspose2d]
    output_shape: tuple[int, int, int] = eqx.static_field()

    def __init__(
        self,
        output_shape: tuple[int, int, int],
        *,
        key: jax.Array,
    ):
        kernels = [5, 5, 6, 6]
        depth = 32
        linear_key, *keys = jax.random.split(key, len(kernels) + 1)
        in_channels = 32 * depth
        self.linear = eqx.nn.Linear(_EMBEDDING_SIZE, in_channels, key=linear_key)
        self.cnn_layers = []
        for i, (key, kernel) in enumerate(zip(keys, kernels)):
            if i != len(kernels) - 1:
                out_channels = 2 ** (len(kernels) - i - 2) * depth
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(
                        in_channels, out_channels, kernel, 2, key=key
                    )
                )
            else:
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(in_channels, 3, kernel, 2, key=key)
                )
            in_channels = out_channels
        self.output_shape = output_shape

    def __call__(self, flat_state: jax.Array) -> jax.Array:
        x = self.linear(flat_state)
        for layer in self.cnn_layers:
            x = jnn.relu(layer(x))
        output = x.reshape(self.output_shape)
        return output


class InferenceResult(NamedTuple):
    state: State
    image: jax.Array
    reward_cost: jax.Array
    posteriors: ShiftScale
    priors: ShiftScale


class WorldModel(eqx.Module):
    cell: RSSM
    encoder: Encoder
    image_decoder: Decoder
    reward_cost_decoder: eqx.nn.Linear

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        *,
        key,
    ):
        (
            cell_key,
            encoder_key,
            image_decoder_key,
            reward_cost_decoder_key,
        ) = jax.random.split(key, 4)
        self.cell = RSSM(
            deterministic_size,
            stochastic_size,
            hidden_size,
            _EMBEDDING_SIZE,
            action_dim,
            cell_key,
        )
        self.encoder = Encoder(key=encoder_key)
        self.image_decoder = Decoder(image_shape, key=image_decoder_key)
        # 1 + 1 = cost + reward
        # TODO (yarden): should have more layers
        self.reward_cost_decoder = eqx.nn.Linear(
            deterministic_size + stochastic_size,
            1 + 1,
            key=reward_cost_decoder_key,
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        key: jax.Array,
        init_state: State | None = None,
    ) -> InferenceResult:
        obs_embeddings = jnn.elu(jax.vmap(self.encoder)(features.observation))

        def fn(carry, inputs):
            prev_state = carry
            embedding, prev_action, key = inputs
            state, posterior, prior = self.cell.filter(
                prev_state, embedding, prev_action, key
            )
            return state, (state, posterior, prior)

        keys = jax.random.split(key, obs_embeddings.shape[0])
        _, (states, posteriors, priors) = jax.lax.scan(
            fn,
            init_state if init_state is not None else self.cell.init,
            (obs_embeddings, actions, keys),
        )
        reward_cost = jax.vmap(self.decoder)(states.flatten())
        image = jax.vmap(self.image_decoder)(states.flatten())
        return InferenceResult(states, image, reward_cost, posteriors, priors)

    def infer_state(
        self,
        state: State,
        observation: jax.Array,
        action: jax.Array,
        key: jax.Array,
    ) -> State:
        obs_embeddings = self.encoder(observation)
        state, *_ = self.cell.filter(state, obs_embeddings, action, key)
        return state

    def sample(
        self,
        horizon: int,
        state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> Prediction:
        def f(carry, inputs):
            prev_state = carry
            if callable_policy:
                key = inputs
                key, p_key = jax.random.split(key)
                action = policy(jax.lax.stop_gradient(prev_state.flatten()), p_key)
            else:
                action, key = inputs
            state = self.cell.predict(prev_state, action, key)
            return state, state

        callable_policy = False
        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.Array] | jax.Array = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
        else:
            callable_policy = True
            inputs = jax.random.split(key, horizon)
        if isinstance(state, jax.Array):
            state = State.from_flat(state, self.cell.stochastic_size)
        _, state = jax.lax.scan(
            f,
            state,
            inputs,
        )
        out = jax.vmap(self.decoder)(state.flatten())
        reward, cost = out[:, -2], out[:, -1]
        out = Prediction(state.flatten(), reward, cost)
        return out


@eqx.filter_jit
def variational_step(
    features: Features,
    actions: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.Array,
    beta: float = 1.0,
    free_nats: float = 0.0,
):
    def loss_fn(model):
        infer_fn = lambda features, actions: model(features, actions, key)
        states, y_hat, posteriors, priors = eqx.filter_vmap(infer_fn)(features, actions)
        y = jnp.concatenate([features.observation, features.reward, features.cost], -1)
        reconstruction_loss = l2_loss(y_hat, y).mean()
        dynamics_kl_loss = kl_divergence(posteriors, priors, free_nats).mean()
        kl_loss = dynamics_kl_loss
        aux = dict(
            reconstruction_loss=reconstruction_loss,
            kl_loss=dynamics_kl_loss,
            states=states,
        )
        return reconstruction_loss + beta * kl_loss, aux

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float = 0.0
) -> jax.Array:
    prior_dist = dtx.MultivariateNormalDiag(*prior)
    posterior_dist = dtx.MultivariateNormalDiag(*posterior)
    return jnp.maximum(posterior_dist.kl_divergence(prior_dist), free_nats)
