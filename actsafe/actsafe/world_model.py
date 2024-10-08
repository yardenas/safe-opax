from typing import NamedTuple, TypedDict
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import distrax as dtx
from optax import OptState

from actsafe.common.learner import Learner
from actsafe.common.mixed_precision import apply_mixed_precision
from actsafe.actsafe.rssm import RSSM, Features, ShiftScale, State
from actsafe.rl.types import Prediction
from actsafe.actsafe.utils import marginalize_prediction
from actsafe.rl.types import Policy
from actsafe.rl.utils import nest_vmap

_EMBEDDING_SIZE = 1024


class Encoder(eqx.Module):
    cnn_layers: list[eqx.nn.Conv2d]

    def __init__(
        self,
        image_channels: int,
        *,
        key: jax.Array,
    ):
        kernels = [4, 4, 4, 4]
        depth = 32
        keys = jax.random.split(key, len(kernels))
        in_channels = image_channels
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
        for layer in self.cnn_layers[:-1]:
            x = jnn.elu(layer(x))
        x = self.cnn_layers[-1](x)
        x = x.ravel()
        return x


class ImageDecoder(eqx.Module):
    linear: eqx.nn.Linear
    cnn_layers: list[eqx.nn.ConvTranspose2d]
    output_shape: tuple[int, int, int] = eqx.static_field()

    def __init__(
        self,
        state_dim: int,
        output_shape: tuple[int, int, int],
        *,
        key: jax.Array,
    ):
        kernels = [5, 5, 6, 6]
        depth = 32
        linear_key, *keys = jax.random.split(key, len(kernels) + 1)
        in_channels = _EMBEDDING_SIZE
        self.linear = eqx.nn.Linear(state_dim, in_channels, key=linear_key)
        self.cnn_layers = []
        for i, (key, kernel) in enumerate(zip(keys, kernels)):
            out_channels = 2 ** (len(kernels) - i - 2) * depth
            if i != len(kernels) - 1:
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(
                        in_channels, out_channels, kernel, 2, key=key
                    )
                )
            else:
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(
                        in_channels, output_shape[0], kernel, 2, key=key
                    )
                )
            in_channels = out_channels
        self.output_shape = output_shape

    def __call__(self, flat_state: jax.Array) -> jax.Array:
        x = self.linear(flat_state)
        x = x.reshape(_EMBEDDING_SIZE, 1, 1)
        for layer in self.cnn_layers[:-1]:
            x = jnn.elu(layer(x))
        x = self.cnn_layers[-1](x)
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
    image_decoder: ImageDecoder
    reward_cost_decoder: eqx.nn.MLP

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        ensemble_size: int,
        initialization_scale: float,
        num_rewards: int,
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
            ensemble_size,
            initialization_scale,
            key=cell_key,
        )
        self.encoder = Encoder(image_channels=image_shape[0], key=encoder_key)
        state_dim = stochastic_size + deterministic_size
        self.image_decoder = ImageDecoder(state_dim, image_shape, key=image_decoder_key)
        # num_rewards + 1 = cost + reward
        # width = 400, layers = 2
        self.reward_cost_decoder = eqx.nn.MLP(
            state_dim,
            num_rewards + 1,
            400,
            3,
            key=reward_cost_decoder_key,
            activation=jnn.elu,
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        key: jax.Array,
        init_state: State | None = None,
    ) -> InferenceResult:
        obs_embeddings = jax.vmap(self.encoder)(features.observation)

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
        reward_cost = jax.vmap(self.reward_cost_decoder)(states.flatten())
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
        initial_state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, ShiftScale]:
        def f(carry, inputs):
            prev_state = carry
            if callable(policy):
                key = inputs
                key, p_key = jax.random.split(key)
                action = policy(jax.lax.stop_gradient(prev_state.flatten()), p_key)
            else:
                action, key = inputs
            ensemble_states, prior = self.cell.predict(prev_state, action, key)
            key, prior_key = jax.random.split(key)
            id = jax.random.randint(prior_key, (), 0, self.cell.ensemble_size)
            state = jax.tree_map(lambda x: x[id], ensemble_states)
            return state, (state, ensemble_states, prior)

        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.Array] | jax.Array = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
        elif callable(policy):
            inputs = jax.random.split(key, horizon)
        else:
            raise ValueError("policy must be callable or jax.Array")
        if isinstance(initial_state, jax.Array):
            initial_state = State.from_flat(initial_state, self.cell.stochastic_size)
        _, (trajectory, ensemble_trajectories, priors) = jax.lax.scan(
            f, initial_state, inputs
        )
        # vmap twice: once for the ensemble, and second time for the horizon
        out = nest_vmap(self.reward_cost_decoder, 2)(ensemble_trajectories.flatten())
        # Ensemble axis before time axis.
        out, priors = _ensemble_first((out, priors))
        reward, cost = out[..., :-1], out[..., -1]
        out = Prediction(trajectory.flatten(), reward, cost)
        return out, priors


class TrainingResults(TypedDict):
    reconstruction_loss: jax.Array
    kl_loss: jax.Array
    states: State


@eqx.filter_jit
@apply_mixed_precision(
    target_input_names=["features", "actions"],
    target_module_names=["model"],
)
def variational_step(
    features: Features,
    actions: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.Array,
    beta: float = 1.0,
    free_nats: float = 0.0,
    kl_mix: float = 0.8,
    with_reward: bool = True,
    inference_only: bool = False,
) -> tuple[tuple[WorldModel, OptState], tuple[jax.Array, TrainingResults]]:
    def loss_fn(model, static_part=None):
        if static_part is not None:
            model = eqx.combine(model, static_part)
        infer_fn = lambda features, actions: model(features, actions, key)
        inference_result: InferenceResult = eqx.filter_vmap(infer_fn)(features, actions)
        batch_ndim = 2
        logprobs = (
            lambda predictions, targets: dtx.Independent(
                dtx.Normal(targets, 1.0), targets.ndim - batch_ndim
            )
            .log_prob(predictions)
            .mean()
        )
        if not with_reward:
            reward = jnp.zeros_like(features.reward)
            _, pred_cost = jnp.split(inference_result.reward_cost, 2, -1)
            reward_cost = jnp.concatenate([reward, pred_cost[..., None]], -1)
        else:
            reward = features.reward
            reward_cost = inference_result.reward_cost
        reward_cost_logprobs = logprobs(
            reward_cost,
            jnp.concatenate([reward, features.cost[..., None]], -1),
        )
        image_logprobs = logprobs(inference_result.image, features.observation)
        reconstruction_loss = -reward_cost_logprobs - image_logprobs
        kl_loss = kl_divergence(
            inference_result.posteriors, inference_result.priors, free_nats, kl_mix
        )
        assert isinstance(reconstruction_loss, jax.Array)
        aux = TrainingResults(
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            states=inference_result.state,
        )
        return reconstruction_loss + beta * kl_loss, aux

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    if inference_only:
        return (model, opt_state), (loss, rest)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


def partition_dynamics_rewards(model: WorldModel) -> tuple[WorldModel, WorldModel]:
    filter_spec = jax.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: tree.reward_cost_decoder, filter_spec, True)
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


# https://github.com/danijar/dreamerv2/blob/259e3faa0e01099533e29b0efafdf240adeda4b5/common/nets.py#L130
def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float, mix: float
) -> jax.Array:
    sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)
    mvn = lambda scale_shift: dtx.MultivariateNormalDiag(*scale_shift)
    lhs = mvn(posterior).kl_divergence(mvn(sg(prior))).mean()
    rhs = mvn(sg(posterior)).kl_divergence(mvn(prior)).mean()
    return (1.0 - mix) * jnp.maximum(lhs, free_nats) + mix * jnp.maximum(rhs, free_nats)


@eqx.filter_jit
def evaluate_model(
    model: WorldModel, features: Features, actions: jax.Array, key: jax.Array
) -> jax.Array:
    observations = features.observation
    length = min(observations.shape[1] + 1, 50)
    conditioning_length = length // 5
    key, subkey = jax.random.split(key)
    features = jax.tree_map(lambda x: x[0, :conditioning_length], features)
    inference_result = model(features, actions[0, :conditioning_length], subkey)
    state = jax.tree_map(lambda x: x[-1], inference_result.state)
    prediction, _ = model.sample(
        length - conditioning_length,
        state,
        key,
        actions[0, conditioning_length:],
    )
    prediction = marginalize_prediction(prediction)
    y_hat = jax.vmap(model.image_decoder)(prediction.next_state)
    y = observations[0, conditioning_length:]
    error = jnp.abs(y - y_hat) / 2.0 - 0.5
    normalize = lambda image: ((image + 0.5) * 255).astype(jnp.uint8)
    out = jnp.stack([normalize(x) for x in [y, y_hat, error]])
    return out


def _ensemble_first(x):
    return jax.tree_map(lambda x: x.swapaxes(0, 1), x)
