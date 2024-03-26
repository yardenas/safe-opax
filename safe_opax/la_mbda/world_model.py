from typing import NamedTuple, TypedDict
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import distrax as dtx
from optax import OptState

from safe_opax.common.learner import Learner
from safe_opax.common.mixed_precision import apply_mixed_precision
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
                    eqx.nn.ConvTranspose2d(in_channels, 3, kernel, 2, key=key)
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
    cells: RSSM
    encoder: Encoder
    image_decoder: ImageDecoder
    reward_cost_decoder: eqx.nn.MLP
    ensemble_size: int = eqx.field(static=True)

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        ensemble_size: int,
        *,
        key,
    ):
        (
            cell_key,
            encoder_key,
            image_decoder_key,
            reward_cost_decoder_key,
        ) = jax.random.split(key, 4)
        self.ensemble_size = ensemble_size
        cell_keys = jax.random.split(cell_key, ensemble_size)
        make_ensemble = jax.vmap(
            lambda key: RSSM(
                deterministic_size,
                stochastic_size,
                hidden_size,
                _EMBEDDING_SIZE,
                action_dim,
                key,
            )
        )
        self.cells = make_ensemble(jnp.asarray(cell_keys))
        self.encoder = Encoder(key=encoder_key)
        state_dim = stochastic_size + deterministic_size
        self.image_decoder = ImageDecoder(state_dim, image_shape, key=image_decoder_key)
        # 1 + 1 = cost + reward
        # width = 400, layers = 2
        self.reward_cost_decoder = eqx.nn.MLP(
            state_dim, 1 + 1, 400, 2, key=reward_cost_decoder_key, activation=jnn.elu
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
            state, posterior, prior = _ensemble_infer(
                self.cells, prev_state, embedding, prev_action, key, vmap_state=True
            )
            return state, (state, posterior, prior)

        keys = jax.random.split(key, obs_embeddings.shape[0])
        _, (states, posteriors, priors) = jax.lax.scan(
            fn,
            init_state if init_state is not None else _init_rssm_state(self.cells),
            (obs_embeddings, actions, keys),
        )
        states, posteriors, priors = _ensemble_first((states, posteriors, priors))
        states, posteriors, priors = marginalize_prediction(
            (states, posteriors, priors)
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
        state, *_ = _ensemble_infer(self.cells, state, obs_embeddings, action, key)
        state = marginalize_prediction(state)
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
            state, prior = predict_fn(self.cells, prev_state, action, key)
            return state, (state, prior)

        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.Array] | jax.Array = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
            predict_fn = eqx.filter_vmap(
                lambda rssm, state, action, key: rssm.predict(state, action, key),
                in_axes=(eqx.if_array(0), 0, None, None),
            )
        elif callable(policy):
            policy = jax.vmap(policy, in_axes=(0, None))
            inputs = jax.random.split(key, horizon)
            predict_fn = eqx.filter_vmap(
                lambda rssm, state, action, key: rssm.predict(state, action, key),
                in_axes=(eqx.if_array(0), 0, 0, None),
            )
        else:
            raise ValueError("policy must be callable or jax.Array")
        if isinstance(initial_state, jax.Array):
            initial_state = State.from_flat(initial_state, self.cells.stochastic_size)
        initial_state = jax.tree_map(
            lambda x: jnp.repeat(x[None], self.ensemble_size, 0), initial_state
        )
        _, (trajectory, priors) = jax.lax.scan(f, initial_state, inputs)
        # vmap twice: once for the ensemble, and second time for the horizon
        out = jax.vmap(jax.vmap(self.reward_cost_decoder))(trajectory.flatten())
        reward, cost = out[..., 0], out[..., -1]
        out = Prediction(trajectory.flatten(), reward, cost)
        # ensemble dimension first, then horizon
        out, priors = _ensemble_first((out, priors))
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
) -> tuple[tuple[WorldModel, OptState], tuple[jax.Array, TrainingResults]]:
    def loss_fn(model):
        infer_fn = lambda features, actions: model(features, actions, key)
        inference_result: InferenceResult = eqx.filter_vmap(infer_fn)(features, actions)
        y = features.observation, jnp.concatenate([features.reward, features.cost], -1)
        y_hat = inference_result.image, inference_result.reward_cost
        reconstruction_loss = -sum(
            map(
                lambda predictions, targets: dtx.Independent(
                    dtx.Normal(targets, 1.0), targets.ndim - 2
                )
                .log_prob(predictions)
                .mean(),
                y_hat,
                y,
            )
        )
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
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


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


def marginalize_prediction(x):
    return jax.tree_map(lambda x: x.mean(0), x)


def _init_rssm_state(cells):
    return eqx.filter_vmap(lambda cells: cells.init())(cells)


def _ensemble_first(x):
    return jax.tree_map(lambda x: x.swapaxes(0, 1), x)


def _ensemble_infer(rssm, prev_state, embedding, prev_action, key, *, vmap_state=False):
    prev_state_in_axis = 0 if vmap_state else None
    filter_fn = eqx.filter_vmap(
        lambda rssm, prev_state, embedding, prev_action, key: rssm.filter(
            prev_state, embedding, prev_action, key
        ),
        in_axes=(eqx.if_array(0), prev_state_in_axis, None, None, None),
    )
    return filter_fn(rssm, prev_state, embedding, prev_action, key)
