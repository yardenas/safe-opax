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

class WorldModel(eqx.Module):
    cell: RSSM
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        *,
        key,
    ):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = RSSM(
            deterministic_size,
            stochastic_size,
            hidden_size,
            hidden_size,
            action_dim,
            cell_key,
        )
        self.encoder = eqx.nn.Linear(state_dim, hidden_size, key=encoder_key)
        # 1 + 1 = cost + reward
        self.decoder = eqx.nn.Linear(
            deterministic_size + stochastic_size, state_dim + 1 + 1, key=decoder_key
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        key: jax.random.KeyArray,
        init_state: State | None = None,
    ) -> tuple[State, jax.Array, ShiftScale, ShiftScale]:
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
        outs = jax.vmap(self.decoder)(states.flatten())
        return states, outs, posteriors, priors

    def step(
        self,
        state: State,
        observation: jax.Array,
        action: jax.Array,
        key: jax.random.KeyArray,
    ) -> State:
        obs_embeddings = jnn.elu(self.encoder(observation))
        state, *_ = self.cell.filter(state, obs_embeddings, action, key)
        return state

    def sample(
        self,
        horizon: int,
        state: State | jax.Array,
        key: jax.random.KeyArray,
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
            inputs: tuple[jax.Array, jax.random.KeyArray] | jax.random.KeyArray = (
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
    key: jax.random.KeyArray,
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
