from typing import NamedTuple

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from safe_opax.common.learner import Learner
from safe_opax.la_mbda.types import Prediction
from safe_opax.rl.types import Policy


class State(NamedTuple):
    stochastic: jax.Array
    deterministic: jax.Array

    def flatten(self):
        return jnp.concatenate([self.stochastic, self.deterministic], axis=-1)

    @classmethod
    def from_flat(cls, flat, stochastic_size):
        stochastic, deterministic = jnp.split(
            flat,
            [
                stochastic_size,
            ],  # type: ignore
        )
        self = cls(stochastic, deterministic)
        return self


class Features(NamedTuple):
    observation: jax.Array
    reward: jax.Array
    cost: jax.Array
    terminal: jax.Array
    done: jax.Array

    def flatten(self):
        return jnp.concatenate(
            [self.observation, self.reward, self.cost, self.terminal, self.done],
            axis=-1,
        )


class ShiftScale(NamedTuple):
    shift: jax.Array
    scale: jax.Array


class Prior(eqx.Module):
    cell: eqx.nn.GRUCell
    encoder: eqx.nn.Linear
    decoder1: eqx.nn.Linear
    decoder2: eqx.nn.Linear

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        action_dim: int,
        key: jax.random.KeyArray,
    ):
        encoder_key, cell_key, decoder1_key, decoder2_key = jax.random.split(key, 4)
        self.encoder = eqx.nn.Linear(
            stochastic_size + action_dim, deterministic_size, key=encoder_key
        )
        self.cell = eqx.nn.GRUCell(deterministic_size, deterministic_size, key=cell_key)
        self.decoder1 = eqx.nn.Linear(deterministic_size, hidden_size, key=decoder1_key)
        self.decoder2 = eqx.nn.Linear(
            hidden_size, stochastic_size * 2, key=decoder2_key
        )

    def __call__(
        self, prev_state: State, action: jax.Array
    ) -> tuple[dtx.Normal, jax.Array]:
        x = jnp.concatenate([prev_state.stochastic, action], -1)
        x = jnn.elu(self.encoder(x))
        hidden = self.cell(x, prev_state.deterministic)
        x = jnn.elu(self.decoder1(hidden))
        x = self.decoder2(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev), hidden


class Posterior(eqx.Module):
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        embedding_size: int,
        key: jax.random.KeyArray,
    ):
        encoder_key, decoder_key = jax.random.split(key)
        self.encoder = eqx.nn.Linear(
            deterministic_size + embedding_size, hidden_size, key=encoder_key
        )
        self.decoder = eqx.nn.Linear(hidden_size, stochastic_size * 2, key=decoder_key)

    def __call__(self, prev_state: State, embedding: jax.Array):
        x = jnp.concatenate([prev_state.deterministic, embedding], -1)
        x = jnn.elu(self.encoder(x))
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)


class RSSM(eqx.Module):
    prior: Prior
    posterior: Posterior
    deterministic_size: int = eqx.static_field()
    stochastic_size: int = eqx.static_field()

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        embedding_size: int,
        action_dim: int,
        key: jax.random.KeyArray,
    ):
        prior_key, posterior_key = jax.random.split(key)
        self.prior = Prior(
            deterministic_size,
            stochastic_size,
            hidden_size,
            action_dim,
            prior_key,
        )
        self.posterior = Posterior(
            deterministic_size,
            stochastic_size,
            hidden_size,
            embedding_size,
            posterior_key,
        )
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size

    def predict(
        self, prev_state: State, action: jax.Array, key: jax.random.KeyArray
    ) -> State:
        prior, deterministic = self.prior(prev_state, action)
        stochastic = dtx.Normal(*prior).sample(seed=key)
        return State(stochastic, deterministic)

    def filter(
        self,
        prev_state: State,
        embeddings: jax.Array,
        action: jax.Array,
        key: jax.random.KeyArray,
    ) -> tuple[State, ShiftScale, ShiftScale]:
        prior, deterministic = self.prior(prev_state, action)
        state = State(prev_state.stochastic, deterministic)
        posterior = self.posterior(state, embeddings)
        stochastic = dtx.Normal(*posterior).sample(seed=key)
        return State(stochastic, deterministic), posterior, prior

    @property
    def init(self) -> State:
        return State(
            jnp.zeros(self.stochastic_size), jnp.zeros(self.deterministic_size)
        )


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
