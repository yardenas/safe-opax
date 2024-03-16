from typing import NamedTuple

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


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
        key: jax.Array,
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
        key: jax.Array,
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
        key: jax.Array,
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

    def predict(self, prev_state: State, action: jax.Array, key: jax.Array) -> State:
        prior, deterministic = self.prior(prev_state, action)
        stochastic = dtx.Normal(*prior).sample(seed=key)
        return State(stochastic, deterministic)

    def filter(
        self,
        prev_state: State,
        embeddings: jax.Array,
        action: jax.Array,
        key: jax.Array,
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
