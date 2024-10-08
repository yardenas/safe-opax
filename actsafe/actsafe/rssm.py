from typing import NamedTuple

import distrax as dtx
import equinox as eqx
import jax
import jax.flatten_util
import jax.nn as jnn
import jax.numpy as jnp

from actsafe.rl.utils import init_linear_weights_and_biases


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
            axis=-1,
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
            [self.observation, self.reward, self.cost, self.terminal],
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
    ) -> tuple[ShiftScale, jax.Array]:
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
    priors: Prior
    posterior: Posterior
    deterministic_size: int = eqx.field(static=True)
    stochastic_size: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        embedding_size: int,
        action_dim: int,
        ensemble_size: int,
        initialization_scale: float | None = None,
        *,
        key: jax.Array,
    ):
        self.ensemble_size = ensemble_size
        prior_key, posterior_key = jax.random.split(key)
        dummy_prior = Prior(
            deterministic_size,
            stochastic_size,
            hidden_size,
            action_dim,
            key,
        )
        initialization_scale = (
            initialization_scale
            if initialization_scale is not None
            else jax.flatten_util.ravel_pytree(dummy_prior)[0].std()
        )
        initialization_scale = initialization_scale if ensemble_size > 1 else 0.
        self.priors = jitter_priors(
            dummy_prior, prior_key, initialization_scale, ensemble_size
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
        self, prev_state: State, action: jax.Array, key: jax.Array
    ) -> tuple[State, ShiftScale]:
        prior, deterministic = _priors_predict(self.priors, prev_state, action)
        stochastic = dtx.Independent(dtx.Normal(*prior)).sample(seed=key)
        return State(stochastic, deterministic), prior

    def filter(
        self,
        prev_state: State,
        embeddings: jax.Array,
        action: jax.Array,
        key: jax.Array,
    ) -> tuple[State, ShiftScale, ShiftScale]:
        key, prior_key = jax.random.split(key)
        id = jax.random.randint(prior_key, (), 0, self.ensemble_size)
        prior_model_sample = jax.tree_map(
            lambda x: x[id], self.priors, is_leaf=eqx.is_array
        )
        prior, deterministic = prior_model_sample(prev_state, action)
        state = State(prev_state.stochastic, deterministic)
        posterior = self.posterior(state, embeddings)
        stochastic = dtx.Normal(*posterior).sample(seed=key)
        return State(stochastic, deterministic), posterior, prior

    @property
    def init(self) -> State:
        dtype = self.dtype
        return State(
            jnp.zeros(self.stochastic_size, dtype),
            jnp.zeros(self.deterministic_size, dtype),
        )

    @property
    def dtype(self):
        dtype = self.priors.encoder.weight.dtype
        assert all(dtype == x for x in jax.tree_flatten(self)[0] if eqx.is_array(x))
        return dtype


def _priors_predict(
    priors: RSSM,
    prev_state: State,
    action: jax.Array,
    *,
    vmap_state: bool = False,
    vmap_action: bool = False,
):
    prev_state_in_axis = 0 if vmap_state else None
    action_in_axis = 0 if vmap_action else None
    priors_fn = eqx.filter_vmap(
        lambda prior, prev_state, action: prior(prev_state, action),
        in_axes=(eqx.if_array(0), prev_state_in_axis, action_in_axis),
    )
    return priors_fn(priors, prev_state, action)


def jitter_priors(
    prior: Prior, key: jax.Array, scale: float, ensemble_size: int
) -> Prior:
    make_priors = eqx.filter_vmap(
        lambda key: init_linear_weights_and_biases(
            prior, lambda x, subkey: x + scale * jax.random.normal(subkey, x.shape), key
        )
    )
    return make_priors(jnp.asarray(jax.random.split(key, ensemble_size)))
