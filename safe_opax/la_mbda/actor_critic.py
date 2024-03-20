from typing import Any
import equinox as eqx
import jax
import distrax as trx
import jax.numpy as jnp
import jax.nn as jnn

from safe_opax.common.math import inv_softplus


class ContinuousActor(eqx.Module):
    net: eqx.nn.MLP
    init_stddev: float = eqx.static_field()

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        init_stddev: float,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            state_dim,
            action_dim * 2,
            hidden_size,
            n_layers,
            key=key,
            activation=jnn.elu,
        )
        self.init_stddev = init_stddev

    def __call__(self, state: jax.Array) -> trx.Transformed:
        x = self.net(state)
        mu, stddev = jnp.split(x, 2, axis=-1)
        init_std = inv_softplus(self.init_stddev)
        stddev = jnn.softplus(stddev + init_std) + 0.1
        mu = 5.0 * jnn.tanh(mu / 5.0)
        dist = trx.Normal(mu, stddev)
        dist = trx.Transformed(dist, trx.Tanh())
        return dist

    def act(
        self,
        observation: Any,
        key: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if deterministic:
            samples, log_probs = self(observation).sample_and_log_prob(
                seed=jax.random.PRNGKey(0), sample_shape=100
            )
            most_likely = jnp.argmax(log_probs)
            return samples[most_likely]
        else:
            assert key is not None
            return self(observation).sample(seed=key)


class Critic(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            state_dim, 1, hidden_size, n_layers, key=key, activation=jnn.elu
        )

    def __call__(self, observation: Any) -> jax.Array:
        x = self.net(observation)
        return x.squeeze(-1)
