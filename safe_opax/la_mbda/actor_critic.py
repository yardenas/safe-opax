from typing import Any
import equinox as eqx
import jax
import distrax as trx
import jax.numpy as jnp
import jax.nn as jnn

from safe_opax.common.math import inv_softplus
from safe_opax.rl.utils import rl_initialize_weights_trick


class StableTanh(trx.Tanh):
    def inverse_and_log_det(self, y):
        dtype = y.dtype
        y = y.astype(jnp.float32)
        # Clip to avoid computing very large gradients outside of
        # the given range.
        y = jnp.clip(y, -0.99999997, 0.99999997)
        x = jnp.arctanh(y)
        x = x.astype(dtype)
        return x, -self.forward_log_det_jacobian(x)


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
        initialization_scale: float,
        *,
        key: jax.Array,
    ):
        self.net = rl_initialize_weights_trick(
            eqx.nn.MLP(
                state_dim,
                action_dim * 2,
                hidden_size,
                n_layers + 1,
                key=key,
                activation=jnn.elu,
            ),
            weight_scale=initialization_scale,
        )
        self.init_stddev = init_stddev

    def __call__(self, state: jax.Array) -> trx.Transformed:
        x = self.net(state)
        mu, stddev = jnp.split(x, 2, axis=-1)
        init_std = inv_softplus(self.init_stddev)
        stddev = jnn.softplus(stddev + init_std) + 1e-4
        mu = 5.0 * jnn.tanh(mu / 5.0)
        dist = trx.MultivariateNormalDiag(mu, stddev)
        bijector = trx.Block(StableTanh(), 1)
        dist = trx.Transformed(dist, bijector)
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
            state_dim, "scalar", hidden_size + 1, n_layers, key=key, activation=jnn.elu
        )

    def __call__(self, observation: Any) -> jax.Array:
        x = self.net(observation)
        return x


def actor_entropy(actor, states):
    log_prob = lambda state: actor(state).sample_and_log_prob(
        seed=jax.random.PRNGKey(0)
    )[1]
    return -eqx.filter_vmap(log_prob)(states).mean()
