from typing import Protocol
import jax
import jax.numpy as jnp


class Sentiment(Protocol):
    def __call__(self, values: jax.Array) -> jax.Array:
        ...


def bayes(values: jax.Array) -> jax.Array:
    return values.mean()


class Optimism:
    def __init__(self, exploration_scale: float):
        self.exploration_scale = exploration_scale

    def __call__(self, values: jax.Array) -> jax.Array:
        exploration_bonus = value_epistemic_uncertainty(values)
        return bayes(values) + self.exploration_scale * exploration_bonus


def value_epistemic_uncertainty(values: jax.Array) -> jax.Array:
    return jnp.var(values[..., 0], 1).mean()
