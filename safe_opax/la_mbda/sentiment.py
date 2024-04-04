from typing import Callable, Protocol
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
        return bayes(values) * 0.0 + self.exploration_scale * exploration_bonus


def value_epistemic_uncertainty(values: jax.Array) -> jax.Array:
    return values.std(1).mean()


def _emprirical_estimate(
    values: jax.Array, reduce_fn: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    ids = reduce_fn(values.mean((0, 2)))
    return values[:, ids].mean()


def empirical_optimism(values: jax.Array) -> jax.Array:
    return _emprirical_estimate(values, jnp.argmax)


def empirical_robustness(values: jax.Array) -> jax.Array:
    return _emprirical_estimate(values, jnp.argmin)
