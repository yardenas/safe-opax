from typing import Callable, Protocol
import jax
import jax.numpy as jnp


class Sentiment(Protocol):
    def __call__(self, values: jax.Array) -> jax.Array:
        ...


def identity(values: jax.Array) -> jax.Array:
    return values


def bayes(values: jax.Array) -> jax.Array:
    return values.mean(1)


class UpperConfidenceBound(Sentiment):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, values: jax.Array) -> jax.Array:
        return upper_confidence_bound(values, self.alpha)


def upper_confidence_bound(
    values: jax.Array, alpha: float, stop_gradient: bool = True
) -> jax.Array:
    stddev = jnp.std(values, axis=1)
    if stop_gradient:
        stddev = jax.lax.stop_gradient(stddev)
    return jnp.mean(values, axis=1) + alpha * stddev


def _emprirical_estimate(
    values: jax.Array, reduce_fn: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    ids = reduce_fn(values.mean((0, 2)))
    return values[:, ids].mean()


def empirical_optimism(values: jax.Array) -> jax.Array:
    return _emprirical_estimate(values, jnp.argmax)


def empirical_robustness(values: jax.Array) -> jax.Array:
    return _emprirical_estimate(values, jnp.argmin)
