from typing import Protocol
import jax


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
    # FIXME (yarden): The problem here is that if this is being summed
    # with values from a whole trajectory, gradients of steps 1,...,T
    # are going to use the value function without exploration at all.
    return values[..., 0].std(1).mean()
