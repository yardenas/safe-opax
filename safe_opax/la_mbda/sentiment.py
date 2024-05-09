from typing import Protocol
import jax

from safe_opax.la_mbda.rssm import ShiftScale
from safe_opax.opax import normalized_epistemic_uncertainty


class Sentiment(Protocol):
    def __call__(self, values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
        ...


def identity(values: jax.Array) -> jax.Array:
    return values


def bayes(values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
    return values.mean(1)


class UpperConfidenceBound(Sentiment):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
        bonus = normalized_epistemic_uncertainty(state_distribution, 1)
        return values.mean(1) + self.alpha * bonus
