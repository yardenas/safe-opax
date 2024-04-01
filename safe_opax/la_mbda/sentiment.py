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
        exploration_bonus = values.mean((0, -1)).std()
        return bayes(values) + self.exploration_scale * exploration_bonus
