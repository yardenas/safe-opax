from typing import (
    NamedTuple,
    Optional,
    Protocol,
)

import jax

from safe_opax.rl.types import FloatArray, Policy


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array
    cost: jax.Array


class Model(Protocol):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...

    def step(self, state: jax.Array, action: jax.Array) -> Prediction:
        ...

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: Policy,
    ) -> Prediction:
        ...


class Actor(Protocol):
    def act(
        self,
        state: FloatArray,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> FloatArray:
        ...


class RolloutFn(Protocol):
    def __call__(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: Policy,
    ) -> Prediction:
        ...


class Moments(NamedTuple):
    mean: jax.Array
    stddev: Optional[jax.Array] = None
