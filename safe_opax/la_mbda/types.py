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
    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> Prediction:
        ...


class Actor(Protocol):
    def act(
        self,
        state: FloatArray,
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> FloatArray:
        ...


class RolloutFn(Protocol):
    def __call__(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> Prediction:
        ...


class Moments(NamedTuple):
    mean: jax.Array
    stddev: Optional[jax.Array] = None
