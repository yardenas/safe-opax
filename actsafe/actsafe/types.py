from typing import (
    NamedTuple,
    Optional,
    Protocol,
)

import jax

from actsafe.rl.types import FloatArray


class Actor(Protocol):
    def act(
        self,
        state: FloatArray,
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> FloatArray:
        ...


class Moments(NamedTuple):
    mean: jax.Array
    stddev: Optional[jax.Array] = None
