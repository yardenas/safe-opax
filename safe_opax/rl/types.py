from typing import (
    Callable,
    Protocol,
    Union,
)

import jax
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import typing as npt
from omegaconf import DictConfig

from safe_opax.rl.trajectory import Transition

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

Policy = Union[Callable[[jax.Array], jax.Array], jax.Array]


class Agent(Protocol):
    config: DictConfig

    def __call__(self, observation: FloatArray, train: bool = False) -> FloatArray:
        ...

    def observe(self, transition: Transition) -> None:
        ...
