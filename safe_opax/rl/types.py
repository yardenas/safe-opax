from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    NamedTuple,
    Protocol,
    Union,
)

import jax
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import typing as npt
from omegaconf import DictConfig

from safe_opax.rl.epoch_summary import EpochSummary
from safe_opax.rl.trajectory import TrajectoryData

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

Policy = Union[Callable[[jax.Array, jax.Array | None], jax.Array], jax.Array]


@dataclass
class Report:
    metrics: dict[str, float]
    videos: dict[str, npt.ArrayLike] = field(default_factory=dict)


class Agent(Protocol):
    config: DictConfig

    def __call__(self, observation: FloatArray, train: bool = False) -> FloatArray:
        ...

    def observe(self, trajectory: TrajectoryData) -> None:
        ...

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        ...


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
    ) -> tuple[Prediction, Any]:
        ...


class RolloutFn(Protocol):
    def __call__(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, Any]:
        ...


class ShiftScale(NamedTuple):
    shift: jax.Array
    scale: jax.Array
