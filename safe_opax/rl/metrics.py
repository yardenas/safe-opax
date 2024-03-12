from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


@dataclass
class Metrics:
    mean: npt.ArrayLike
    var: npt.ArrayLike
    min: npt.ArrayLike
    max: npt.ArrayLike

    @property
    def std(self) -> npt.NDArray[Any]:
        return np.sqrt(self.var)


class MetricsAccumulator:
    def __init__(self):
        self._state = Metrics(
            np.zeros((1,)),
            np.ones((1,)),
            np.empty((1,)),
            np.empty((1,)),
        )
        self._count = 0
        self._m2 = 0.0

    def update_state(
        self, sample: float | npt.NDArray[Any], axis: int | Sequence[int] = 0
    ) -> None:
        if isinstance(sample, float) or sample.ndim == 0:
            sample = np.array(
                [
                    sample,
                ]
            )
        if isinstance(axis, int) and not isinstance(sample, float):
            batch_count = sample.shape[axis]
        elif isinstance(axis, Sequence) and not isinstance(sample, float):
            batch_count = int(np.prod([sample.shape[i] for i in axis]))
        else:
            batch_count = 1
        batch_mean = sample.mean(axis=axis)
        batch_var = sample.var(axis=axis)
        batch_min = sample.min(axis=axis)
        batch_max = sample.max(axis=axis)
        delta = batch_mean - self._state.mean
        new_mean = self._state.mean + delta * batch_count / (self._count + batch_count)
        m_a = self._state.var * self._count  # type: ignore
        m_b = batch_var * batch_count
        self._m2 = (
            m_a
            + m_b
            + np.square(delta) * self._count * batch_count / (self._count + batch_count)
        )
        new_var = self._m2 / (self._count + batch_count)
        self._count += batch_count
        self._state = Metrics(
            new_mean,
            new_var,
            np.minimum(self._state.min if self._count > 1 else batch_min, batch_min),
            np.maximum(self._state.max if self._count > 1 else batch_max, batch_max),
        )

    @property
    def result(self):
        return self._state

    def reset_states(self):
        self._state = Metrics(
            np.zeros((1,)),
            np.ones((1,)),
            np.zeros((1,)),
            np.zeros((1,)),
        )
        self._count = 0
        self._m2 = 0.0
