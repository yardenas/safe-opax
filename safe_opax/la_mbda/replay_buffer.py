from typing import Iterator
import numpy as np

from tensorflow import data as tfd
from safe_opax.rl.trajectory import TrajectoryData


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        batch_size: int,
        sequence_length: int,
    ):
        self.episode_id = 0
        self.dtype = np.float32
        self.obs_dtype = np.uint8
        self.observation = np.zeros(
            (
                capacity,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.obs_dtype,
        )
        self.action = np.zeros(
            (
                capacity,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (
                capacity,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                capacity,
                max_length,
            ),
            dtype=self.dtype,
        )
        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        example = next(iter(self._sample_batch(batch_size, sequence_length, capacity)))
        self._generator = lambda: self._sample_batch(batch_size, sequence_length)
        self._dataset = _make_dataset(self._generator, example)

    def add(self, trajectory: TrajectoryData):
        capacity, episode_length = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.episode_id + batch_size, capacity)
        episode_slice = slice(self.episode_id, end)
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[episode_slice] = val[:batch_size].astype(self.dtype)
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size],
                trajectory.next_observation[:batch_size, -1:],
            ],
            axis=1,
        )
        self.observation[episode_slice] = observation.astype(self.obs_dtype)
        self.episode_id = (self.episode_id + batch_size) % capacity
        self._valid_episodes = min(self._valid_episodes + batch_size, capacity)

    def _sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        valid_episodes: int | None = None,
    ):
        if valid_episodes is not None:
            valid_episodes = valid_episodes
        else:
            valid_episodes = self._valid_episodes
        time_limit = self.observation.shape[1]
        assert time_limit > sequence_length
        while True:
            low = self.rs.choice(time_limit - sequence_length - 1, batch_size)
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # bootstrapping)
            a, r, c = [
                x[episode_ids[:, None], timestep_ids[:, :-1]]
                for x in (
                    self.action,
                    self.reward,
                    self.cost,
                )
            ]
            o = self.observation[episode_ids[:, None], timestep_ids]
            if self.obs_dtype == np.uint8:
                o = preprocess(o).astype(self.dtype)
            yield o, a, r, c

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        if self.empty:
            return
        for batch in self._dataset.take(n_batches):
            yield TrajectoryData(*map(lambda x: x.numpy(), batch))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_dataset"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        example = next(iter(self._generator()))
        self._dataset = _make_dataset(self._generator, example)

    @property
    def empty(self):
        return self._valid_episodes == 0


def preprocess(image):
    return image / 255.0 - 0.5


def _make_dataset(generator, example):
    dataset = tfd.Dataset.from_generator(
        generator,
        *zip(*tuple((v.dtype, v.shape) for v in example)),
    )
    dataset = dataset.prefetch(10)
    return dataset
