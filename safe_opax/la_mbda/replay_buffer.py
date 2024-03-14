from typing import Any, Iterator

import numpy as np
from safe_opax.rl.trajectory import TrajectoryData
from tensorflow import data as tfd


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        num_episodes: int,
        batch_size: int,
        num_shots: int,
        sequence_length: int,
        precision: int,
    ):
        self.task_count = 0
        self.dtype = {16: np.float16, 32: np.float32}[precision]
        self.observation = np.zeros(
            (
                capacity,
                num_episodes,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.dtype,
        )
        self.action = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.episode_ids = np.zeros((capacity,), dtype=np.int32)
        self.rs = np.random.RandomState(seed)
        example = next(
            iter(self._sample_batch(batch_size, sequence_length, num_shots, False))
        )
        self._generator = lambda: self._sample_batch(
            batch_size, sequence_length, num_shots
        )
        self._dataset = _make_dataset(self._generator, example)

    @property
    def num_episodes(self):
        return self.cost.shape[1]

    @property
    def task_id(self):
        return self.task_count % self.cost.shape[0]

    def add(self, trajectory: TrajectoryData) -> None:
        capacity, *_ = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.task_id + batch_size, capacity)
        task_slice = slice(self.task_id, end)
        current_episode = self.episode_ids[self.task_id] % self.num_episodes
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size],
                trajectory.next_observation[:batch_size, -1:],
            ],
            axis=1,
        )
        for data, val in zip(
            (self.observation, self.action, self.reward, self.cost),
            (observation, trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[task_slice, current_episode] = val[:batch_size].astype(self.dtype)
        current_episode += 1
        self.episode_ids[task_slice] = current_episode
        if current_episode == self.num_episodes:
            self.task_count += batch_size

    def _sample_batch(
        self, batch_size: int, sequence_length: int, num_shots: int, strict=True
    ) -> Iterator[tuple[Any, ...]]:
        num_episodes, time_limit = self.observation.shape[1:3]
        if strict:
            valid_tasks = self.valid_tasks
        else:
            valid_tasks = self.observation.shape[0]
        assert time_limit > sequence_length and num_episodes >= num_shots
        while True:
            timestep_ids = _make_ids(
                self.rs,
                time_limit - sequence_length - 1,
                sequence_length + 1,
                batch_size,
                (0, 1),
            )
            task_ids = self.rs.choice(valid_tasks, size=batch_size)
            highs = (
                self.episode_ids[task_ids]
                if strict
                else np.ones((batch_size,), dtype=np.int32) * num_episodes
            )
            episode_ids = self.rs.randint(0, highs, size=(batch_size, num_shots, 1))
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # value-function bootstrapping)
            take = lambda x: x[
                task_ids[:, None, None], episode_ids, timestep_ids[..., :-1]
            ]
            a = take(self.action)
            r = take(self.reward)
            c = take(self.cost)
            obs_sequence = self.observation[
                task_ids[:, None, None], episode_ids, timestep_ids
            ]
            o = obs_sequence[:, :, :-1]
            next_o = obs_sequence[:, :, 1:]
            yield o, next_o, a, r, c

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        for batch in self._dataset.take(n_batches):
            yield TrajectoryData(*map(lambda x: x.numpy(), batch))  # type: ignore

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
        return self.valid_tasks <= 1

    @property
    def valid_tasks(self):
        capacity = self.observation.shape[0]
        if self.task_count <= capacity:
            # Finds first occurence of episodes with episode_id == 0.
            # See documentation of np.argmax(). Use only one task if buffer is empty.
            valid_tasks = max((self.episode_ids == 0).argmax(), 1)
        else:
            valid_tasks = capacity
        return valid_tasks


def _make_ids(rs, low, n_samples, batch_size, dim):
    low = rs.choice(low, batch_size)
    ids_axis = np.arange(n_samples)
    ids_axis = np.expand_dims(ids_axis, axis=dim)
    dims_to_expand = list(range(1, len(dim) + 1))
    ids = np.expand_dims(low, axis=dims_to_expand) + np.repeat(
        ids_axis, batch_size, axis=0
    )
    return ids


def _make_dataset(generator, example):
    dataset = tfd.Dataset.from_generator(
        generator,
        *zip(*tuple((v.dtype, v.shape) for v in example)),
    )
    dataset = dataset.prefetch(10)
    return dataset
