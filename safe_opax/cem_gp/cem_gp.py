import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig
import gpjax as gpx

from safe_opax.rl import metrics as m
from safe_opax.cem_gp import cem
from safe_opax.rl.trajectory import TrajectoryData
from safe_opax.rl.types import FloatArray, RolloutFn
from safe_opax.rl.utils import normalize


@eqx.filter_jit
def policy(
    observation: jax.Array,
    sample: RolloutFn,
    horizon: int,
    init_guess: jax.Array,
    key: jax.random.KeyArray,
    cem_config: cem.CEMConfig,
):
    # vmap over batches of observations (e.g., solve cem separately for
    # each individual environment)
    cem_per_env = jax.vmap(
        lambda o: cem.policy(o, sample, horizon, init_guess, key, cem_config)
    )
    return cem_per_env(observation)


class CEMGP:
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        self.obs_normalizer = m.MetricsAccumulator()
        self.data = None
        self.model = None
        self.metrics_monitor = m.MetricsMonitor()
        self.plan = np.zeros(
            (config.training.parallel_envs, config.agent.plan_horizon)
            + action_space.shape
        )
        self.config = config
        self.action_space = action_space

    def __call__(self, observation: FloatArray, train: bool = False) -> FloatArray:
        normalized_obs = normalize(
            observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        horizon = self.config.agent.plan_horizon
        if self.model is not None:
            init_guess = jnp.zeros((self.plan, self.action_space.shape[-1]))
            action = policy(
                normalized_obs,
                self.model.sample,
                horizon,
                init_guess,
                next(self.prng),
                self.config.agent.cem,
            )
            self.plan = np.asarray(action)
        else:
            # TODO (yarden): make this nicer (uniform with scale as parameter)
            return np.repeat(
                self.action_space.sample()[None], self.config.training.parallel_envs
            ) * self.config.agent.initial_action_scale
        return self.plan[:, 0]

    def observe(self, trajectory: TrajectoryData) -> None:
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]],
                axis=1,
            ),
            axis=(0, 1),
        )
        new_data = _prepare_data(trajectory, self.obs_normalizer)
        if self.data is None:
            self.data = gpx.Dataset(*new_data)
        else:
            self.data += gpx.Dataset(*new_data)


def _prepare_data(trajectory: TrajectoryData, normalizer):
    results = normalizer.result
    normalize_fn = lambda x: normalize(x, results.mean, results.std)
    normalized_obs = normalize_fn(trajectory.observation)
    normalized_next_obs = normalize_fn(trajectory.next_observation)
    x = np.concatenate([normalized_obs, trajectory.action], axis=-1)
    y = normalized_next_obs
    return x, y
