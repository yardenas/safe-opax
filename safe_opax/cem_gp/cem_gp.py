import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig
import gpjax as gpx

from safe_opax.cem_gp.gp_model import GPModel
from safe_opax.rl import metrics as m
from safe_opax.cem_gp import cem
from safe_opax.rl.epoch_summary import EpochSummary
from safe_opax.rl.trajectory import TrajectoryData, Transition
from safe_opax.rl.types import FloatArray, Report, RolloutFn
from safe_opax.rl.utils import PRNGSequence, normalize
from safe_opax.cem_gp.rewards import tolerance


@eqx.filter_jit
def policy(
    observation: jax.Array,
    sample: RolloutFn,
    horizon: int,
    init_guess: jax.Array,
    key: jax.Array,
    cem_config: cem.CEMConfig,
):
    # vmap over batches of observations (e.g., solve cem separately for
    # each individual environment)
    cem_per_env = jax.vmap(
        lambda o, i: cem.policy(o, sample, horizon, i, key, cem_config)
    )
    return cem_per_env(observation, init_guess)


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
        self.prng = PRNGSequence(config.training.seed)

    def __call__(self, observation: FloatArray, train: bool = False) -> FloatArray:
        normalized_obs = normalize(
            observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        horizon = self.config.agent.plan_horizon
        if self.model is not None:
            init_guess = self.plan
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
            return (
                np.tile(
                    self.action_space.sample()[:, None],
                    (self.config.training.parallel_envs, 1),
                )
                * self.config.agent.initial_action_scale
            )
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
        self.model = GPModel(
            self.data.X,  # type: ignore
            self.data.y,  # type: ignore
            cartpole_reward,
            lambda obs: cartpole_cost(
                obs, self.config.environment.dm_cartpole.slider_position_bound
            ),
        )

    def observe_transition(self, transition: Transition) -> None:
        pass

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        return Report({})


def _prepare_data(trajectory: TrajectoryData, normalizer):
    results = normalizer.result
    normalize_fn = lambda x: normalize(x, results.mean, results.std)
    normalized_obs = normalize_fn(trajectory.observation)
    normalized_next_obs = normalize_fn(trajectory.next_observation)
    x = np.concatenate([normalized_obs, trajectory.action], axis=-1)
    y = normalized_next_obs
    flat = lambda x: x.reshape((-1, x.shape[-1]))
    return flat(x), flat(y)


def cartpole_reward(observation):
    cart_position = observation[..., 0]
    cart_in_bounds = tolerance(cart_position, (-0.25, 0.25))
    pole_angle_cosine = observation[..., 1]
    angle_in_bounds = tolerance(pole_angle_cosine, (0.995, 1)).prod()
    return angle_in_bounds * cart_in_bounds


def cartpole_cost(observation, slider_position_bound):
    cart_position = observation[..., 0]
    return jnp.where(jnp.abs(cart_position) >= slider_position_bound)
