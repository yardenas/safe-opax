from gymnasium.envs import register
from omegaconf import DictConfig

from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ImageObservation

from .env import ROBOTS, TASKS

for robot in ROBOTS:
    if robot == "g1" or robot == "digit":
        control = "torque"
    else:
        control = "pos"
    for task, task_info in TASKS.items():
        task_info = task_info()
        kwargs = task_info.kwargs.copy()
        kwargs["robot"] = robot
        kwargs["control"] = control
        kwargs["task"] = task
        register(
            id=f"{robot}-{task}-v0",
            entry_point="humanoid_bench.env:HumanoidEnv",
            max_episode_steps=task_info.max_episode_steps,
            kwargs=kwargs,
        )


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import gymnasium as gym

        domain_name, task_cfg = get_domain_and_task(cfg)
        env_name = "h1hand-pole-v0"
        env = gym.make(env_name)
        if task_cfg.image_observation.enabled:
            env = ImageObservation(
                env,
                task_cfg.image_observation.image_size,
                task_cfg.image_observation.image_format,
                render_kwargs={
                    "visualize_reward": task_cfg.image_observation.visualize_reward,
                    "camera_id": 0,
                },
            )
        else:
            from gymnasium.wrappers.flatten_observation import FlattenObservation

            env = FlattenObservation(env)
        return env

    return make_env
