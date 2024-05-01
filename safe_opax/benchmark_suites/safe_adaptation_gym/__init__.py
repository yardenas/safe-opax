from omegaconf import DictConfig
import numpy as np
from gymnasium.wrappers.compatibility import EnvCompatibility
from safe_opax.benchmark_suites.utils import get_domain_and_task

from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ChannelFirst


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import safe_adaptation_gym

        easy_tasks = (
            "go_to_goal",
            "push_box",
            "collect",
            "dribble_ball",
            "catch_goal",
            "press_buttons",
        )
        task = np.random.RandomState(cfg.training.seed).choice(easy_tasks)
        _, task_cfg = get_domain_and_task(cfg)
        env = safe_adaptation_gym.make(
            robot_name=task_cfg.robot_name,
            task_name=task,
            seed=cfg.training.seed,
            rgb_observation=task_cfg.image_observation.enabled,
            render_lidar_and_collision=not task_cfg.image_observation.enabled,
        )
        env = EnvCompatibility(env)
        if (
            task_cfg.image_observation.enabled
            and task_cfg.image_observation.image_format == "channels_first"
        ):
            env = ChannelFirst(env)
        return env

    return make_env  # type: ignore
