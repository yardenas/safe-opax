from omegaconf import DictConfig

from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ImageObservation




def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        from .env import HumanoidEnv

        _, task_cfg = get_domain_and_task(cfg)
        env_name = "h1hand-pole-v0"
        env = HumanoidEnv(robot="h1hand", control="reach", task="pole")
        if task_cfg.image_observation.enabled:
            env = ImageObservation(
                env,
                task_cfg.image_observation.image_size,
                task_cfg.image_observation.image_format
            )
        else:
            from gymnasium.wrappers.flatten_observation import FlattenObservation

            env = FlattenObservation(env)
        return env
    return make_env
