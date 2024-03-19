from omegaconf import DictConfig
from safe_opax.benchmark_suites.utils import get_domain_and_task

from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ImageObservation


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        from gymnasium.wrappers.flatten_observation import FlattenObservation
        import safe_adaptation_gym

        domain_name, task_cfg = get_domain_and_task(cfg)
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
            env = FlattenObservation(env)
        return env

    return make_env
