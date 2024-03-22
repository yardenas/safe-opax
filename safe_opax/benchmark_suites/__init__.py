from omegaconf import DictConfig

from safe_opax.benchmark_suites.dm_control import ENVIRONMENTS as dm_control_envs
from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory


def make(cfg: DictConfig) -> EnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, task_config = get_domain_and_task(cfg)
    if "task" in task_config and (domain_name, task_config.task) in dm_control_envs:
        from safe_opax.benchmark_suites.dm_control import make

        make_env = make(cfg)
    elif domain_name == "safe_adaptation_gym":
        from safe_opax.benchmark_suites.safe_adaptation_gym import make

        make_env = make(cfg)
    else:
        raise NotImplementedError(f"Environment {domain_name} not implemented")
    return make_env
