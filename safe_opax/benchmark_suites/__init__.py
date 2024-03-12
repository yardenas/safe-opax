from omegaconf import DictConfig

from safe_opax.benchmark_suites.dm_control import ENVIRONMENTS as dm_control_envs
from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory




def make(cfg: DictConfig) -> EnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, _ = get_domain_and_task(cfg)
    if domain_name in dm_control_envs:
        from safe_opax.benchmark_suites.dm_control import make

        make_env = make(cfg)
    return make_env
