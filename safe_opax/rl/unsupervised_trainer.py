import logging

from omegaconf import DictConfig

from safe_adaptation_gym.benchmark import TASKS
from safe_opax.benchmark_suites.safe_adaptation_gym import sample_task
from safe_opax.rl.epoch_summary import EpochSummary
from safe_opax.rl.trainer import Trainer
from safe_opax.rl.types import Agent, EnvironmentFactory
from safe_opax.rl.utils import PRNGSequence

_LOG = logging.getLogger(__name__)


class UnsupervisedTrainer(Trainer):
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: Agent | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        super().__init__(config, make_env, agent, start_epoch, step, seeds)

    def __enter__(self):
        super().__enter__()
        self.env.reset(
            options={
                "task": [
                    TASKS["unsupervised"]()
                    for _ in range(self.config.training.parallel_envs)
                ]
            }
        )
        return self

    def _run_training_epoch(
        self, episodes_per_epoch: int
    ) -> tuple[EpochSummary, float, int]:
        outs = super()._run_training_epoch(episodes_per_epoch)
        if self.step >= self.config.training.exploration_steps:
            task_name = sample_task(self.config.training.seed + self.step)
            _LOG.info(f"Exploration complete. Changing to task {task_name}")
            tasks = [
                TASKS[task_name.lower()]()
                for _ in range(self.config.training.parallel_envs)
            ]
            assert self.env is not None
            self.env.reset(options={"task": tasks})
        return outs
