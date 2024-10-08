import logging
import os
import time
from typing import Optional

import cloudpickle
from omegaconf import DictConfig
import numpy as np

from actsafe import benchmark_suites
from actsafe.actsafe.actsafe import ActSafe
from actsafe.rl import acting, episodic_async_env
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.logging import StateWriter, TrainingLogger
from actsafe.rl.types import EnvironmentFactory
from actsafe.rl.utils import PRNGSequence

from safe_adaptation_gym.benchmark import TASKS
from safe_adaptation_gym.tasks import Task

_LOG = logging.getLogger(__name__)

_TRAINING_STATE = "state.pkl"


def get_trainer(name):
    trainers = {"rl": Trainer, "unsupervised": UnsupervisedTrainer}
    if name in trainers:
        return trainers[name]
    else:
        raise NotImplementedError(f"Unknown trainer type: {name}")


def get_state_path() -> str:
    log_path = os.getcwd()
    state_path = os.path.join(log_path, _TRAINING_STATE)
    return state_path


def should_resume(state_path: str) -> bool:
    return os.path.exists(state_path)


def start_fresh(
    cfg: DictConfig,
) -> "Trainer":
    make_env = benchmark_suites.make(cfg)
    return get_trainer(cfg.training.trainer)(cfg, make_env)


def load_state(cfg, state_path) -> "Trainer":
    return get_trainer(cfg.training.trainer).from_pickle(cfg, state_path)


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: ActSafe | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        self.config = config
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: TrainingLogger | None = None
        self.state_writer: StateWriter | None = None
        self.env: episodic_async_env.EpisodicAsync | None = None
        self.agent = agent

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = TrainingLogger(self.config)
        self.state_writer = StateWriter(log_path, _TRAINING_STATE)
        self.env = episodic_async_env.EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        if self.seeds is None:
            self.seeds = PRNGSequence(self.config.training.seed)
        if self.agent is None:
            self.agent = self.make_agent()
        return self

    def make_agent(self) -> ActSafe:
        assert self.env is not None
        if self.config.agent.name == "actsafe":
            agent = ActSafe(
                self.env.observation_space,
                self.env.action_space,
                self.config,
            )
        else:
            raise NotImplementedError(f"Unknown agent type: {self.config.agent.name}")
        return agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer, agent = (
            self.epoch,
            self.logger,
            self.state_writer,
            self.agent,
        )
        assert logger is not None and state_writer is not None and agent is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            _LOG.info(f"Training epoch #{epoch}")
            summary, wall_time, steps = self._run_training_epoch(
                self.config.training.episodes_per_epoch
            )
            objective, cost_return, feasibilty = summary.metrics
            if isinstance(objective, np.ndarray):
                metrics = {
                    f"train/objective_{i}": val for i, val in enumerate(objective)
                }
            else:
                metrics = {"train/objective": objective}
            metrics |= {
                "train/cost_return": cost_return,
                "train/feasibility": feasibilty,
                "train/fps": steps / wall_time,
            }
            report = agent.report(summary, epoch, self.step)
            report.metrics.update(metrics)
            if (maybe_videos := summary.videos) is not None:
                report.videos.update({"train/video": maybe_videos})
            logger.log(report.metrics, self.step)
            for k, v in report.videos.items():
                logger.log_video(v, self.step, k)
            self.epoch = epoch + 1
            state_writer.write(self.state)

    def _run_training_epoch(
        self,
        episodes_per_epoch: int,
    ) -> tuple[EpochSummary, float, int]:
        agent, env, logger, seeds = self.agent, self.env, self.logger, self.seeds
        assert (
            env is not None
            and agent is not None
            and logger is not None
            and seeds is not None
        )
        start_time = time.time()
        env.reset(seed=int(next(seeds)[0].item()))
        summary, step = acting.epoch(
            agent,
            env,
            episodes_per_epoch,
            True,
            self.step,
            self.config.training.render_episodes,
        )
        steps = step - self.step
        self.step = step
        next(seeds)
        end_time = time.time()
        wall_time = end_time - start_time
        return summary, wall_time, steps

    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "Trainer":
        with open(state_path, "rb") as f:
            make_env, seeds, agent, epoch, step = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        _LOG.info(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=seeds,
            agent=agent,
            step=step,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "seeds": self.seeds,
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }


class UnsupervisedTrainer(Trainer):
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: ActSafe | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        super().__init__(config, make_env, agent, start_epoch, step, seeds)
        self.test_task_name = self.config.training.test_task_name
        self.train_task_name = self.config.training.train_task_name
        # After a few iterations, we realized `test_tasks` are not useful, as we just use multiple rewards.
        # just ignore this.
        self.test_tasks: list[Task] | None = None

    def __enter__(self):
        super().__enter__()
        self.env.reset(
            options={
                "task": [
                    get_task(self.train_task_name)
                    for _ in range(self.config.training.parallel_envs)
                ]
            }
        )
        if (
            self.step >= self.config.training.exploration_steps
            and self.test_tasks is None
        ):
            self.test_tasks = [
                get_task(self.test_task_name)
                for _ in range(self.config.training.parallel_envs)
            ]
            assert self.env is not None
            self.env.reset(options={"task": self.test_tasks})
        return self

    def _run_training_epoch(
        self, episodes_per_epoch: int
    ) -> tuple[EpochSummary, float, int]:
        outs = super()._run_training_epoch(episodes_per_epoch)
        if (
            self.step >= self.config.training.exploration_steps
            and self.test_tasks is None
        ):
            _LOG.info(f"Exploration complete. Changing to task {self.test_task_name}")
            self.test_tasks = [
                get_task(self.test_task_name)
                for _ in range(self.config.training.parallel_envs)
            ]
            assert self.env is not None
            self.env.reset(options={"task": self.test_tasks})
            assert self.agent is not None
            new_agent = self.make_agent()
            self.agent.replay_buffer = new_agent.replay_buffer
        return outs


def get_task(task_name: str) -> Task:
    # Handles the interface difference between
    # cartpole unsupervised and safe-adaptation-gym
    if task_name in TASKS:
        return TASKS[task_name.lower()]()
    elif task_name == "swingup" or task_name == "keepdown":
        return task_name
    else:
        raise ValueError(f"Unknown task name: {task_name}")
