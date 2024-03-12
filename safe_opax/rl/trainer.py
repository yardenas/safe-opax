import logging
import os
from typing import List, Optional

import cloudpickle
import numpy as np
from omegaconf import DictConfig

from safe_opax.rl import acting, agents, episodic_async_env
from safe_opax.rl import logging as rllogging
from safe_opax.rl.types import Agent, EnvironmentFactory

_LOG = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: Optional[Agent] = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: Optional[List[int]] = None,
    ):
        self.config = config
        self.agent = agent
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: Optional[rllogging.TrainingLogger] = None
        self.state_writer: Optional[rllogging.StateWriter] = None
        self.env: Optional[episodic_async_env.EpisodicAsync] = None

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = rllogging.TrainingLogger(log_path)
        self.state_writer = rllogging.StateWriter(log_path)
        self.env = episodic_async_env.EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        if self.seeds is None:
            self.seeds = self.config.training.seed
        self.env.reset(seed=self.seeds)
        if self.agent is None:
            self.agent = agents.make(
                self.env.observation_space,
                self.env.action_space,
                self.config,
                self.logger,
            )
        else:
            self.agent.logger = self.logger
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.write(self.state)
        self.state_writer.close()
        self.logger.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
        assert logger is not None and state_writer is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            _LOG.info(f"Training epoch #{epoch}")
            self._run_training_epoch(
                train=True,
                episodes_per_task=self.config.training.episodes_per_task,
                prefix="train",
            )
            self.epoch = epoch + 1
            state_writer.write(self.state)
        logger.flush()

    def _run_training_epoch(
        self,
        episodes_per_task: int,
        prefix: str,
    ) -> None:
        agent, env, logger = self.agent, self.env, self.logger
        assert env is not None and agent is not None and logger is not None
        summary, step = acting.epoch(
            agent,
            env,
            episodes_per_task,
            True,
            self.step,
        )
        objective, cost_rate, feasibilty = summary.metrics
        logger.log(
            {
                f"{prefix}/objective": objective,
                f"{prefix}/cost_rate": cost_rate,
                f"{prefix}/feasibility": feasibilty,
            },
            self.step,
        )
        self.step = step

    def get_env_random_state(self):
        assert self.env is not None
        rs = [
            state.get_state()[1]
            for state in self.env.get_attr("rs")
            if state is not None
        ]
        if not rs:
            rs = [
                _infer_and_extract_state(state)
                for state in self.env.get_attr("np_random")
            ]
        return rs

    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "Trainer":
        with open(state_path, "rb") as f:
            make_env, env_rs, agent, epoch, step = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        _LOG.info(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=env_rs,
            agent=agent,
            step=step,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "env_rs": self.get_env_random_state(),
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }


def _infer_and_extract_state(state):
    if isinstance(state, np.random.RandomState):
        return state.get_state()[1]
    else:
        return state.bit_generator.state["state"]["state"]
