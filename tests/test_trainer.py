import pathlib
import time
import numpy as np
import pytest
from hydra import compose, initialize
from safe_opax import benchmark_suites
from safe_opax.rl.trainer import Trainer


class DummyAgent:
    def __init__(self, action_space, config) -> None:
        self.config = config
        parallel_envs = config.training.parallel_envs
        self._policy = lambda: np.repeat(action_space.sample(), parallel_envs)

    def __call__(self, *args, **kwargs):
        return self._policy()

    def observe(self, *args, **kwargs):
        pass


@pytest.fixture
def config():
    with initialize(version_base=None, config_path="../safe_opax/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "writers=[stderr]",
                "training.time_limit=32",
                "training.parallel_envs=5",
                "training.action_repeat=4",
                "training.episodes_per_epoch=1",
            ],
        )
        return cfg


@pytest.fixture
def trainer(config):
    make_env = benchmark_suites.make(config)
    dummy_env = make_env()
    with Trainer(
        config,
        make_env,
        DummyAgent(dummy_env.action_space, config),
    ) as trainer:
        yield trainer
    pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").unlink()


def test_epoch(trainer):
    trainer.train(1)
    for _ in range(5):
        if not pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").exists():
            time.sleep(1)
        else:
            break

    new_trainer = Trainer.from_pickle(
        trainer.config, f"{trainer.state_writer.log_dir}/state.pkl"
    )
    assert new_trainer.step == trainer.step
    assert new_trainer.epoch == trainer.epoch
