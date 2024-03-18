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

    def log(self, *args, **kwargs):
        return {}


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
                "environment.dm_cartpole.image_observation.enabled=false",
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
    assert trainer.state_writer is not None
    pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").unlink()


def test_epoch(trainer):
    trainer.train(1)
    wait_count = 10
    while wait_count > 0:
        time.sleep(0.5)
        if not pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").exists():
            wait_count -= 1
            if wait_count == 0:
                pytest.fail("state file was not written")
        else:
            break
    new_trainer = Trainer.from_pickle(
        trainer.config, f"{trainer.state_writer.log_dir}/state.pkl"
    )
    assert new_trainer.step == trainer.step
    assert new_trainer.epoch == trainer.epoch
    assert new_trainer.seeds is not None
    assert (new_trainer.seeds.key == trainer.seeds.key).all()
    with new_trainer as new_trainer:
        new_trainer_summary = new_trainer._run_training_epoch(1)
    old_trainer_summary = trainer._run_training_epoch(1, "train")
    assert old_trainer_summary.metrics == new_trainer_summary.metrics
