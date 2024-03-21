import pathlib
import time
import numpy as np
import pytest
from tests import make_test_config
from safe_opax.rl.trainer import Trainer
from safe_opax.rl.types import Report


class DummyAgent:
    def __init__(self, action_space, config) -> None:
        self.config = config
        parallel_envs = config.training.parallel_envs
        self._policy = lambda: np.repeat(action_space.sample(), parallel_envs)

    def __call__(self, *args, **kwargs):
        return self._policy()

    def observe(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs) -> Report:
        return Report(metrics={}, videos={})


@pytest.fixture
def config():
    cfg = make_test_config(
        [
            "action_repeat=4",
            "environment.dm_cartpole.image_observation.enabled=false",
        ]
    )
    return cfg


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
    old_trainer_summary = trainer._run_training_epoch(1)
    assert old_trainer_summary.metrics == new_trainer_summary.metrics
