import numpy as np
import pytest
from hydra import compose, initialize
from safe_opax import benchmark_suites
from safe_opax.rl.trainer import Trainer


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
        lambda *args, **kwargs: np.repeat(
            dummy_env.action_space.sample(), config.training.parallel_envs
        ),
    ) as trainer:
        yield trainer


def test_epoch(trainer):
    trainer.train(1)
    new_trainer = Trainer.from_pickle(trainer.config, trainer.state_writer.log_dir)
    assert new_trainer.step == trainer.step
    assert new_trainer.epoch == trainer.epoch
    assert new_trainer.seeds == trainer.seeds
