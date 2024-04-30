import pathlib
from unittest.mock import patch
import pytest
from safe_opax.rl.trainer import UnsupervisedTrainer
from tests import DummyAgent, make_test_config
from safe_opax import benchmark_suites


@pytest.fixture
def config():
    cfg = make_test_config(
        [
            "training.action_repeat=4",
            "environment=safe_adaptation_gym",
            "environment.safe_adaptation_gym.image_observation.enabled=false",
            "training.exploration_steps=100",
        ]
    )
    return cfg


@pytest.fixture
def trainer(config):
    make_env = benchmark_suites.make(config)
    dummy_env = make_env()
    trainer = UnsupervisedTrainer(
        config,
        make_env,
        DummyAgent(dummy_env.action_space, config),
    )
    yield trainer
    assert trainer.state_writer is not None
    pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").unlink()


def test_epoch(trainer):
    with trainer as trainer:
        with patch.object(trainer.env, "reset", wraps=trainer.env.reset) as mock:
            trainer.train(1)
    assert mock.call_count == 4
