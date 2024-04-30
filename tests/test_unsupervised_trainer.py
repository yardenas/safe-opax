import pathlib
from unittest.mock import patch
import pytest
from tests import DummyAgent, make_test_config
from safe_opax.rl.trainer import Trainer
from safe_opax import benchmark_suites
from safe_adaptation_gym.benchmark import TASKS


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
    trainer = Trainer(
        config,
        make_env,
        DummyAgent(dummy_env.action_space, config),
    )
    yield trainer
    assert trainer.state_writer is not None
    pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").unlink()


def test_epoch(trainer):
    with patch.object(trainer.env, "reset", wraps=trainer.env.reset) as mock:
        with trainer as trainer:
            mock.assert_called_once_with(options={"task": TASKS["unsupervised"]()})
            trainer.train(1)
    # Should be called twice:
    # First time upon initialization, second time when transitioning to
    # task exploitation.
    assert mock.call_count == 2
