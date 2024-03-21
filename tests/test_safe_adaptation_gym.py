import pytest

from tests import make_test_config

from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym
from safe_opax.benchmark_suites import make


@pytest.fixture
def config():
    cfg = make_test_config(
        [
            "environment=safe_adaptation_gym",
            "environment.safe_adaptation_gym.image_observation.enabled=false",
        ],
    )
    return cfg


def test_make(config):
    env = make(config)()
    assert isinstance(env, SafeAdaptationGym)
    outs = env.step(env.action_space.sample())  # type: ignore
    assert len(outs) == 4
