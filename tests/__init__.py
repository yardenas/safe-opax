from hydra import compose, initialize
import numpy as np

from actsafe.rl.types import Report


class DummyAgent:
    def __init__(self, action_space, config) -> None:
        self.config = config
        parallel_envs = config.training.parallel_envs
        self._policy = lambda: np.repeat(action_space.sample(), parallel_envs)
        self.replay_buffer = 123

    def __call__(self, *args, **kwargs):
        return self._policy()

    def observe(self, *args, **kwargs):
        pass

    def observe_transition(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs) -> Report:
        return Report(metrics={}, videos={})


def make_test_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../actsafe/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "writers=[stderr]",
                "+experiment=debug",
            ]
            + additional_overrides,
        )
        return cfg
