import os
import numpy as np
from gymnasium import RewardWrapper
from omegaconf import DictConfig
from gymnasium.spaces import Box

from actsafe.benchmark_suites.utils import get_domain_and_task
from actsafe.rl.types import EnvironmentFactory
from actsafe.rl.wrappers import ImageObservation

class ConstraintWrapper(RewardWrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        observation, reward, terminal, truncated, info = self.env.step(action)
        small_control = info.get("small_control", 0)
        stand_reward = info.get("stand_reward", 0)
        move = info.get("move", 0)
        reward = (
            0.5 * (small_control * stand_reward) + 0.5 * move
        )
        collision_discount = info.get("collision_discount", 0.)
        terminated = info.get("terminated", False)
        info["cost"] = collision_discount < 1. or terminated
        return observation, reward, terminal, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class HumanoidImageObservation(ImageObservation):
    def __init__(self, env, image_size, image_format="channels_first", *, render_kwargs=None):
        super().__init__(env, image_size, image_format, render_kwargs=render_kwargs)
        size = image_size + (6,) if image_format == "chw" else (6,) + image_size
        self.observation_space = Box(0, 255, size, np.float32)

    def observation(self, observation):
        third_person = super().observation(observation)
        left = observation["image_left_eye"]
        left = self.preprocess(left)
        return np.concatenate([third_person, left], axis=0)

def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        from .env import HumanoidEnv

        _, task_cfg = get_domain_and_task(cfg)
        reach_data_path = os.path.join(os.path.dirname(__file__), "data", "reach_one_hand")
        robot, task = task_cfg.task.split("-", 1)
        env = HumanoidEnv(robot=robot,
                        control="pos",
                        task=task,
                        policy_type="reach_single",
                        policy_path=reach_data_path + "/model.ckpt",
                        mean_path=reach_data_path + "/mean.npy",
                        var_path=reach_data_path + "/var.npy",
                        sensors="image",
                        obs_wrapper="true",
                        )
        env = ConstraintWrapper(env)
        if task_cfg.image_observation.enabled:
            env = ImageObservation(
                env,
                task_cfg.image_observation.image_size,
                task_cfg.image_observation.image_format,
            )
        else:
            from gymnasium.wrappers.flatten_observation import FlattenObservation

            env = FlattenObservation(env)
        return env
    return make_env
