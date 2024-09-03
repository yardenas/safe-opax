import os
import numpy as np
from gymnasium import RewardWrapper
from omegaconf import DictConfig

from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ImageObservation

class ConstraintWrapper(RewardWrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        observation, reward, terminal, truncated, info = self.env.step(action)
        small_control = info["small_control"]
        stand_reward = info["stand_reward"]
        move = info["move"]
        reward = (
            0.5 * (small_control * stand_reward) + 0.5 * move
        )
        collision_discount = info["collision_discount"]
        info["cost"] = collision_discount < 1.
        return observation, reward, terminal, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class HumanoidImageObservation(ImageObservation):
    def __init__(self, env, image_size, image_format="channels_first"):
        super().__init__(env, image_size, image_format)

    def observation(self, observation):
        third_person = super().observation(observation)
        left = observation["image_left_eye"]
        left = self.preprocess(left)
        return np.concatenate([third_person, left], axis=-1)

def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        from .env import HumanoidEnv

        _, task_cfg = get_domain_and_task(cfg)
        env_name = "h1hand-pole-v0"
        reach_data_path = os.path.join(os.path.dirname(__file__), "data", "reach_one_hand")
        env = HumanoidEnv(robot="h1hand",
                        control="pos",
                        task="pole",
                        policy_type="reach_single",
                        policy_path=reach_data_path + "/model.ckpt",
                        mean_path=reach_data_path + "/mean.npy",
                        var_path=reach_data_path + "/var.npy",
                        sensors="image",
                        obs_wrapper="true",
                        )
        env = ConstraintWrapper(env)
        if task_cfg.image_observation.enabled:
            env = HumanoidImageObservation(
                env,
                task_cfg.image_observation.image_size,
                task_cfg.image_observation.image_format
            )
        else:
            from gymnasium.wrappers.flatten_observation import FlattenObservation

            env = FlattenObservation(env)
        return env
    return make_env
