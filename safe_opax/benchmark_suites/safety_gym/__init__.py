from gymnasium.wrappers.compatibility import EnvCompatibility
from omegaconf import DictConfig
import numpy as np

from safe_opax.benchmark_suites.utils import get_domain_and_task
from safe_opax.rl.types import EnvironmentFactory
from safe_opax.rl.wrappers import ImageObservation




class SafetyGymCompatibility(EnvCompatibility):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset(), {}

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import safety_gym  # noqa: F401
        import gym

        _, task_cfg = get_domain_and_task(cfg)
        task_name = task_cfg.safety_gym.task
        robot_name = task_cfg.safety_gym.robot_name
        env = gym.make(f"Safexp-{robot_name}{task_name}-v0")
        env = SafetyGymCompatibility(env)
        # Turning manually on the 'observe_vision' flag so a rendering context gets
        # opened and
        # all object types rendering is on (L.302, safety_gym.world.py).
        env.unwrapped.vision_size = (64, 64)
        env.unwrapped.observe_vision = True
        env.unwrapped.vision_render = False
        obs_vision_swap = env.unwrapped.obs_vision

        # Making rendering within obs() function (in safety_gym) not actually
        # render the scene on
        # default so that rendering only occur upon calling to 'render()'.
        from PIL import ImageOps
        from PIL import Image

        def render_obs(fake=True):
            if fake:
                return np.ones(())
            else:
                image = Image.fromarray(
                    np.array(obs_vision_swap() * 255, dtype=np.uint8, copy=False)
                )
                image = np.asarray(ImageOps.flip(image))
                return image

        env.unwrapped.obs_vision = render_obs

        def safety_gym_render(**kwargs):
            mode = kwargs.get("mode", "human")
            if mode in ["human", "rgb_array"]:
                # Use regular rendering
                return env.unwrapped.render(mode, camera_id=3, **kwargs)
            elif mode == "vision":
                return render_obs(fake=False)
            else:
                raise NotImplementedError

        env.render = safety_gym_render
        env = ImageObservation(
            env,
            task_cfg.image_observation.image_size,
            task_cfg.image_observation.image_format,
            render_kwargs={"mode": "vision"},
        )
        return env

    return make_env  # type: ignore
