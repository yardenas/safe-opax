import numpy as np
from PIL import Image
from gymnasium import ObservationWrapper
from gymnasium.core import Wrapper
from gymnasium.spaces import Box


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        while current_step < self.repeat and not done:
            obs, reward, terminal, truncated, info = self.env.step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            current_step += 1
            done = truncated or terminal
        info["steps"] = current_step
        info["cost"] = total_cost
        return obs, total_reward, terminal, truncated, info


class RenderedObservation(ObservationWrapper):
    def __init__(self, env, image_size, render_kwargs):
        super(RenderedObservation, self).__init__(env)
        self.observation_space = Box(0, 255, image_size + (3,), np.float32)
        self._render_kwargs = render_kwargs
        self.image_size = image_size

    def observation(self, _):
        image = self.env.render(**self._render_kwargs)
        image = Image.fromarray(image)
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image, copy=False)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image
