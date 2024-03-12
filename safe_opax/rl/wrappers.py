from gymnasium.core import Wrapper


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


class MetaEnv(Wrapper):
    def __init__(self, env, alter_env_fn):
        self.env = env
        self.alter_env_fn = alter_env_fn

    def reset(self, *, seed=None, options=None):
        if options is not None and "task" in options:
            self.alter_env_fn(self.env, options["task"])
        return super().reset(seed=seed, options=options)
