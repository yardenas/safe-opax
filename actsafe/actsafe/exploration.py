import jax
from omegaconf import DictConfig

from actsafe.actsafe.opax_bridge import OpaxBridge
from actsafe.actsafe.make_actor_critic import make_actor_critic
from actsafe.actsafe.sentiment import identity, make_sentiment
from actsafe.rl.types import Model, Policy


class Exploration:
    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        return {}

    def get_policy(self) -> Policy:
        raise NotImplementedError("Must be implemented by subclass")


def make_exploration(
    config: DictConfig, action_dim: int, key: jax.Array
) -> Exploration:
    if config.agent.exploration_strategy == "opax":
        return OpaxExploration(config, action_dim, key)
    elif config.agent.exploration_strategy == "uniform":
        return UniformExploration(action_dim)
    else:
        raise NotImplementedError("Unknown exploration strategy")


class OpaxExploration(Exploration):
    def __init__(
        self,
        config: DictConfig,
        action_dim: int,
        key: jax.Array,
    ):
        self.actor_critic = make_actor_critic(
            config,
            config.training.safe,
            config.agent.model.stochastic_size + config.agent.model.deterministic_size,
            action_dim,
            key,
            objective_sentiment=identity,
            constraint_sentiment=make_sentiment(
                config.agent.sentiment.constraint_pessimism
            ),
        )
        self.reward_scale = config.agent.exploration_reward_scale
        self.epistemic_scale = config.agent.exploration_epistemic_scale

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        model = OpaxBridge(model, self.reward_scale, self.epistemic_scale)
        outs = self.actor_critic.update(model, initial_states, key)
        outs = {f"{_append_opax(k)}": v for k, v in outs.items()}
        return outs

    def get_policy(self) -> Policy:
        return self.actor_critic.actor.act


def _append_opax(string):
    parts = string.split("/")
    parts.insert(2, "opax")
    return "/".join(parts)


class UniformExploration(Exploration):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.policy = lambda _, key: jax.random.uniform(key, (self.action_dim,))

    def get_policy(self) -> Policy:
        return self.policy
