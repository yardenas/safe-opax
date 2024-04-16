import jax
from omegaconf import DictConfig

from safe_opax.la_mbda.opax_bridge import OpaxBridge
from safe_opax.la_mbda.make_actor_critic import make_actor_critic
from safe_opax.la_mbda.sentiment import identity
from safe_opax.rl.types import Model


class Exploration:
    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        raise NotImplementedError("Must be implemented by subclass")

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        return {}


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
            sentiment=identity,
        )

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        model = OpaxBridge(model)
        outs = self.actor_critic.update(model, initial_states, key)

        def append_opax(string):
            parts = string.split("/")
            parts.insert(2, "opax")
            return "/".join(parts)

        outs = {f"{append_opax(k)}": v for k, v in outs.items()}
        return outs

    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        return self.actor_critic.actor.act(state, key)


class UniformExploration(Exploration):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.uniform(key, (self.action_dim,))
