import jax
from omegaconf import DictConfig

from safe_opax.la_mbda.opax_bridge import opax_bridge
from safe_opax.la_mbda.make_actor_critic import make_actor_critic
from safe_opax.la_mbda.sentiment import identity
from safe_opax.rl.types import RolloutFn


class Exploration:
    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        raise NotImplementedError("Must be implemented by subclass")

    def update(
        self,
        rollout_fn: RolloutFn,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        return {}


def make_exploration(
    config: DictConfig, action_dim: int, key: jax.Array
) -> Exploration:
    if config.agent.exploration_strategy == "opax":
        return OpaxExploration(config, action_dim, key)
    else:
        return UniformExploration()


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
        rollout_fn: RolloutFn,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        outs = self.actor_critic.update(opax_bridge(rollout_fn), initial_states, key)

        def append_opax(string):
            parts = string.split("/")
            parts.insert(2, "opax")
            return "/".join(parts)

        outs = {f"{append_opax(k)}": v for k, v in outs.items()}
        return outs

    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        return self.actor_critic.actor.act(state, key)


class UniformExploration(Exploration):
    def __call__(self, state: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.uniform(jax.random.PRNGKey(key), state.shape)
