from copy import deepcopy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig
from optax import OptState, l2_loss

from actsafe.common.learner import Learner
from actsafe.common.mixed_precision import apply_mixed_precision
from actsafe.la_mbda.la_mbda import LaMBDA
from actsafe.rl.trajectory import TrajectoryData, Transition
from actsafe.rl.types import FloatArray


@eqx.filter_jit
def filter_actions(cost_model, states, actions, prev_costs, scale):
    def pre_env_action(state, action, prev_cost):
        g = cost_model(state)
        multiplier = jnp.clip((g.dot(action) + prev_cost) / g.dot(g), a_min=0.0)
        safe_action = action - multiplier * scale
        return safe_action

    return jax.vmap(pre_env_action)(states, actions, prev_costs)


class LaMBDADalal(LaMBDA):
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        # A hack to make LaMBDA's policy unsafe
        config_agent = deepcopy(config)
        config_agent.training.safe = False
        super().__init__(observation_space, action_space, config_agent)
        self.config = config
        self.cost_model = eqx.nn.MLP(
            config.agent.model.stochastic_size + config.agent.model.deterministic_size,
            int(np.prod(action_space.shape)),
        )
        self.prev_cost = np.zeros((self.config.training.parallel_envs,))

    def __call__(
        self,
        observation: FloatArray,
        train: bool = False,
    ) -> FloatArray:
        action = super().__call__(observation, train)
        if self.config.training.safe:
            return filter_actions(
                self.cost_model,
                self.state.rssm_state.flatten(),
                action,
                self.prev_cost,
                1.0,
            )
        else:
            return action

    def observe_transition(self, transition: Transition) -> None:
        self.prev_cost = transition.cost

    def update_model(self, batch: TrajectoryData) -> jax.Array:
        inferred_states = super().update_model(batch)
        return inferred_states

    def observe(self, trajectory: TrajectoryData) -> None:
        super().observe(trajectory)
        self.prev_cost = np.zeros((self.config.training.parallel_envs,))


@eqx.filter_jit
@apply_mixed_precision(
    target_input_names=["batch", "inferred_states"],
    target_module_names=["model"],
)
def update_cost_model(
    model: eqx.nn.MLP,
    learner: Learner,
    opt_state: OptState,
    batch: TrajectoryData,
    inferred_states: jax.Array,
) -> jax.Array:
    costs = batch.cost[:, :-1]
    next_costs = batch.cost[:, 1:]

    def loss_fn(cost_model):
        predict_costs = (
            lambda inferred_state, cost: cost_model(inferred_state).dot(batch.action)
            + cost
        )
        predict_costs = jax.vmap(predict_costs)
        predicted_costs = predict_costs(inferred_states, costs)
        return l2_loss(predicted_costs, next_costs).mean()

    value, grads = eqx.filter_value_and_grad(loss_fn)(model)
    new_model, new_opt_state = learner.grad_step(model, grads, opt_state)
    return (new_model, new_opt_state), value
