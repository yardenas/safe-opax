from functools import partial
from typing import Any, Callable, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from optax import OptState, l2_loss

from safe_opax.common.learner import Learner
from safe_opax.common.mixed_precision import apply_mixed_precision
from safe_opax.la_mbda import sentiment
from safe_opax.la_mbda.actor_critic import ContinuousActor, Critic
from safe_opax.la_mbda.types import Model, RolloutFn
from safe_opax.rl.utils import nest_vmap


class ActorEvaluation(NamedTuple):
    reward_objective_model: sentiment.ObjectiveModel
    cost_objective_model: sentiment.ObjectiveModel
    loss: jax.Array
    constraint: jax.Array
    safe: jax.Array


class Penalizer(Protocol):
    state: PyTree

    def __call__(
        self,
        evaluate: Callable[[ContinuousActor], ActorEvaluation],
        state: Any,
        actor: ContinuousActor,
    ) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
        ...


class SafeModelBasedActorCritic:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_config: dict[str, Any],
        critic_config: dict[str, Any],
        actor_optimizer_config: dict[str, Any],
        critic_optimizer_config: dict[str, Any],
        safety_critic_optimizer_config: dict[str, Any],
        horizon: int,
        discount: float,
        safety_discount: float,
        lambda_: float,
        safety_budget: float,
        key: jax.Array,
        penalizer: Penalizer,
    ):
        actor_key, critic_key, safety_critic_key = jax.random.split(key, 3)
        self.actor = ContinuousActor(
            state_dim=state_dim,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        self.critic = Critic(state_dim=state_dim, **critic_config, key=critic_key)
        self.safety_critic = Critic(
            state_dim=state_dim, **critic_config, key=safety_critic_key
        )
        self.actor_learner = Learner(self.actor, actor_optimizer_config)
        self.critic_learner = Learner(self.critic, critic_optimizer_config)
        self.safety_critic_learner = Learner(
            self.safety_critic, safety_critic_optimizer_config
        )
        self.horizon = horizon
        self.discount = discount
        self.lambda_ = lambda_
        self.safety_discount = safety_discount
        self.safety_budget = safety_budget
        self.update_fn = batched_update_safe_actor_critic
        self.penalizer = penalizer

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        actor_critic_fn = partial(self.update_fn, model.sample)
        results: SafeActorCriticStepResults = actor_critic_fn(
            self.horizon,
            initial_states,
            self.actor,
            self.critic,
            self.safety_critic,
            self.actor_learner.state,
            self.critic_learner.state,
            self.safety_critic_learner.state,
            self.actor_learner,
            self.critic_learner,
            self.safety_critic_learner,
            key,
            self.discount,
            self.safety_discount,
            self.lambda_,
            self.safety_budget,
            self.penalizer,
            self.penalizer.state,
        )
        self.actor = results.new_actor
        self.critic = results.new_critic
        self.safety_critic = results.new_safety_critic
        self.actor_learner.state = results.new_actor_learning_state
        self.critic_learner.state = results.new_critic_learning_state
        self.safety_critic_learner.state = results.new_safety_critic_learning_state
        self.penalizer.state = results.new_penalty_state
        return {
            "agent/actor/loss": results.actor_loss.item(),
            "agent/critic/loss": results.critic_loss.item(),
            "agent/safety_critic/loss": results.safety_critic_loss.item(),
            "agent/safety_critic/safe": float(results.safe.item()),
            "agent/safety_critic/constraint": results.constraint.item(),
            "agent/safety_critic/safety": results.safety.item(),
            **{k: v.item() for k, v in results.metrics.items()},
        }


class SafeActorCriticStepResults(NamedTuple):
    new_actor: ContinuousActor
    new_critic: Critic
    new_safety_critic: Critic
    new_actor_learning_state: OptState
    new_critic_learning_state: OptState
    new_safety_critic_learning_state: OptState
    actor_loss: jax.Array
    critic_loss: jax.Array
    safety_critic_loss: jax.Array
    safe: jax.Array
    constraint: jax.Array
    safety: jax.Array
    new_penalty_state: Any
    metrics: dict[str, jax.Array]


def discounted_cumsum(x: jax.Array, discount: float) -> jax.Array:
    # [1, discount, discount^2 ...]
    scales = jnp.cumprod(jnp.ones_like(x)[:-1] * discount)
    scales = jnp.concatenate([jnp.ones_like(scales[:1]), scales], -1)
    # Flip scales since jnp.convolve flips it as default.
    return jnp.convolve(x, scales[::-1])[-x.shape[0] :]


def compute_lambda_values(
    next_values: jax.Array, rewards: jax.Array, discount: float, lambda_: float
) -> jax.Array:
    tds = rewards + (1.0 - lambda_) * discount * next_values
    tds = tds.at[-1].add(lambda_ * discount * next_values[-1])
    return discounted_cumsum(tds, lambda_ * discount)


def critic_loss_fn(
    critic: Critic, trajectories: jax.Array, lambda_values: jax.Array
) -> jax.Array:
    values = nest_vmap(critic, 2)(trajectories)
    return l2_loss(values[:, :-1], lambda_values[:, 1:]).mean()


def evaluate_actor(
    actor: ContinuousActor,
    critic: Critic,
    safety_critic: Critic,
    rollout_fn: RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    key: jax.Array,
    discount: float,
    safety_discount: float,
    lambda_: float,
    safety_budget: float,
) -> ActorEvaluation:
    trajectories, _ = rollout_fn(horizon, initial_states, key, actor.act)
    # vmap over batch, ensemble and time axes.
    bootstrap_values = nest_vmap(critic, 3)(trajectories.next_state)
    lambda_values = nest_vmap(compute_lambda_values, 2, eqx.filter_vmap)(
        bootstrap_values, trajectories.reward, discount, lambda_
    )
    bootstrap_safety_values = nest_vmap(safety_critic, 3)(trajectories.next_state)
    safety_lambda_values = nest_vmap(compute_lambda_values, 2, eqx.filter_vmap)(
        bootstrap_safety_values, trajectories.cost, safety_discount, lambda_
    )
    reward_objective_model = jax.tree_map(
        lambda x: x[:, 0],
        sentiment.ObjectiveModel(lambda_values, trajectories.next_state),
    )
    cost_objective_model = jax.tree_map(
        lambda x: x[:, 0],
        sentiment.ObjectiveModel(safety_lambda_values, trajectories.next_state),
    )
    # reward_objective_model = sentiment.bayes(
    #     sentiment.ObjectiveModel(lambda_values, trajectories.next_state)
    # )
    # cost_objective_model = sentiment.bayes(
    #     sentiment.ObjectiveModel(safety_lambda_values, trajectories.next_state)
    # )
    loss = -reward_objective_model.values.mean()
    constraint = safety_budget - cost_objective_model.values.mean()
    return ActorEvaluation(
        reward_objective_model,
        cost_objective_model,
        loss,
        constraint,
        jnp.greater(constraint, 0.0),
    )


def update_safe_actor_critic(
    rollout_fn: RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: ContinuousActor,
    critic: Critic,
    safety_critic: Critic,
    actor_learning_state: OptState,
    critic_learning_state: OptState,
    safety_critic_learning_state: OptState,
    actor_learner: Learner,
    critic_learner: Learner,
    safety_critic_learner: Learner,
    key: jax.Array,
    discount: float,
    safety_discount: float,
    lambda_: float,
    safety_budget: float,
    penalty_fn: Penalizer,
    penalty_state: Any,
) -> SafeActorCriticStepResults:
    actor_grads, new_penalty_state, evaluation, metrics = penalty_fn(
        lambda actor: evaluate_actor(
            actor,
            critic,
            safety_critic,
            rollout_fn,
            horizon,
            initial_states,
            key,
            discount,
            safety_discount,
            lambda_,
            safety_budget,
        ),
        penalty_state,
        actor,
    )
    new_actor, new_actor_state = actor_learner.grad_step(
        actor, actor_grads, actor_learning_state
    )
    critic_loss, grads = eqx.filter_value_and_grad(critic_loss_fn)(
        critic,
        evaluation.reward_objective_model.trajectory,
        evaluation.reward_objective_model.values,
    )
    new_critic, new_critic_state = critic_learner.grad_step(
        critic, grads, critic_learning_state
    )
    # TODO (yarden): figure out the scaling here?
    scaled_safety = evaluation.cost_objective_model.values
    safety_critic_loss, grads = eqx.filter_value_and_grad(critic_loss_fn)(
        safety_critic,
        evaluation.cost_objective_model.trajectory,
        scaled_safety,
    )
    new_safety_critic, new_safety_critic_state = safety_critic_learner.grad_step(
        safety_critic, grads, safety_critic_learning_state
    )
    return SafeActorCriticStepResults(
        new_actor,
        new_critic,
        new_safety_critic,
        new_actor_state,
        new_critic_state,
        new_safety_critic_state,
        evaluation.loss,
        critic_loss,
        safety_critic_loss,
        evaluation.safe,
        evaluation.constraint,
        scaled_safety.mean(),
        new_penalty_state,
        metrics,
    )


@eqx.filter_jit
@apply_mixed_precision(
    target_module_names=["critic", "safety_critic", "actor", "rollout_fn"],
    target_input_names=["initial_states"],
)
def batched_update_safe_actor_critic(
    rollout_fn: RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: ContinuousActor,
    critic: Critic,
    safety_critic: Critic,
    actor_learning_state: OptState,
    critic_learning_state: OptState,
    safety_critic_learning_state: OptState,
    actor_learner: Learner,
    critic_learner: Learner,
    safety_critic_learner: Learner,
    key: jax.Array,
    discount: float,
    safety_discount: float,
    lambda_: float,
    safety_budget: float,
    penalty_fn: Penalizer,
    penalty_state: Any,
) -> SafeActorCriticStepResults:
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None))
    return update_safe_actor_critic(
        vmapped_rollout_fn,
        horizon,
        initial_states,
        actor,
        critic,
        safety_critic,
        actor_learning_state,
        critic_learning_state,
        safety_critic_learning_state,
        actor_learner,
        critic_learner,
        safety_critic_learner,
        key,
        discount,
        safety_discount,
        lambda_,
        safety_budget,
        penalty_fn,
        penalty_state,
    )
