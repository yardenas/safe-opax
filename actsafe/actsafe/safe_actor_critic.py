from typing import Any, Callable, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from optax import OptState
import distrax as trx

from actsafe.common.learner import Learner
from actsafe.common.mixed_precision import apply_mixed_precision
from actsafe.actsafe.rssm import ShiftScale
from actsafe.actsafe.sentiment import Sentiment
from actsafe.actsafe.actor_critic import ContinuousActor, Critic, actor_entropy
from actsafe.opax import normalized_epistemic_uncertainty
from actsafe.rl.types import Model, RolloutFn
from actsafe.rl.utils import nest_vmap


class ActorEvaluation(NamedTuple):
    trajectories: jax.Array
    objective_values: jax.Array
    cost_values: jax.Array
    loss: jax.Array
    constraint: jax.Array
    safe: jax.Array
    priors: ShiftScale
    reward_stddev: jax.Array
    cost_stddev: jax.Array


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
        objective_sentiment: Sentiment,
        constraint_sentiment: Sentiment,
    ):
        actor_key, critic_key, safety_critic_key = jax.random.split(key, 3)
        self.actor = ContinuousActor(
            state_dim=state_dim,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        make_critic = lambda key: Critic(state_dim=state_dim, **critic_config, key=key)
        self.critic = make_critic(critic_key)
        self.safety_critic = make_critic(safety_critic_key)
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
        self.penalizer = penalizer
        self.objective_sentiment = objective_sentiment
        self.constraint_sentiment = constraint_sentiment

    def update(
        self,
        model: Model,
        initial_states: jax.Array,
        key: jax.Array,
    ) -> dict[str, float]:
        results: SafeActorCriticStepResults = update_safe_actor_critic(
            model.sample,
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
            self.objective_sentiment,
            self.constraint_sentiment,
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
            "agent/safety_critic/cost_value": results.cost_value.item(),
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
    cost_value: jax.Array
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
    critic: Critic,
    trajectories: jax.Array,
    lambda_values: jax.Array,
    discount: float,
    horizon: int,
) -> jax.Array:
    planning_discount = compute_discount(discount, horizon - 1)
    values = nest_vmap(critic, 2)(trajectories)
    log_probs = trx.Independent(
        trx.Normal(lambda_values, jnp.ones_like(lambda_values)), 0
    ).log_prob(values)
    loss = -(log_probs * planning_discount).mean()
    return loss


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
    objective_sentiment: Sentiment,
    constraint_sentiment: Sentiment,
) -> ActorEvaluation:
    trajectories, priors = rollout_fn(horizon, initial_states, key, actor.act)
    next_step = lambda x: x[:, 1:]
    current_step = lambda x: x[:, :-1]
    next_states = next_step(trajectories.next_state)
    bootstrap_values = nest_vmap(critic, 2, eqx.filter_vmap)(next_states)
    rewards = current_step(objective_sentiment(trajectories.reward, priors))
    lambda_values = eqx.filter_vmap(compute_lambda_values)(
        bootstrap_values, rewards, discount, lambda_
    )
    bootstrap_safety_values = nest_vmap(safety_critic, 2, eqx.filter_vmap)(next_states)
    costs = current_step(constraint_sentiment(trajectories.cost, priors))
    safety_lambda_values = eqx.filter_vmap(compute_lambda_values)(
        bootstrap_safety_values,
        costs,
        safety_discount,
        lambda_,
    )
    planning_discount = compute_discount(discount, horizon - 1)
    objective = (lambda_values * planning_discount).mean()
    loss = -objective
    constraint = safety_budget - safety_lambda_values.mean()
    return ActorEvaluation(
        current_step(trajectories.next_state),
        lambda_values,
        safety_lambda_values,
        loss,
        constraint,
        jnp.greater(constraint, 0.0),
        priors,
        trajectories.reward.std(1).mean(),
        trajectories.cost.std(1).mean(),
    )


@eqx.filter_jit
@apply_mixed_precision(
    target_module_names=["critic", "safety_critic", "actor", "rollout_fn"],
    target_input_names=["initial_states"],
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
    objective_sentiment: Sentiment,
    constraint_sentiment: Sentiment,
) -> SafeActorCriticStepResults:
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None))
    actor_grads, new_penalty_state, evaluation, metrics = penalty_fn(
        lambda actor: evaluate_actor(
            actor,
            critic,
            safety_critic,
            vmapped_rollout_fn,
            horizon,
            initial_states,
            key,
            discount,
            safety_discount,
            lambda_,
            safety_budget,
            objective_sentiment,
            constraint_sentiment,
        ),
        penalty_state,
        actor,
    )
    new_actor, new_actor_state = actor_learner.grad_step(
        actor, actor_grads, actor_learning_state
    )
    critics_grads_fn = eqx.filter_value_and_grad(critic_loss_fn)
    critic_loss, grads = critics_grads_fn(
        critic, evaluation.trajectories, evaluation.objective_values, discount, horizon
    )
    new_critic, new_critic_state = critic_learner.grad_step(
        critic, grads, critic_learning_state
    )
    cost_values = evaluation.cost_values
    safety_critic_loss, grads = critics_grads_fn(
        safety_critic,
        evaluation.trajectories,
        cost_values,
        safety_discount,
        horizon,
    )
    new_safety_critic, new_safety_critic_state = safety_critic_learner.grad_step(
        safety_critic, grads, safety_critic_learning_state
    )
    metrics["agent/sentiment/epistemic_uncertainty"] = normalized_epistemic_uncertainty(
        evaluation.priors, 1
    ).mean()
    metrics["agent/sentiment/reward_stddev"] = evaluation.reward_stddev
    metrics["agent/sentiment/cost_stddev"] = evaluation.cost_stddev
    metrics["agent/actor/entropy"] = actor_entropy(new_actor, initial_states)
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
        cost_values.mean(),
        new_penalty_state,
        metrics,
    )


def compute_discount(factor, length):
    d = jnp.cumprod(factor * jnp.ones((length - 1,)))
    d = jnp.concatenate([jnp.ones((1,)), d])
    return d
