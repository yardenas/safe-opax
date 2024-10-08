from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from actsafe.actsafe.actor_critic import ContinuousActor
from actsafe.actsafe.safe_actor_critic import ActorEvaluation


class AugmentedLagrangianUpdate(NamedTuple):
    psi: jax.Array
    new_lagrangian: jax.Array
    new_multiplier: jax.Array


def augmented_lagrangian(
    constraint: jax.Array,
    lagrangian: jax.Array,
    multiplier: jax.Array,
    multiplier_factor: float,
) -> (
    AugmentedLagrangianUpdate
):  # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    g = -constraint
    c = multiplier
    cond = lagrangian + c * g
    psi = jnp.where(
        jnp.greater(cond, 0.0),
        lagrangian * g + c / 2.0 * g**2,
        -1.0 / (2.0 * c) * lagrangian**2,
    )
    new_multiplier = jnp.clip(multiplier * (1.0 + multiplier_factor), multiplier, 1.0)
    new_lagrangian = jnn.relu(cond)
    return AugmentedLagrangianUpdate(psi, new_lagrangian, new_multiplier)


class AugmentedLagrangianState(NamedTuple):
    lagrangian: jax.Array
    multiplier: jax.Array


class AugmentedLagrangianPenalizer:
    def __init__(
        self,
        initial_lagrangian: float,
        initial_multiplier: float,
        multiplier_factor: float,
    ) -> None:
        self.multiplier_factor = multiplier_factor
        self.state = AugmentedLagrangianState(
            jnp.asarray(initial_lagrangian),
            jnp.asarray(initial_multiplier),
        )

    def __call__(
        self,
        evaluate: Callable[[ContinuousActor], ActorEvaluation],
        state: Any,
        actor: ContinuousActor,
    ) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
        def loss_fn(actor):
            evaluation = evaluate(actor)
            loss = evaluation.loss
            update = augmented_lagrangian(
                evaluation.constraint,
                state.lagrangian,
                state.multiplier,
                self.multiplier_factor,
            )
            loss += update.psi
            return loss, (update, evaluation)

        grads, (rest, evaluation) = eqx.filter_grad(loss_fn, has_aux=True)(actor)
        new_lagrangian_state = AugmentedLagrangianState(
            rest.new_lagrangian, rest.new_multiplier
        )
        metrics = {
            "agent/augmented_lagrangian/lagrangian": jnp.asarray(state.lagrangian),
            "agent/augmented_lagrangian/multiplier": jnp.asarray(state.multiplier),
        }
        return grads, new_lagrangian_state, evaluation, metrics
