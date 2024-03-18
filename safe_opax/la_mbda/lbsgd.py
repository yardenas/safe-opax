"""https://github.com/lasgroup/lbsgd-rl/blob/main/lbsgd_rl/"""

from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from safe_opax.common.pytree_utils import pytrees_unstack
from safe_opax.la_mbda.actor_critic import ContinuousActor
from safe_opax.la_mbda.safe_actor_critic import ActorEvaluation


class LBSGDState(NamedTuple):
    eta: jax.Array


def compute_lr(constraint, loss_grads, constraint_grads, m_0, m_1, eta):
    constraint_grads, _ = jax.flatten_util.ravel_pytree(constraint_grads)
    loss_grads, _ = jax.flatten_util.ravel_pytree(loss_grads)
    projection = constraint_grads.dot(loss_grads)
    lhs = (
        constraint
        / (
            2.0 * jnp.abs(projection) / jnp.linalg.norm(loss_grads)
            + jnp.sqrt(constraint * m_1 + 1e-8)
        )
        / (jnp.linalg.norm(loss_grads) + 1e-8)
    )
    m_2 = (
        m_0
        + 10.0 * eta * (m_1 / (constraint + 1e-8))
        + 8.0
        * eta
        * jnp.linalg.norm(projection) ** 2
        / ((jnp.linalg.norm(loss_grads) * constraint) ** 2)
    )
    rhs = 1.0 / m_2
    return jnp.minimum(lhs, rhs)


def lbsgd_update(
    state: LBSGDState, updates: PyTree, eta_rate: float, m_0: float, m_1: float
) -> tuple[PyTree, LBSGDState]:
    def happy_case():
        lr = compute_lr(constraint, loss_grads, constraints_grads, m_0, m_1, eta_t)
        new_eta = eta_t / eta_rate
        updates = jax.tree_map(lambda x: x * lr, loss_grads)
        return updates, LBSGDState(new_eta)

    def fallback():
        # Taking the negative gradient of the constraints to minimize the costs
        updates = jax.tree_map(lambda x: x * -1.0, constraints_grads)
        return updates, LBSGDState(eta_t)

    loss_grads, constraints_grads, constraint = updates
    eta_t = state.eta
    return jax.lax.cond(
        jnp.greater(constraint, 0.0),
        happy_case,
        fallback,
    )


def jacrev(f, has_aux=False):
    def jacfn(x):
        y, vjp_fn, aux = eqx.filter_vjp(f, x, has_aux=has_aux)
        (J,) = eqx.filter_vmap(vjp_fn, in_axes=0)(jnp.eye(len(y)))
        return J, aux

    return jacfn


class LBSGDPenalizer:
    def __init__(self, m_0, m_1, eta, eta_rate) -> None:
        self.m_0 = m_0
        self.m_1 = m_1
        self.eta_rate = eta_rate + 1.0
        self.state = LBSGDState(eta)

    def __call__(
        self,
        evaluate: Callable[[ContinuousActor], ActorEvaluation],
        state: Any,
        actor: ContinuousActor,
    ) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
        def evaluate_helper(actor):
            evaluation = evaluate(actor)
            outs = jnp.stack([evaluation.loss, evaluation.constraint])
            return outs, evaluation

        jacobian, rest = jacrev(evaluate_helper, has_aux=True)(actor)
        loss_grads, constraint_grads = pytrees_unstack(jacobian)
        updates, state = lbsgd_update(
            state,
            (loss_grads, constraint_grads, rest.constraint),
            self.eta_rate,
            self.m_0,
            self.m_1,
        )
        metrics = {
            "agent/lbsgd/eta": state.eta,
        }
        return updates, state, rest, metrics
