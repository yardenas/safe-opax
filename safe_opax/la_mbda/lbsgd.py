"""https://github.com/lasgroup/lbsgd-rl/blob/main/lbsgd_rl/"""

from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from safe_opax.common.mixed_precision import apply_dtype
from safe_opax.common.pytree_utils import pytrees_unstack
from safe_opax.la_mbda.actor_critic import ContinuousActor
from safe_opax.la_mbda.safe_actor_critic import ActorEvaluation

_EPS = 1e-8


class LBSGDState(NamedTuple):
    eta: jax.Array


def compute_lr(alpha_1, g, grad_f_1, m_0, m_1, eta):
    grad_f_1, _ = jax.flatten_util.ravel_pytree(grad_f_1)
    g, _ = jax.flatten_util.ravel_pytree(g)
    theta_1 = grad_f_1.dot(g / (jnp.linalg.norm(g) + _EPS))
    lhs = (
        alpha_1
        / (2.0 * jnp.abs(theta_1) + jnp.sqrt(alpha_1 * m_1 + _EPS))
        / (jnp.linalg.norm(g) + _EPS)
    )
    m_2 = (
        m_0
        + 10.0 * eta * (m_1 / (alpha_1 + _EPS))
        + 8.0
        * eta
        * jnp.linalg.norm(theta_1) ** 2
        / ((jnp.linalg.norm(g) * alpha_1) ** 2 + _EPS)
    )
    rhs = 1.0 / m_2
    return jnp.minimum(lhs, rhs)


def lbsgd_update(
    state: LBSGDState, updates: PyTree, eta_rate: float, m_0: float, m_1: float
) -> tuple[PyTree, LBSGDState, float]:
    def happy_case():
        lr = compute_lr(alpha_1, g, grad_f_1, m_0, m_1, eta_t)
        new_eta = eta_t / eta_rate
        updates = jax.tree_map(lambda x: x * lr, g)
        return updates, LBSGDState(new_eta), lr

    def fallback():
        # Taking the negative gradient of the constraints to minimize the costs
        updates = grad_f_1
        return updates, LBSGDState(eta_t), jnp.nan

    g, grad_f_1, alpha_1 = updates
    eta_t = state.eta
    return jax.lax.cond(
        jnp.greater(alpha_1, 0.0),
        happy_case,
        fallback,
    )


def jacrev(f, has_aux=False):
    def jacfn(x):
        y, vjp_fn, aux = eqx.filter_vjp(f, x, has_aux=has_aux)  # type: ignore
        (J,) = eqx.filter_vmap(vjp_fn, in_axes=eqx.if_array(0))(jnp.eye(len(y)))
        return J, aux

    return jacfn


class LBSGDPenalizer:
    def __init__(self, m_0: float, m_1: float, eta: float, eta_rate: float) -> None:
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
            loss = evaluation.loss - state.eta * jnp.log(evaluation.constraint)
            outs = jnp.stack([loss, -evaluation.constraint])
            return outs, evaluation

        jacobian, rest = jacrev(evaluate_helper, has_aux=True)(actor)
        g, grad_f_1 = pytrees_unstack(jacobian)
        alpha = rest.constraint
        updates, state, lr = lbsgd_update(
            state,
            apply_dtype((g, grad_f_1, alpha), jnp.float32),
            self.eta_rate,
            self.m_0,
            self.m_1,
        )
        metrics = {
            "agent/lbsgd/eta": jnp.asarray(state.eta),
            "agent/lbsgd/lr": jnp.asarray(lr),
        }
        return updates, state, rest, metrics
