from typing import Any, Callable

import equinox as eqx
import jax
from jaxtyping import PyTree

from safe_opax.la_mbda.actor_critic import ContinuousActor
from safe_opax.la_mbda.safe_actor_critic import ActorEvaluation


class DummyPenalizer:
    state: None

    def __call__(
        self,
        evaluate: Callable[[ContinuousActor], ActorEvaluation],
        state: Any,
        actor: ContinuousActor,
    ) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
        def loss_fn(actor):
            evaluation = evaluate(actor)
            loss = evaluation.loss
            return loss, evaluation

        grads, evaluation = eqx.filter_grad(loss_fn, has_aux=True)(actor)
        metrics: dict[str, jax.Array] = dict()
        return grads, None, evaluation, metrics
