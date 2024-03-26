from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp

from safe_opax.la_mbda.utils import marginalize_prediction


class ObjectiveModel(NamedTuple):
    values: jax.Array
    trajectory: jax.Array
    reward: jax.Array


Sentiment = Callable[[ObjectiveModel], ObjectiveModel]


def bayes(model: ObjectiveModel) -> ObjectiveModel:
    return marginalize_prediction(model, 1)


def _emprirical_estimate(
    model: ObjectiveModel, reduce_fn: Callable[[jax.Array], jax.Array]
) -> ObjectiveModel:
    ids = reduce_fn(model.values.mean((0, 2)))
    take = lambda x: x[:, ids, :]
    return jax.tree_map(take, model)


def empirical_optimism(model: ObjectiveModel) -> ObjectiveModel:
    return _emprirical_estimate(model, jnp.argmax)


def empirical_robustness(model: ObjectiveModel) -> ObjectiveModel:
    return _emprirical_estimate(model, jnp.argmin)
