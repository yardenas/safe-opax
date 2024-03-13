from typing import Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax
from jaxtyping import PyTree

from safe_opax.rl.trajectory import TrajectoryData


class Learner:
    def __init__(
        self, model: PyTree, optimizer_config: dict[str, Any], batched: bool = False
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("clip", float("inf"))),
            optax.scale_by_adam(eps=optimizer_config.get("eps", 1e-8)),
            optax.scale(-optimizer_config.get("lr", 1e-3)),
        )
        if batched:
            init_fn = eqx.filter_vmap(lambda model: self.optimizer.init(model))
        else:
            init_fn = self.optimizer.init
        self.state = init_fn(eqx.filter(model, eqx.is_array))

    def grad_step(
        self, model: PyTree, grads: PyTree, state: optax.OptState
    ) -> tuple[PyTree, optax.OptState]:
        updates, new_opt_state = self.optimizer.update(grads, state)
        all_ok = all_finite(updates)
        updates = update_if(
            all_ok, updates, jax.tree_map(lambda x: jnp.zeros_like(x), updates)
        )
        new_opt_state = update_if(all_ok, new_opt_state, state)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state


def all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = list(map(jnp.isfinite, leaves))
        leaves = list(map(jnp.all, leaves))
        return jnp.stack(list(leaves)).all()


def update_if(pred, update, fallback):
    return jax.tree_map(lambda x, y: jax.lax.select(pred, x, y), update, fallback)


def inv_softplus(x):
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


def clip_stddev(stddev, stddev_min, stddev_max, stddev_scale=1.0):
    stddev = jnp.clip(
        (stddev + inv_softplus(0.1)) * stddev_scale,
        inv_softplus(stddev_min),
        inv_softplus(stddev_max),
    )
    return jnn.softplus(stddev)


class PRNGSequence:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def take_n(self, n):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        return keys[1:]


def add_to_buffer(buffer, trajectory, normalizer, reward_scale):
    results = normalizer.result
    normalize_fn = lambda x: normalize(x, results.mean, results.std)
    buffer.add(
        TrajectoryData(
            normalize_fn(trajectory.observation),
            normalize_fn(trajectory.next_observation),
            trajectory.action,
            trajectory.reward * reward_scale,
            trajectory.cost,
        )
    )


def normalize(
    observation,
    mean,
    std,
):
    diff = observation - mean
    return diff / (std + 1e-8)


def ensemble_predict(fn, in_axes=0):
    """
    A decorator that wraps (parameterized-)functions such that if they define
    an ensemble, predictions are made for each member of the ensemble individually.
    """

    def vmap_ensemble(*args, **kwargs):
        # First vmap along the batch dimension.
        ensemble_predict = lambda fn: jax.vmap(fn, in_axes=in_axes)(*args, **kwargs)
        # then vmap over members of the ensemble, such that each
        # individually computes outputs.
        ensemble_predict = eqx.filter_vmap(ensemble_predict)
        return ensemble_predict(fn)

    return vmap_ensemble


class Count:
    def __init__(self, n: int):
        self.count = 0
        self.n = n

    def __call__(self):
        bingo = (self.count + 1) == self.n
        self.count = (self.count + 1) % self.n
        return bingo


def pytrees_unstack(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees
