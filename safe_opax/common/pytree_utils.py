import jax
import jax.numpy as jnp


def pytrees_unstack(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def pytrees_stack(pytrees, axis=0):
    results = jax.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results
