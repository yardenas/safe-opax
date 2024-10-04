import jax


def marginalize_prediction(x, axis=0):
    return jax.tree_map(lambda x: x.mean(axis), x)
