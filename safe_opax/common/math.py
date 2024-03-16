import jax.nn as jnn
import jax.numpy as jnp


def inv_softplus(x):
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


def clip_stddev(stddev, stddev_min, stddev_max, stddev_scale=1.0):
    stddev = jnp.clip(
        (stddev + inv_softplus(0.1)) * stddev_scale,
        inv_softplus(stddev_min),
        inv_softplus(stddev_max),
    )
    return jnn.softplus(stddev)
