import jax
import jax.numpy as jnp
import equinox as eqx

from safe_opax.rl.trajectory import TrajectoryData


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


def add_to_buffer(buffer, trajectory, reward_scale):
    buffer.add(
        TrajectoryData(
            trajectory.observation,
            trajectory.next_observation,
            trajectory.action,
            trajectory.reward * reward_scale,
            trajectory.cost,
        )
    )


def normalize(observation, mean, std):
    diff = observation - mean
    return diff / (std + 1e-8)


class Count:
    def __init__(self, n: int, steps: int = 1):
        self.count = 0
        self.n = (n // steps) * steps
        self.steps = steps

    def __call__(self):
        bingo = (self.count + self.steps) == self.n
        self.count = (self.count + self.steps) % self.n
        return bingo


class Until:
    def __init__(self, n: int):
        self.count = 0
        self.n = n

    def __call__(self):
        return self.count <= self.n

    def tick(self):
        self.count += 1


def nest_vmap(f, count, vmap_fn=jax.vmap):
    for _ in range(count):
        f = vmap_fn(f)
    return f


def glorot_uniform(weight, key, scale=1.0):
    fan_in, fan_out = weight.shape
    limit = jnp.sqrt(6.0 * scale / (fan_in + fan_out))
    return jax.random.uniform(key, weight.shape, minval=-limit, maxval=limit)


def rl_initialize_weights_trick(model, bias_shift=0.0, weight_scale=0.01):
    """Follows https://arxiv.org/pdf/2006.05990.pdf"""
    model = eqx.tree_at(
        lambda model: model.layers[-1].weight,
        model,
        model.layers[-1].weight * weight_scale,
    )
    model = eqx.tree_at(
        lambda model: model.layers[-1].bias, model, model.layers[-1].bias + bias_shift
    )
    return model


def init_linear_weights(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model
