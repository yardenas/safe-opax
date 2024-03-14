import equinox as eqx
import jax

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
