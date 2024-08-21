import json
import jax
import jax.numpy as jnp
import equinox as eqx

class Model(eqx.Module):
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    dense3: eqx.nn.Linear

    def __init__(self, inputs, num_classes=1, *, key):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        self.dense1 = eqx.nn.Linear(inputs, 256, key=key1)
        self.dense2 = eqx.nn.Linear(256, 256, key=key2)
        self.dense3 = eqx.nn.Linear(256, num_classes, key=key3)

    def __call__(self, x):
        x = jax.nn.tanh(self.dense1(x))
        x = jax.nn.tanh(self.dense2(x))
        x = self.dense3(x)
        return x

class Policy:
    def __init__(self, model):
        self.model = model
        self.mean = None
        self.var = None

    def step(self, obs):
        if self.mean is not None and self.var is not None:
            obs = (obs - self.mean) / jnp.sqrt(self.var + 1e-8)
        obs = jnp.array(obs, dtype=jnp.float32)
        action = self.model(obs)
        return action

    def load(self, path, mean=None, var=None):
        self.model, _ = load_model(path)
        if mean is not None and var is not None:
            self.mean = jnp.load(mean)[0]
            self.var = jnp.load(var)[0]

    def __call__(self, obs):
        return self.step(obs)

    def __str__(self):
        return "EquinoxPolicy"

    def __repr__(self):
        return "EquinoxPolicy"

def load_model(path: str) -> tuple[Model, dict[str, int]]:
    with open(path, "rb") as f:
        parameters = json.loads(f.readline().decode())
        model = Model(**parameters, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model), parameters
