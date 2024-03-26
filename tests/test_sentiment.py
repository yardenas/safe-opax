import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from safe_opax.la_mbda.rssm import State
from safe_opax.la_mbda.sentiment import (
    ObjectiveModel,
    empirical_optimism,
)

ENSEMBLE_SIZE = 5
BATCH_SIZE = 32
HORIZON = 15
BEST_MEMBER = 3


@pytest.fixture
def objective_model():
    all_values = []
    for _ in range(ENSEMBLE_SIZE):
        all_values.append(
            jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, HORIZON))
        )
    all_values[BEST_MEMBER] = 1e3 + all_values[BEST_MEMBER]
    values = jnp.stack(all_values, axis=1)
    rewards = (
        jnp.ones((BATCH_SIZE, ENSEMBLE_SIZE, HORIZON))
        * jnp.arange(ENSEMBLE_SIZE)[None, :, None]
    )
    trajectory = State(rewards, rewards).flatten()
    return ObjectiveModel(values, trajectory, rewards)


def test_optimism(objective_model):
    result = empirical_optimism(objective_model)
    ground_truth_best = jax.tree_map(lambda x: x[:, BEST_MEMBER], objective_model)
    assert eqx.tree_equal(result, ground_truth_best)
