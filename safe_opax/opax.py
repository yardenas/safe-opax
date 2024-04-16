import jax.numpy as jnp
from safe_opax.la_mbda.rssm import ShiftScale
from safe_opax.rl.types import Prediction

_EPS = 1e-5

def modify_reward(
    trajectory: Prediction, distributions: ShiftScale
) -> tuple[Prediction, ShiftScale]:
    return Prediction(
        trajectory.next_state,
        normalized_epistemic_uncertainty(distributions),
        trajectory.cost,
    ), distributions


def normalized_epistemic_uncertainty(distributions: ShiftScale) -> jnp.ndarray:
    return 0.5 * jnp.log(
        1.0 + (distributions.shift.std(1) / (distributions.scale.mean(1) + _EPS)) ** 2
    )
