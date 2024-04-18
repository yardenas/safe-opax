import jax.numpy as jnp
from safe_opax.la_mbda.rssm import ShiftScale
from safe_opax.rl.types import Prediction

_EPS = 1e-5


def modify_reward(
    trajectory: Prediction, distributions: ShiftScale, scale: float = 1.0
) -> tuple[Prediction, ShiftScale]:
    return Prediction(
        trajectory.next_state,
        normalized_epistemic_uncertainty(distributions) * scale,
        trajectory.cost,
    ), distributions


def normalized_epistemic_uncertainty(
    distributions: ShiftScale, axis: int = 0
) -> jnp.ndarray:
    epistemic_uncertainty = distributions.shift.std(axis)
    aleatoric_uncertainty = distributions.scale.mean(axis)
    return 0.5 * jnp.log(
        1.0
        + (epistemic_uncertainty.mean(-1) / (aleatoric_uncertainty.mean(-1) + _EPS))
        ** 2
    )
