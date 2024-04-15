import jax.numpy as jnp
from safe_opax.la_mbda.rssm import ShiftScale
from safe_opax.rl.types import Prediction


# Better use the rewards than RSSM states since it is consistent
def reward(
    trajectory: Prediction, distributions: ShiftScale
) -> tuple[Prediction, ShiftScale]:
    opax_reward = 1.0 / 2.0 * jnp.log(1.0 + (trajectory.reward.std() ** 2))
    return Prediction(
        trajectory.next_state, opax_reward, trajectory.cost
    ), distributions


# def state(trajectory: Prediction, distributions: ShiftScale) -> tuple[Prediction, ShiftScale]:
#     opax_reward = (
#         1.0
#         / 2.0
#         * jnp.log(
#             1.0
#             + (
#                 (distributions.shift.std() ** 2)
#                 / marginalize_prediction(distributions.scale)
#             )
#         )
#     )
#     return Prediction(
#         trajectory.next_state, opax_reward, trajectory.cost
#     ), distributions
