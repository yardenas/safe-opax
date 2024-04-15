import jax
from safe_opax import opax
from safe_opax.la_mbda.rssm import ShiftScale, State
from safe_opax.rl.types import Policy, RolloutFn, Prediction


def opax_bridge(rollout_fn: RolloutFn) -> RolloutFn:
    def intrinsic_reward_sample(
        horizon: int,
        initial_state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, ShiftScale]:
        samples: tuple[Prediction, ShiftScale] = rollout_fn(
            horizon, initial_state, key, policy
        )
        trajectory, distributions = samples
        assert isinstance(distributions, ShiftScale)
        return opax.reward(trajectory, distributions)

    return intrinsic_reward_sample
