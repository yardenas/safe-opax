import jax
import equinox as eqx
from safe_opax import opax
from safe_opax.la_mbda.rssm import ShiftScale, State
from safe_opax.la_mbda.world_model import WorldModel
from safe_opax.rl.types import Policy, Prediction


class OpaxBridge(eqx.Module):
    model: WorldModel
    reward_scale: float = eqx.field(static=True)

    def sample(
        self,
        horizon: int,
        initial_state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, ShiftScale]:
        samples: tuple[Prediction, ShiftScale] = self.model.sample(
            horizon, initial_state, key, policy
        )
        trajectory, distributions = samples
        return opax.modify_reward(trajectory, distributions, self.reward_scale)
