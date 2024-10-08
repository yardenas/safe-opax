import jax
import equinox as eqx
from actsafe import opax
from actsafe.actsafe.rssm import ShiftScale, State
from actsafe.actsafe.world_model import WorldModel
from actsafe.rl.types import Policy, Prediction


class OpaxBridge(eqx.Module):
    model: WorldModel
    reward_scale: float = eqx.field(static=True)
    reward_epistemic_scale: float = eqx.field(static=True)

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
        return opax.modify_reward(
            trajectory, distributions, self.reward_scale, self.reward_epistemic_scale
        )
