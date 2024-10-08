import logging
import time
import hydra

from actsafe.rl import acting
from actsafe.rl.trainer import load_state

_LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="actsafe/configs", config_name="evaluate")
def main(cfg):
    _LOG.info(f"Setting up evaluation from checkpoint: {cfg.checkpoint}")
    trainer = load_state(cfg, cfg.checkpoint)
    agent, env, logger, seeds = (
        trainer.agent,
        trainer.env,
        trainer.logger,
        trainer.seeds,
    )
    assert (
        env is not None
        and agent is not None
        and logger is not None
        and seeds is not None
    )
    start_time = time.time()
    env.reset(seed=int(next(seeds)[0].item()))
    summary, step = acting.epoch(agent, env, 1, False, 0, 1)
    end_time = time.time()
    wall_time = end_time - start_time
    objective, cost_return, _ = summary.metrics
    _LOG.info(
        f"Evaluated {step} steps in {wall_time} seconds."
        f"objective={objective}, cost_return={cost_return}"
    )


if __name__ == "__main__":
    main()
