import logging
import time
import hydra

from safe_opax import benchmark_suites
from safe_opax.rl import acting
from safe_opax.rl.trainer import get_trainer, load_state

_LOG = logging.getLogger(__name__)


def evaluate(agent, env, seeds):
    assert env is not None and agent is not None and seeds is not None
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


@hydra.main(version_base=None, config_path="safe_opax/configs", config_name="evaluate")
def main(cfg):
    _LOG.info(f"Setting up evaluation from checkpoint: {cfg.checkpoint}")
    trainer = load_state(cfg, cfg.checkpoint)
    agent, env, seeds = (
        trainer.agent,
        trainer.env,
        trainer.seeds,
    )
    make_env = benchmark_suites.make(agent.config)
    del trainer
    trainer = get_trainer(cfg.training.trainer)(agent.config, make_env)
    with trainer:
        _LOG.info("Starting evaluation.")
        evaluate(agent, env, seeds)


if __name__ == "__main__":
    main()
