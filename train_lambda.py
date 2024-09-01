import logging

import hydra
from omegaconf import OmegaConf
import jax

from safe_opax.common.mixed_precision import mixed_precision
from safe_opax.rl.trainer import get_state_path, load_state, should_resume, start_fresh

_LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="safe_opax/configs", config_name="config")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    state_path = get_state_path()
    if should_resume(state_path):
        _LOG.info(f"Resuming experiment from: {state_path}")
        trainer = load_state(cfg, state_path)
    else:
        _LOG.info("Starting a new experiment.")
        trainer = start_fresh(cfg)
    with trainer, jax.disable_jit(not cfg.jit), mixed_precision(cfg.mixed_precision):
        if cfg.enable_x64:
            jax.config.update("jax_enable_x64", True)
        trainer.train()
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
