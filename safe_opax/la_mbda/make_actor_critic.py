import numpy as np
from safe_opax.la_mbda.augmented_lagrangian import AugmentedLagrangianPenalizer
from safe_opax.la_mbda.dummy_penalizer import DummyPenalizer
from safe_opax.la_mbda.lbsgd import LBSGDPenalizer
from safe_opax.la_mbda.safe_actor_critic import SafeModelBasedActorCritic


def make_actor_critic(config, safe, state_dim, action_dim, key):
    # Account for the the discount factor in the budget.
    episode_safety_budget = (
        (
            (config.training.safety_budget / config.training.time_limit)
            / (1.0 - config.agent.safety_discount)
        )
        if config.agent.safety_discount < 1.0 - np.finfo(np.float32).eps
        else config.training.safety_budget
    ) + config.agent.safety_slack
    if safe:
        if config.agent.penalizer.name == "lbsgd":
            penalizer = LBSGDPenalizer(
                config.agent.penalizer.m_0,
                config.agent.penalizer.m_1,
                config.agent.penalizer.eta,
                config.agent.penalizer.eta_rate,
            )
        elif config.agent.penalizer.name == "lagrangian":
            penalizer = AugmentedLagrangianPenalizer(
                config.agent.penalizer.initial_lagrangian,
                config.agent.penalizer.initial_multiplier,
                config.agent.penalizer.multiplier_factor,
            )
        else:
            raise NotImplementedError
    else:
        penalizer = DummyPenalizer()
    return SafeModelBasedActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_config=config.agent.actor,
        critic_config=config.agent.critic,
        actor_optimizer_config=config.agent.actor_optimizer,
        critic_optimizer_config=config.agent.critic_optimizer,
        safety_critic_optimizer_config=config.agent.safety_critic_optimizer,
        horizon=config.agent.plan_horizon,
        discount=config.agent.discount,
        safety_discount=config.agent.safety_discount,
        lambda_=config.agent.lambda_,
        safety_budget=episode_safety_budget,
        penalizer=penalizer,
        ensemble_size=config.agent.ensemble_size,
        key=key,
    )
