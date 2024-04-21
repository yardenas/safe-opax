import numpy as np
from safe_opax.la_mbda.augmented_lagrangian import AugmentedLagrangianPenalizer
from safe_opax.la_mbda.dummy_penalizer import DummyPenalizer
from safe_opax.la_mbda.lbsgd import LBSGDPenalizer
from safe_opax.la_mbda.safe_actor_critic import SafeModelBasedActorCritic
from safe_opax.la_mbda.sentiment import bayes


def make_actor_critic(cfg, safe, state_dim, action_dim, key, sentiment=bayes):
    # Account for the the discount factor in the budget.
    episode_safety_budget = (
        (
            (cfg.training.safety_budget / cfg.training.time_limit)
            / (1.0 - cfg.agent.safety_discount)
        )
        if cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps
        else cfg.training.safety_budget
    ) + cfg.agent.safety_slack
    if safe:
        if cfg.agent.penalizer.name == "lbsgd":
            penalizer = LBSGDPenalizer(
                cfg.agent.penalizer.m_0,
                cfg.agent.penalizer.m_1,
                cfg.agent.penalizer.eta,
                cfg.agent.penalizer.eta_rate,
            )
        elif cfg.agent.penalizer.name == "lagrangian":
            penalizer = AugmentedLagrangianPenalizer(
                cfg.agent.penalizer.initial_lagrangian,
                cfg.agent.penalizer.initial_multiplier,
                cfg.agent.penalizer.multiplier_factor,
            )
        else:
            raise NotImplementedError
    else:
        penalizer = DummyPenalizer()
    return SafeModelBasedActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_config=cfg.agent.actor,
        critic_config=cfg.agent.critic,
        actor_optimizer_config=cfg.agent.actor_optimizer,
        critic_optimizer_config=cfg.agent.critic_optimizer,
        safety_critic_optimizer_config=cfg.agent.safety_critic_optimizer,
        initialization_scale=cfg.agent.sentiment.critics_initialization_scale,
        horizon=cfg.agent.plan_horizon,
        discount=cfg.agent.discount,
        safety_discount=cfg.agent.safety_discount,
        lambda_=cfg.agent.lambda_,
        safety_budget=episode_safety_budget,
        penalizer=penalizer,
        key=key,
        objective_sentiment=sentiment,
    )
