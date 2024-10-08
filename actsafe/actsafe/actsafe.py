from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig

from actsafe.common.learner import Learner
from actsafe.actsafe import rssm
from actsafe.actsafe.exploration import UniformExploration, make_exploration
from actsafe.actsafe.make_actor_critic import make_actor_critic
from actsafe.actsafe.multi_reward import MultiRewardBridge
from actsafe.actsafe.replay_buffer import ReplayBuffer
from actsafe.actsafe.sentiment import make_sentiment
from actsafe.actsafe.world_model import WorldModel, evaluate_model, variational_step
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.metrics import MetricsMonitor
from actsafe.rl.trajectory import TrajectoryData, Transition
from actsafe.rl.types import FloatArray, Report
from actsafe.rl.utils import Count, PRNGSequence, Until, add_to_buffer


@eqx.filter_jit
def policy(policy_fn, model, prev_state, observation, key):
    def per_env_policy(prev_state, observation, key):
        model_key, policy_key = jax.random.split(key)
        current_rssm_state = model.infer_state(
            prev_state.rssm_state, observation, prev_state.prev_action, model_key
        )
        action = policy_fn(current_rssm_state.flatten(), policy_key)
        return action, AgentState(current_rssm_state, action)

    observation = preprocess(observation)
    return jax.vmap(per_env_policy)(
        prev_state, observation, jax.random.split(key, observation.shape[0])
    )


class AgentState(NamedTuple):
    rssm_state: rssm.State
    prev_action: jax.Array

    @classmethod
    def init(cls, batch_size: int, cell: rssm.RSSM, action_dim: int) -> "AgentState":
        rssm_state = cell.init
        rssm_state = jax.tree_map(
            lambda x: jnp.repeat(x[None], batch_size, 0), rssm_state
        )
        prev_action = jnp.zeros((batch_size, action_dim))
        self = cls(rssm_state, prev_action)
        return self


class ActSafe:
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        self.config = config
        num_rewards = 2 if self.config.agent.unsupervised else 1
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            sequence_length=config.agent.replay_buffer.sequence_length,
            batch_size=config.agent.replay_buffer.batch_size,
            capacity=config.agent.replay_buffer.capacity,
            num_rewards=num_rewards,
        )
        self.prng = PRNGSequence(config.training.seed)
        action_dim = int(np.prod(action_space.shape))
        assert len(observation_space.shape) == 3
        self.model = WorldModel(
            image_shape=observation_space.shape,
            action_dim=action_dim,
            key=next(self.prng),
            ensemble_size=config.agent.sentiment.ensemble_size,
            initialization_scale=config.agent.sentiment.model_initialization_scale,
            num_rewards=num_rewards,
            **config.agent.model,
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.actor_critic = make_actor_critic(
            config,
            config.training.safe,
            config.agent.model.stochastic_size + config.agent.model.deterministic_size,
            action_dim,
            next(self.prng),
            make_sentiment(self.config.agent.sentiment.objective_optimism),
            make_sentiment(self.config.agent.sentiment.constraint_pessimism),
        )
        self.exploration = make_exploration(
            config,
            action_dim,
            next(self.prng),
        )
        self.offline = UniformExploration(action_dim)
        self.state = AgentState.init(
            config.training.parallel_envs, self.model.cell, action_dim
        )
        environment_steps_per_agent_step = (
            config.training.parallel_envs * config.training.action_repeat
        )
        self.should_train = Count(
            config.agent.train_every,
            environment_steps_per_agent_step,
        )
        self.should_explore = Until(
            config.agent.exploration_steps, environment_steps_per_agent_step
        )
        self.should_collect_offline = Until(
            config.agent.offline_steps, environment_steps_per_agent_step
        )
        learn_model_steps = (
            config.agent.learn_model_steps
            if config.agent.learn_model_steps is not None
            else float("inf")
        )
        self.learn_model = Until(learn_model_steps, environment_steps_per_agent_step)
        self.metrics_monitor = MetricsMonitor()
        self.zero_shot = False

    def __call__(
        self,
        observation: FloatArray,
        train: bool = False,
    ) -> FloatArray:
        if train and self.should_train() and not self.replay_buffer.empty:
            self.update()
        if self.should_collect_offline():
            policy_fn = self.offline.get_policy()
        else:
            policy_fn = (
                self.exploration.get_policy()
                if self.should_explore()
                else self.actor_critic.actor.act
            )
        self.should_explore.tick()
        self.should_collect_offline.tick()
        self.learn_model.tick()
        actions, self.state = policy(
            policy_fn,
            self.model,
            self.state,
            observation,
            next(self.prng),
        )
        return np.asarray(actions)

    def observe(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.replay_buffer,
            trajectory,
            self.config.training.scale_reward,
        )
        self.state = jax.tree_map(lambda x: jnp.zeros_like(x), self.state)

    def observe_transition(self, transition: Transition) -> None:
        pass

    def update(self):
        total_steps = self.config.agent.update_steps
        if (
            not self.should_explore()
            and self.config.agent.unsupervised
            and not self.learn_model()
            and not self.zero_shot
        ):
            total_steps = self.config.agent.zero_shot_steps
        for batch in self.replay_buffer.sample(total_steps):
            batch = TrajectoryData(
                batch.observation,
                batch.next_observation,
                batch.action,
                batch.reward * self.config.agent.reward_scale,
                batch.cost,
            )
            inferred_rssm_states = self.update_model(batch)
            initial_states = inferred_rssm_states.reshape(
                -1, inferred_rssm_states.shape[-1]
            )
            if self.should_explore():
                if not self.config.agent.unsupervised:
                    outs = self.actor_critic.update(
                        MultiRewardBridge(self.model, 0),
                        initial_states,
                        next(self.prng),
                    )
                else:
                    outs = {}
                exploration_outs = self.exploration.update(
                    self.model, initial_states, next(self.prng)
                )
                outs.update(exploration_outs)
            else:
                if self.config.agent.unsupervised:
                    if self.learn_model():
                        index = 0
                    else:
                        self.zero_shot = True
                        index = self.config.agent.reward_index
                else:
                    index = -1
                outs = self.actor_critic.update(
                    MultiRewardBridge(self.model, index),
                    initial_states,
                    next(self.prng),
                )
            for k, v in outs.items():
                self.metrics_monitor[k] = v

    def update_model(self, batch: TrajectoryData) -> jax.Array:
        features, actions = _prepare_features(batch)
        inference_only = self.config.agent.unsupervised and not self.learn_model()
        (self.model, self.model_learner.state), (loss, rest) = variational_step(
            features,
            actions,
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
            self.config.agent.beta,
            self.config.agent.free_nats,
            self.config.agent.kl_mix,
            inference_only=inference_only,
        )
        self.metrics_monitor["agent/model/loss"] = float(loss.mean())
        self.metrics_monitor["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.metrics_monitor["agent/model/kl"] = float(rest["kl_loss"].mean())
        return rest["states"].flatten()

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        if self.config.agent.evaluate_model:
            batch = next(self.replay_buffer.sample(1))
            features, actions = _prepare_features(batch)
            video = evaluate_model(self.model, features, actions, next(self.prng))
            return Report(metrics=metrics, videos={"agent/model/prediction": video})
        else:
            return Report(metrics=metrics)


@jax.jit
def _prepare_features(batch: TrajectoryData) -> tuple[rssm.Features, jax.Array]:
    terminals = jnp.zeros_like(batch.reward)
    features = rssm.Features(
        jnp.asarray(preprocess(batch.next_observation)),
        jnp.asarray(batch.reward),
        jnp.asarray(batch.cost),
        jnp.asarray(terminals),
    )
    actions = jnp.asarray(batch.action)
    return features, actions


def preprocess(image):
    return image / 255.0 - 0.5
