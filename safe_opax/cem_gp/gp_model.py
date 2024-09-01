from typing import Callable
import jax
import jax.numpy as jnp
import gpjax as gpx
import equinox as eqx
from jmp import get_policy

from safe_opax.common.mixed_precision import apply_dtype, apply_mixed_precision
from safe_opax.rl.types import Policy, Prediction, ShiftScale


class GPModel(eqx.Module):
    x: jax.Array
    y: jax.Array
    posteriors: list[gpx.gps.ConjugatePosterior] = eqx.field(static=True)
    reward_fn: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    cost_fn: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        reward_fn: Callable[[jax.Array], jax.Array],
        cost_fn: Callable[[jax.Array], jax.Array],
    ):
        # FIXME (yarden): this should be over multiple dimensions
        self.x, self.y = x, y
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=gpx.kernels.RBF()
        )
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=x.shape[0])
        posterior = prior * likelihood
        posteriors = compute_posteriors(x, y, posterior)
        self.posteriors = posteriors
        self.reward_fn = reward_fn
        self.cost_fn = cost_fn

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.Array,
        policy: Policy,
    ) -> tuple[Prediction, ShiftScale]:
        def f(carry, inputs):
            prev_state = carry
            if callable(policy):
                key = inputs
                key, p_key = jax.random.split(key)
                action = policy(jax.lax.stop_gradient(prev_state.flatten()), p_key)
            else:
                action, key = inputs
            x = jnp.concatenate([prev_state, action], axis=-1)
            key, prior_key = jax.random.split(key)
            predictive_distribution = _multioutput_predict(
                self.x, self.y, x, self.posteriors
            )
            state = predictive_distribution.sample(seed=prior_key).squeeze()
            return state, (
                state,
                (
                    predictive_distribution.mean().squeeze(),
                    predictive_distribution.stddev().squeeze(),
                ),
            )

        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.Array] | jax.Array = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
        elif callable(policy):
            inputs = jax.random.split(key, horizon)
        _, (trajectory, priors) = jax.lax.scan(f, initial_state, inputs)
        reward = self.reward_fn(trajectory)
        cost = self.cost_fn(trajectory)
        out = Prediction(trajectory, reward, cost)
        return out, priors


def _multioutput_predict(x_train, y_train, x_test, posteriors):
    # TODO (yarden): Can technically stack trees then vmap, but not important now.
    distributions = []
    for i, posterior in enumerate(posteriors):
        distribution = posterior.predict(
            test_inputs=x_test[None],
            train_data=gpx.Dataset(x_train, y_train[:, i : i + 1]),
        )
        predictive_distribution = posterior.likelihood(distribution)
        distributions.append(predictive_distribution)
    return _pytrees_stack(distributions)


def _pytrees_stack(pytrees, axis=0):
    results = jax.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results


@eqx.filter_jit
@apply_mixed_precision(
    policy=get_policy("params=float32,compute=float64,output=float32"),
    target_input_names=["x", "y"],
)
def compute_posteriors(x, y, posterior):
    posterior_f64 = apply_dtype(posterior, jnp.float64)
    posteriors = []
    for i in range(y.shape[-1]):
        p, _ = gpx.fit_scipy(
            model=posterior_f64,
            train_data=gpx.Dataset(x, y[:, i : i + 1]),
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            max_iters=1,
            verbose=False
        )
        posteriors.append(p)
    return posteriors
