import pytest
import jax.numpy as jnp
import jax
import equinox as eqx
from jmp import get_policy
from safe_opax.common.mixed_precision import apply_mixed_precision


@pytest.fixture
def policy():
    return get_policy("params=float32,compute=float16,output=float32")


class MockModule(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self):
        super().__init__()
        self.linear = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0))

    def __call__(self, x):
        return self.linear(x)


def test_with_mixed_precision(policy):
    @apply_mixed_precision(policy=policy)
    def mock_function(input_array, module):
        assert input_array.dtype == policy.compute_dtype
        out = module(input_array)
        assert out.dtype == policy.compute_dtype
        return out

    mock_input_array = jnp.zeros(1, dtype=jnp.float32)
    mock_module = MockModule()
    out = mock_function(mock_input_array, mock_module)
    assert out.dtype == policy.output_dtype
