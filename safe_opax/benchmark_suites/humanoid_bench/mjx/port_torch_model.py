import json
import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from safe_opax.benchmark_suites.humanoid_bench.mjx.flax_to_torch import TorchModel
from safe_opax.benchmark_suites.humanoid_bench.mjx.policy import Model

data_path = os.path.join(
    os.path.dirname(__file__),
    "safe_opax",
    "benchmark_suites",
    "humanoid_bench",
    "data",
    "reach_one_hand",
)
inputs = 55
num_classes = 19
pytorch_model = TorchModel(inputs, num_classes)
pytorch_model.load_state_dict(torch.load(data_path + "/torch_model.pt"))

# Create the Equinox model and set the weights
key = jax.random.PRNGKey(0)
equinox_model = Model(inputs, num_classes, key=key)


def load_weights(path, leaf):
    path_pieces = []
    for path_elem in path:
        if isinstance(path_elem, jax.tree_util.GetAttrKey):
            path_pieces.append(path_elem.name)
        elif isinstance(path_elem, jax.tree_util.SequenceKey):
            path_pieces.append(str(path_elem.idx))
        else:
            raise ValueError(f"Unsupported path type {type(path_elem)}")
    weight = pytorch_model.state_dict()[".".join(path_pieces)]
    weight = jnp.asarray(np.asarray(weight))
    assert weight.shape == leaf.shape
    assert weight.dtype == leaf.dtype
    return weight


equinox_model = jax.tree_util.tree_map_with_path(load_weights, equinox_model)


def test_model_equivalence():
    # Create random input
    input_data = np.random.randn(1, inputs)

    # Get outputs from PyTorch and Equinox models
    pytorch_output = pytorch_model(torch.from_numpy(input_data.astype(np.float32)))
    equinox_output = eqx.filter_vmap(equinox_model)(input_data)

    # Compare the outputs
    assert jnp.allclose(equinox_output, pytorch_output.detach().numpy(), atol=1e-6)


test_model_equivalence()


def dump_model(model: Model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        parameters = dict(inputs=inputs, num_classes=num_classes)
        serialized_config = json.dumps(parameters)
        f.write((serialized_config + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


dump_model(equinox_model, data_path + "/model.ckpt")
