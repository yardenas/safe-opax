from contextlib import contextmanager
from functools import wraps
from inspect import getfullargspec
import logging
import os
from typing import Any, Callable, overload

import equinox as eqx

import jax
import jax.numpy as jnp
from jmp import Policy, get_policy


_POLICY_ENV_VAR_NAME = "MIXED_PRECISION_POLICY"
_LOG = logging.getLogger(__name__)

CastOutputPredicate = Callable[[Any], bool]


def _get_policy_from_env():
    return get_policy(
        os.environ.get(
            _POLICY_ENV_VAR_NAME, "params=float32,compute=float32,output=float32"
        )
    )


@overload
def apply_mixed_precision(
    func: None = None,
    *,
    policy: Policy | None = None,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
    cast_output_predicate: CastOutputPredicate = lambda tree: tree,
):
    ...


@overload
def apply_mixed_precision(
    func: Callable[..., Any],
    *,
    policy: Policy | None = None,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
    cast_output_predicate: CastOutputPredicate = lambda tree: tree,
):
    ...


def apply_mixed_precision(
    func: Callable[..., Any] | None = None,
    *,
    policy: Policy | None = None,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
    cast_output_predicate: CastOutputPredicate | None = None,
):
    def decorator(func):
        argspec = getfullargspec(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if policy is None:
                _LOG.warning("Falling back to the default mixed precision policy.")
                concrete_policy = _get_policy_from_env()
            else:
                concrete_policy = policy
            target_inputs, target_modules = _infer_targets(
                target_input_names, target_module_names, args, kwargs, argspec
            )

            def _validate_type(
                value: Any,
                index: int | None = None,
                name: str | None = None,
            ) -> None:
                if index is not None:
                    name = argspec.args[index]
                if name in target_inputs and not _all_is_array(value):
                    raise ValueError(f"{name} must be a pytree of np.ndarray")
                if name in target_modules and not isinstance(value, eqx.Module):
                    raise ValueError(f"{name} must be an eqx.Module")

            args = list(args)
            for i, arg in enumerate(args):
                if argspec.args[i] in target_inputs + target_modules:
                    _validate_type(index=i, value=arg)
                    args[i] = apply_dtype(arg, concrete_policy.compute_dtype)
            for name, value in kwargs.items():
                if name in target_inputs + target_modules:
                    _validate_type(name=name, value=value)
                    kwargs[name] = apply_dtype(value, concrete_policy.compute_dtype)
            outs = func(*args, **kwargs)
            if cast_output_predicate is None:
                concrete_cast_output_predicate = lambda tree: jax.tree_map(
                    lambda node: eqx.is_array(node)
                    and node.dtype == concrete_policy.compute_dtype,
                    tree,
                )
            else:
                concrete_cast_output_predicate = cast_output_predicate
            to_cast, keep = eqx.partition(outs, concrete_cast_output_predicate)
            to_cast = apply_dtype(to_cast, concrete_policy.output_dtype)
            outs = eqx.combine(to_cast, keep)
            return outs

        return wrapper

    # https://stackoverflow.com/questions/3931627/how-to-build-a-decorator-with-optional-parameters/3931654#3931654
    if func is not None and callable(func):
        return decorator(func)
    else:
        return decorator


def _infer_targets(
    target_input_names, target_module_names, func_args, func_kwargs, argspec
):
    if target_input_names is None:
        target_inputs = [
            argspec.args[i] for i, arg in enumerate(func_args) if _all_is_array(arg)
        ] + [name for name, value in func_kwargs.items() if _all_is_array(value)]
    else:
        if not set(target_input_names) <= set(argspec.args):
            raise ValueError(
                "target_input_names must be a subset of the function arguments."
            )
        target_inputs = target_input_names
    if target_module_names is None:
        target_modules = [
            argspec.args[i]
            for i, arg in enumerate(func_args)
            if isinstance(arg, eqx.Module)
        ] + [
            name for name, value in func_kwargs.items() if isinstance(value, eqx.Module)
        ]
    else:
        if not set(target_module_names) <= set(argspec.args):
            raise ValueError(
                "traget_module_names must be a subset of the function arguments."
            )
        target_modules = target_module_names
    return target_inputs, target_modules


def _all_is_array(tree: Any) -> bool:
    return all(eqx.is_array(x) for x in jax.tree_util.tree_leaves(tree))


def apply_dtype(tree: Any, dtype: jnp.dtype) -> Any:
    return jax.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, tree)


@contextmanager
def mixed_precision(
    policy_string: str = "params=float32,compute=bfloat16,output=float32",
):
    os.environ[_POLICY_ENV_VAR_NAME] = policy_string
    yield
