from functools import wraps
from inspect import getfullargspec
from typing import Any, Callable, TypeVar, overload

import equinox as eqx

import jax
import jax.numpy as jnp
from jmp import Policy

_T = TypeVar("_T")

@overload
def with_mixed_precision(
    func: None = None,
    *,
    policy: Policy,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
):
    ...


@overload
def with_mixed_precision(
    func: Callable[..., Any],
    *,
    policy: Policy,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
):
    ...


def with_mixed_precision(
    func: Callable[..., Any] | None = None,
    *,
    policy: Policy,
    target_module_names: list[str] | None = None,
    target_input_names: list[str] | None = None,
):
    def decorator(func):
        argspec = getfullargspec(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
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
                if name in target_inputs and not eqx.is_array(value):
                    raise ValueError(f"{name} must be an eqx.Array or np.ndarray")
                if name in target_modules and not isinstance(value, eqx.Module):
                    raise ValueError(f"{name} must be an eqx.Module")

            args = list(args)
            for i, arg in enumerate(args):
                if argspec.args[i] in target_inputs + target_modules:
                    _validate_type(index=i, value=arg)
                    args[i] = _apply_dtype(arg, policy.compute_dtype)
            for name, value in kwargs.items():
                if name in target_inputs + target_modules:
                    _validate_type(name=name, value=value)
                    kwargs[name] = _apply_dtype(value, policy.compute_dtype)
            outs = func(*args, **kwargs)
            outs = _apply_dtype(outs, policy.output_dtype)
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
            argspec.args[i] for i, arg in enumerate(func_args) if eqx.is_array(arg)
        ] + [name for name, value in func_kwargs.items() if eqx.is_array(value)]
    else:
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
        target_modules = target_module_names
    return target_inputs, target_modules


def _apply_dtype(tree: Any, dtype: jnp.dtype) -> Any:
    return jax.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, tree)
