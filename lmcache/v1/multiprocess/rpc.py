# SPDX-License-Identifier: Apache-2.0
"""Discover transport-neutral RPC contracts from the request client API."""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, Signature, signature
from types import MappingProxyType
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture

F = TypeVar("F", bound=Callable[..., Any])
RpcOperation = str

_RPC_METHOD_ATTR = "__lmcache_rpc_method__"


def rpc_method(func: F) -> F:
    """Mark a request-client method as part of the RPC contract.

    Args:
        func: Typed method on ``RequestClient``.

    Returns:
        The unchanged method with RPC metadata attached.
    """
    setattr(func, _RPC_METHOD_ATTR, True)
    return func


@dataclass(frozen=True)
class RpcSpec:
    """Describe one transport-neutral RPC.

    Args:
        operation: Stable snake-case operation name.
        signature: Client-call signature without ``self``.
        payload_types: Declared payload types in parameter order.
        response_type: Value type returned through ``MessagingFuture``.
    """

    operation: RpcOperation
    signature: Signature
    payload_types: tuple[Any, ...]
    response_type: Any

    def bind_payloads(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        """Bind a client invocation to ordered transport payloads.

        Args:
            args: Positional client arguments.
            kwargs: Keyword client arguments.

        Returns:
            Payload values in the order declared by the RPC contract.

        Raises:
            TypeError: If the invocation does not match the contract signature.
        """
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.arguments.values())


def _build_rpc_spec(operation: str, method: Callable[..., Any]) -> RpcSpec:
    method_signature = signature(method)
    hints = get_type_hints(method)
    parameters = tuple(method_signature.parameters.values())[1:]
    for parameter in parameters:
        if parameter.kind not in (
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"RPC {operation!r} uses unsupported parameter kind "
                f"{parameter.kind.name}"
            )
        if parameter.name not in hints:
            raise TypeError(
                f"RPC {operation!r} parameter {parameter.name!r} has no type hint"
            )

    return_hint = hints.get("return")
    if get_origin(return_hint) is not MessagingFuture:
        raise TypeError(
            f"RPC {operation!r} must return MessagingFuture[T], got {return_hint!r}"
        )
    response_args = get_args(return_hint)
    if len(response_args) != 1:
        raise TypeError(f"RPC {operation!r} has no concrete response type")

    client_signature = method_signature.replace(parameters=parameters)
    return RpcSpec(
        operation=operation,
        signature=client_signature,
        payload_types=tuple(hints[parameter.name] for parameter in parameters),
        response_type=response_args[0],
    )


@lru_cache(maxsize=1)
def get_rpc_specs() -> Mapping[RpcOperation, RpcSpec]:
    """Discover all RPC contracts declared on ``RequestClient``.

    Returns:
        A read-only mapping from operation names to RPC specifications.

    Raises:
        TypeError: If a declared RPC has an incomplete or unsupported signature.
    """
    # Local import avoids a cycle while RequestClient applies @rpc_method.
    # First Party
    from lmcache.v1.multiprocess.transport.base import RequestClient

    specs = {
        name: _build_rpc_spec(name, method)
        for name, method in RequestClient.__dict__.items()
        if callable(method) and getattr(method, _RPC_METHOD_ATTR, False)
    }
    return MappingProxyType(specs)


def get_rpc_spec(operation: RpcOperation) -> RpcSpec:
    """Return the contract for one operation.

    Args:
        operation: Stable snake-case RPC name.

    Returns:
        The matching RPC specification.

    Raises:
        KeyError: If the operation is not part of the request-client contract.
    """
    return get_rpc_specs()[operation]
