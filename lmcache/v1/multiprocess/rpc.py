# SPDX-License-Identifier: Apache-2.0
"""Discover transport-neutral RPC contracts from the request client API."""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, Signature, signature
from types import MappingProxyType
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture, MessagingStream

F = TypeVar("F", bound=Callable[..., Any])
RpcOperation = str

_RPC_METHOD_ATTR = "__lmcache_rpc_method__"


def rpc_method(func: F) -> F:
    """Mark a typed ``RequestClient`` method as an RPC contract member."""
    setattr(func, _RPC_METHOD_ATTR, True)
    return func


@dataclass(frozen=True)
class RpcSpec:
    """Transport-neutral RPC name, call signature, payloads, and response."""

    operation: RpcOperation
    signature: Signature
    payload_types: tuple[Any, ...]
    response_type: Any
    streaming: bool = False

    @property
    def handler_response_type(self) -> Any:
        """Return the module handler's annotation, including a stream wrapper."""
        return (
            cast(Any, MessagingStream)[self.response_type]
            if self.streaming
            else self.response_type
        )

    def bind_payloads(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[Any, ...]:
        """Bind args/kwargs/defaults to the contract's ordered payloads."""
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
    if get_origin(return_hint) not in (MessagingFuture, MessagingStream):
        raise TypeError(
            f"RPC {operation!r} must return MessagingFuture[T] or MessagingStream[T], "
            f"got {return_hint!r}"
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
        streaming=get_origin(return_hint) is MessagingStream,
    )


@lru_cache(maxsize=1)
def get_rpc_specs() -> Mapping[RpcOperation, RpcSpec]:
    """Return RPC specs discovered from marked ``RequestClient`` methods."""
    # First Party
    from lmcache.v1.multiprocess.transport.base import RequestClient

    specs = {
        name: _build_rpc_spec(name, method)
        for name, method in RequestClient.__dict__.items()
        if callable(method) and getattr(method, _RPC_METHOD_ATTR, False)
    }
    return MappingProxyType(specs)


def get_rpc_spec(operation: RpcOperation) -> RpcSpec:
    """Return the RPC spec for ``operation`` or raise ``KeyError``."""
    return get_rpc_specs()[operation]
