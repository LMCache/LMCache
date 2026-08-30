# SPDX-License-Identifier: Apache-2.0
"""Descriptor-native RPC method tokens for the LMCache gRPC transport."""

# Standard
from typing import Any, Callable, ClassVar, TypeVar
import re

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)

# Generated protobuf classes are dynamic and opaque to static analysis.
lmcache_mq_pb2: Any = _pb2_typed

# Type aliases kept for older callers.
InstanceID = int
KeyType = IPCCacheServerKey

F = TypeVar("F", bound=Callable[..., Any])

_GRPC_HANDLER_TYPE_ATTR = "__lmcache_grpc_handler_type__"
_GRPC_REQUIRES_CLIENT_AFFINITY_ATTR = "__lmcache_grpc_requires_client_affinity__"


def grpc_method(
    handler_type: HandlerType = HandlerType.SYNC,
    *,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Attach server scheduling metadata to a concrete gRPC method.

    Args:
        handler_type: How the server should run this RPC implementation.
        requires_client_affinity: Whether blocking calls must keep requests from
            one client on the same worker thread.

    Returns:
        A decorator that preserves the wrapped function.
    """

    def decorate(func: F) -> F:
        setattr(func, _GRPC_HANDLER_TYPE_ATTR, handler_type)
        setattr(func, _GRPC_REQUIRES_CLIENT_AFFINITY_ATTR, requires_client_affinity)
        return func

    return decorate


def get_grpc_method_options(handler: Callable[..., Any]) -> tuple[HandlerType, bool]:
    """Return scheduling metadata attached by :func:`grpc_method`."""
    source = getattr(handler, "__func__", handler)
    return (
        getattr(source, _GRPC_HANDLER_TYPE_ATTR, HandlerType.SYNC),
        getattr(source, _GRPC_REQUIRES_CLIENT_AFFINITY_ATTR, False),
    )


class _RpcMethodMeta(type):
    """Iterable metaclass for descriptor-derived RPC method tokens."""

    _members: tuple["RpcMethod", ...]
    _by_method_name: dict[str, "RpcMethod"]

    def __iter__(cls):
        return iter(cls._members)

    def __len__(cls) -> int:
        return len(cls._members)

    def __contains__(cls, item: object) -> bool:
        return item in cls._members


class RpcMethod(str, metaclass=_RpcMethodMeta):  # type: ignore[misc]
    """String token representing one protobuf service method."""

    _members: ClassVar[tuple["RpcMethod", ...]] = ()
    _by_method_name: ClassVar[dict[str, "RpcMethod"]] = {}
    _service_name: str

    def __new__(cls, method_name: str, service_name: str):
        instance = str.__new__(cls, method_name)
        instance._service_name = service_name
        return instance

    @property
    def name(self) -> str:
        """Return the protobuf method name."""
        return str(self)

    @property
    def value(self) -> str:
        """Return the concrete protobuf service method name."""
        return str(self)

    @property
    def service_name(self) -> str:
        """Return the generated protobuf service that owns this method."""
        return self._service_name

    @property
    def client_method_name(self) -> str:
        """Return the snake_case client method name for this RPC."""
        return _method_name_to_client_method_name(str(self))

    def __getnewargs__(self) -> tuple[str, str]:  # type: ignore[override]
        """Preserve service metadata across pickle/spawn."""
        return (str(self), self.service_name)


class _RpcNamespace:
    """CamelCase attribute access for descriptor-native RPC methods."""

    def __init__(self, methods: tuple[RpcMethod, ...]) -> None:
        self._methods = methods
        for method in methods:
            setattr(self, str(method), method)

    def __iter__(self):
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def __getattr__(self, name: str) -> RpcMethod:
        for method in self._methods:
            if str(method) == name:
                return method
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")


def _method_name_to_client_method_name(method_name: str) -> str:
    """Convert a protobuf CamelCase method name to a snake_case client method."""
    client_name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", method_name)
    client_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", client_name)
    return client_name.lower().replace("p2_p", "p2p")


def _build_rpc_methods() -> tuple[RpcMethod, ...]:
    methods: list[RpcMethod] = []
    seen_method_names: set[str] = set()
    for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values():
        for method in service.methods:
            if method.name in seen_method_names:
                raise RuntimeError(f"Duplicate gRPC method name: {method.name}")
            seen_method_names.add(method.name)
            rpc_method = RpcMethod(method.name, service.name)
            methods.append(rpc_method)
            setattr(RpcMethod, method.name, rpc_method)

    members = tuple(methods)
    RpcMethod._members = members
    RpcMethod._by_method_name = {str(method): method for method in members}
    return members


RPC_METHODS = _build_rpc_methods()
RPC = _RpcNamespace(RPC_METHODS)


def coerce_rpc_method(req_type: RpcMethod | str) -> RpcMethod:
    """Resolve a protobuf RPC method token."""
    if isinstance(req_type, RpcMethod):
        return req_type
    method = RpcMethod._by_method_name.get(req_type)
    if method is not None:
        return method
    raise ValueError(f"Invalid RPC method: {req_type}")


def get_payload_classes(req_type: RpcMethod | str) -> list[Any]:
    """
    Get the protobuf request class for an RPC method.

    Args:
        req_type: The protobuf RPC method to look up.

    Returns:
        A single-item list containing the generated protobuf request class.
    """
    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
        get_request_message_class,
    )

    return [get_request_message_class(req_type)]


def get_response_class(req_type: RpcMethod | str) -> Any:
    """
    Get the protobuf response class for an RPC method.

    Args:
        req_type: The protobuf RPC method to look up.

    Returns:
        The generated protobuf response class.
    """
    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
        get_response_message_class,
    )

    return get_response_message_class(req_type)


__all__ = [
    "RPC",
    "RPC_METHODS",
    "HandlerType",
    "InstanceID",
    "KeyType",
    "RpcMethod",
    "coerce_rpc_method",
    "get_grpc_method_options",
    "get_payload_classes",
    "get_response_class",
    "grpc_method",
]
