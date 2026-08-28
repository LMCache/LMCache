# SPDX-License-Identifier: Apache-2.0
"""Descriptor-native RPC method tokens for the LMCache gRPC transport."""

# Standard
from typing import Any, ClassVar, Optional
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


class _RpcMethodMeta(type):
    """Iterable metaclass for descriptor-derived RPC method tokens."""

    _members: tuple["RpcMethod", ...]
    _by_method_name: dict[str, "RpcMethod"]
    _by_request_name: dict[str, "RpcMethod"]

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
    _by_request_name: ClassVar[dict[str, "RpcMethod"]] = {}
    _request_name: str
    _service_name: str

    def __new__(cls, method_name: str, request_name: str, service_name: str):
        instance = str.__new__(cls, method_name)
        instance._request_name = request_name
        instance._service_name = service_name
        return instance

    @property
    def name(self) -> str:
        """Return the historical ALL_CAPS operation name."""
        return self._request_name

    @property
    def value(self) -> str:
        """Return the concrete protobuf service method name."""
        return str(self)

    @property
    def service_name(self) -> str:
        """Return the generated protobuf service that owns this method."""
        return self._service_name

    def __getnewargs__(self) -> tuple[str, str, str]:  # type: ignore[override]
        """Preserve the request-name metadata across pickle/spawn."""
        return (str(self), self.name, self.service_name)


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


def _method_name_to_request_name(method_name: str) -> str:
    """Convert a protobuf CamelCase method name to the legacy ALL_CAPS token."""
    request_name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", method_name)
    request_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", request_name)
    return request_name.upper().replace("P2_P", "P2P")


def _build_rpc_methods() -> tuple[RpcMethod, ...]:
    methods: list[RpcMethod] = []
    seen_method_names: set[str] = set()
    for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values():
        for method in service.methods:
            if method.name in seen_method_names:
                raise RuntimeError(f"Duplicate gRPC method name: {method.name}")
            seen_method_names.add(method.name)
            request_name = _method_name_to_request_name(method.name)
            rpc_method = RpcMethod(method.name, request_name, service.name)
            methods.append(rpc_method)
            setattr(RpcMethod, request_name, rpc_method)

    members = tuple(methods)
    RpcMethod._members = members
    RpcMethod._by_method_name = {str(method): method for method in members}
    RpcMethod._by_request_name = {method.name: method for method in members}
    return members


RPC_METHODS = _build_rpc_methods()
RPC = _RpcNamespace(RPC_METHODS)


def coerce_rpc_method(req_type: RpcMethod | str) -> RpcMethod:
    """Resolve an operation from either legacy or protobuf-native spelling."""
    if isinstance(req_type, RpcMethod):
        return req_type
    method = RpcMethod._by_method_name.get(req_type)
    if method is not None:
        return method
    method = RpcMethod._by_request_name.get(req_type)
    if method is not None:
        return method
    raise ValueError(f"Invalid request type: {req_type}")


def _get_typed_spec(req_type: RpcMethod | str) -> Any:
    """Return the typed RPC spec without making protocol import typed_rpc eagerly."""
    # First Party
    from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import TYPED_RPCS

    rpc_method = coerce_rpc_method(req_type)
    try:
        return TYPED_RPCS[rpc_method]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {req_type}") from exc


def get_payload_classes(req_type: RpcMethod | str) -> list[Any]:
    """
    Get the expected Python payload classes for an RPC method.

    Args:
        req_type: The legacy operation name or protobuf method to look up.

    Returns:
        The payload classes expected by the service implementation.

    Raises:
        ValueError: If the request type is not recognized.
    """
    return list(_get_typed_spec(req_type).payload_types)


def get_response_class(req_type: RpcMethod | str) -> Optional[Any]:
    """
    Get the expected Python response class for an RPC method.

    Args:
        req_type: The legacy operation name or protobuf method to look up.

    Returns:
        Expected response class, or None when the RPC has no response payload.

    Raises:
        ValueError: If the request type is not recognized.
    """
    return _get_typed_spec(req_type).response_type


def get_handler_type(req_type: RpcMethod | str) -> HandlerType:
    """
    Get the execution mode for an RPC method.

    Args:
        req_type: The legacy operation name or protobuf method to look up.

    Returns:
        The handler execution mode.

    Raises:
        ValueError: If the request type is not recognized.
    """
    return _get_typed_spec(req_type).handler_type


def requires_client_affinity(req_type: RpcMethod | str) -> bool:
    """Return whether an RPC must run on a stable per-client worker slot.

    Args:
        req_type: The legacy operation name or protobuf method to look up.

    Returns:
        True when the blocking handler requires client affinity.

    Raises:
        ValueError: If the request type is not recognized.
    """
    return _get_typed_spec(req_type).requires_client_affinity


__all__ = [
    "RPC",
    "RPC_METHODS",
    "HandlerType",
    "InstanceID",
    "KeyType",
    "RpcMethod",
    "coerce_rpc_method",
    "get_handler_type",
    "get_payload_classes",
    "get_response_class",
    "requires_client_affinity",
]
