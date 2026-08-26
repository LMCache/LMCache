# SPDX-License-Identifier: Apache-2.0
"""
Main RPC protocol for the LMCache core server and clients.

This module serves as the main entry point for the protocol system.
All protocol definitions are now organized in the protocols/ subdirectory:
- protocols/base.py: HandlerType, ProtocolDefinition, gRPC naming helpers
- protocols/engine.py: Core KV cache operations (REGISTER, STORE, RETRIEVE, etc.)
- protocols/controller.py: Cache management operations (CLEAR, GET_CHUNK_SIZE)
- protocols/debug.py: Debug and testing operations (NOOP)

The protocol definitions are loaded and validated during initialization.
"""

# Standard
from typing import Any, ClassVar, Optional

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import (
    HandlerType,
    request_name_to_method_name,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)

# Initialize the protocol system
_REQUEST_NAME_DEFINITIONS = initialize_protocols()
lmcache_mq_pb2: Any = _pb2_typed

# Type aliases for backwards compatibility
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
    """String token representing one typed gRPC method."""

    _members: ClassVar[tuple["RpcMethod", ...]] = ()
    _by_method_name: ClassVar[dict[str, "RpcMethod"]] = {}
    _by_request_name: ClassVar[dict[str, "RpcMethod"]] = {}
    _request_name: str

    def __new__(cls, method_name: str, request_name: str):
        instance = str.__new__(cls, method_name)
        instance._request_name = request_name
        return instance

    @property
    def name(self) -> str:
        """Return the historical ALL_CAPS request name."""
        return self._request_name

    @property
    def value(self) -> str:
        """Return the concrete gRPC method name."""
        return str(self)

    def __getnewargs__(self) -> tuple[str, str]:  # type: ignore[override]
        """Preserve the request-name metadata across pickle/spawn."""
        return (str(self), self.name)


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


def _build_rpc_methods() -> tuple[tuple[RpcMethod, ...], dict[RpcMethod, Any]]:
    service = lmcache_mq_pb2.DESCRIPTOR.services_by_name["MessageQueue"]
    request_name_by_method = {
        request_name_to_method_name(request_name): request_name
        for request_name in _REQUEST_NAME_DEFINITIONS
    }
    methods: list[RpcMethod] = []
    definitions: dict[RpcMethod, Any] = {}
    for method in service.methods:
        request_name = request_name_by_method[method.name]
        rpc_method = RpcMethod(method.name, request_name)
        methods.append(rpc_method)
        definitions[rpc_method] = _REQUEST_NAME_DEFINITIONS[request_name]
        setattr(RpcMethod, request_name, rpc_method)

    members = tuple(methods)
    RpcMethod._members = members
    RpcMethod._by_method_name = {str(method): method for method in members}
    RpcMethod._by_request_name = {method.name: method for method in members}
    return members, definitions


RPC_METHODS, _PROTOCOL_DEFINITIONS = _build_rpc_methods()
RPC = _RpcNamespace(RPC_METHODS)


def coerce_rpc_method(req_type: RpcMethod | str) -> RpcMethod:
    """Resolve a request token from either legacy or gRPC-native spelling."""
    if isinstance(req_type, RpcMethod):
        return req_type
    method = RpcMethod._by_method_name.get(req_type)
    if method is not None:
        return method
    method = RpcMethod._by_request_name.get(req_type)
    if method is not None:
        return method
    raise ValueError(f"Invalid request type: {req_type}")


def get_payload_classes(req_type: RpcMethod | str) -> list[Any]:
    """
    Get the expected payload classes for a request type.

    Args:
        req_type: The request type or gRPC method to look up

    Returns:
        List of expected payload classes in order

    Raises:
        ValueError: If the request type is not recognized
    """
    rpc_method = coerce_rpc_method(req_type)
    if pd := _PROTOCOL_DEFINITIONS.get(rpc_method, None):
        return pd.payload_classes
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_response_class(req_type: RpcMethod | str) -> Optional[Any]:
    """
    Get the expected response class for a request type.

    Args:
        req_type: The request type or gRPC method to look up

    Returns:
        Expected response class, or None if no response

    Raises:
        ValueError: If the request type is not recognized
    """
    rpc_method = coerce_rpc_method(req_type)
    if pd := _PROTOCOL_DEFINITIONS.get(rpc_method, None):
        return pd.response_class
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_handler_type(req_type: RpcMethod | str) -> HandlerType:
    """
    Get the handler type for a request type.

    Args:
        req_type: The request type or gRPC method to look up

    Returns:
        The handler type (SYNC, BLOCKING, or NON_BLOCKING)

    Raises:
        ValueError: If the request type is not recognized
    """
    rpc_method = coerce_rpc_method(req_type)
    if pd := _PROTOCOL_DEFINITIONS.get(rpc_method, None):
        return pd.handler_type
    else:
        raise ValueError(f"Invalid request type: {req_type}")
