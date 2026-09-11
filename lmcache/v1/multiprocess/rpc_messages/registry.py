# SPDX-License-Identifier: Apache-2.0
"""Small, import-time registry for transport-neutral RPC contracts."""

# Standard
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

# First Party
from lmcache.v1.multiprocess.protocols.base import RequestType

RpcMessagePair = tuple[type[Any], type[Any]]

_by_request_type: dict[RequestType, RpcMessagePair] = {}
_by_request_name: dict[str, RpcMessagePair] = {}
RPC_MESSAGE_TYPES: Mapping[str, RpcMessagePair] = MappingProxyType(_by_request_name)


def register_rpc_message_types(
    request_type: RequestType,
    request_class: type[Any],
    response_class: type[Any],
) -> None:
    """Register one local request/response pair for an RPC name.

    Domain modules call this next to their message definitions.  The registry
    only joins those local declarations; it owns no per-RPC schema itself.
    """
    if request_type in _by_request_type:
        raise RuntimeError(f"Duplicate Python RPC contract: {request_type.name}")
    pair = (request_class, response_class)
    _by_request_type[request_type] = pair
    _by_request_name[request_type.name] = pair


def get_request_message_class(request_type: RequestType) -> type[Any]:
    """Return the canonical Python request class for ``request_type``."""
    try:
        return _by_request_type[request_type][0]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {request_type}") from exc


def get_response_message_class(request_type: RequestType) -> type[Any]:
    """Return the canonical Python response class for ``request_type``."""
    try:
        return _by_request_type[request_type][1]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {request_type}") from exc


def iter_rpc_message_types() -> Iterator[tuple[RequestType, RpcMessagePair]]:
    """Iterate over locally declared RPC contracts."""
    return iter(_by_request_type.items())


def validate_rpc_message_types() -> None:
    """Require one Python request/response pair for every RPC name."""
    request_types = set(RequestType)
    registered_types = set(_by_request_type)
    if request_types != registered_types:
        missing = sorted(item.name for item in request_types - registered_types)
        unknown = sorted(item.name for item in registered_types - request_types)
        raise RuntimeError(
            "Python RPC message registry does not match RequestType: "
            f"missing={missing}, unknown={unknown}"
        )


__all__ = [
    "RPC_MESSAGE_TYPES",
    "RpcMessagePair",
    "get_request_message_class",
    "get_response_message_class",
    "iter_rpc_message_types",
    "register_rpc_message_types",
    "validate_rpc_message_types",
]
