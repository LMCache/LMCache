# SPDX-License-Identifier: Apache-2.0
"""Small, import-time registry for transport-neutral RPC contracts."""

# Standard
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

RpcMessagePair = tuple[type[Any], type[Any]]
RpcOperation = str

_by_operation: dict[RpcOperation, RpcMessagePair] = {}
RPC_MESSAGE_TYPES: Mapping[RpcOperation, RpcMessagePair] = MappingProxyType(
    _by_operation
)


def register_rpc_message_types(
    operation: RpcOperation,
    request_class: type[Any],
    response_class: type[Any],
) -> None:
    """Register one local request/response pair for an RPC route.

    Domain modules call this next to their message definitions.  The registry
    only joins those local declarations; it owns no per-RPC schema itself.
    """
    if not operation.isidentifier() or operation != operation.lower():
        raise ValueError(f"Invalid Python RPC operation: {operation!r}")
    if operation in _by_operation:
        raise RuntimeError(f"Duplicate Python RPC contract: {operation}")
    _by_operation[operation] = (request_class, response_class)


def get_request_message_class(operation: RpcOperation) -> type[Any]:
    """Return the canonical Python request class for ``operation``."""
    try:
        return _by_operation[operation][0]
    except KeyError as exc:
        raise ValueError(f"Invalid RPC operation: {operation}") from exc


def get_response_message_class(operation: RpcOperation) -> type[Any]:
    """Return the canonical Python response class for ``operation``."""
    try:
        return _by_operation[operation][1]
    except KeyError as exc:
        raise ValueError(f"Invalid RPC operation: {operation}") from exc


def iter_rpc_message_types() -> Iterator[tuple[RpcOperation, RpcMessagePair]]:
    """Iterate over locally declared RPC contracts."""
    return iter(_by_operation.items())


def validate_rpc_message_types() -> None:
    """Require every local registration to have one valid RPC route."""
    if not _by_operation:
        raise RuntimeError("Python RPC message registry is empty")


__all__ = [
    "RPC_MESSAGE_TYPES",
    "RpcMessagePair",
    "get_request_message_class",
    "get_response_message_class",
    "iter_rpc_message_types",
    "register_rpc_message_types",
    "validate_rpc_message_types",
]
