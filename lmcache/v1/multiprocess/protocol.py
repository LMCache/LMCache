# SPDX-License-Identifier: Apache-2.0
"""
Main RPC protocol for the LMCache core server and clients.

This module serves as the main entry point for the protocol system.
All protocol definitions are now organized in the protocols/ subdirectory:
- protocols/base.py: RequestType enum, HandlerType, ProtocolDefinition
- protocols/engine.py: Core KV cache operations (REGISTER, STORE, RETRIEVE, etc.)
- protocols/controller.py: Cache management operations (CLEAR, GET_CHUNK_SIZE)
- protocols/debug.py: Debug and testing operations (NOOP)

The protocol definitions are loaded and validated during initialization.
"""

# Standard
# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.rpc_messages import (
    RPC_MESSAGE_TYPES,
    RpcRequest,
    RpcResponse,
)

# Initialize the protocol system
# This loads all protocol definitions and validates them against the RequestType enum
_PROTOCOL_DEFINITIONS = initialize_protocols()

_request_names = {request_type.name for request_type in RequestType}
_message_names = set(RPC_MESSAGE_TYPES)
if _request_names != _message_names:
    missing = sorted(_request_names - _message_names)
    unknown = sorted(_message_names - _request_names)
    raise RuntimeError(
        "Python RPC message registry does not match RequestType: "
        f"missing={missing}, unknown={unknown}"
    )

# Type aliases for backwards compatibility
InstanceID = int
KeyType = IPCCacheServerKey


def get_request_message_class(req_type: RequestType) -> type[RpcRequest]:
    """Return the transport-neutral request message class for an RPC."""
    try:
        return RPC_MESSAGE_TYPES[req_type.name][0]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {req_type}") from exc


def get_response_message_class(req_type: RequestType) -> type[RpcResponse]:
    """Return the transport-neutral response message class for an RPC."""
    try:
        return RPC_MESSAGE_TYPES[req_type.name][1]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {req_type}") from exc


def get_handler_type(req_type: RequestType) -> HandlerType:
    """
    Get the handler type for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        The handler type (SYNC, BLOCKING, or NON_BLOCKING)

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.handler_type
    else:
        raise ValueError(f"Invalid request type: {req_type}")
