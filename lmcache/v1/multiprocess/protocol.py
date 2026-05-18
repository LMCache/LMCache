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
from typing import Any, Optional

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType

# Initialize the protocol system
# This loads all protocol definitions and validates them against the RequestType enum
_PROTOCOL_DEFINITIONS = initialize_protocols()

LMCACHE_MP_PROTOCOL_VERSION = 1
"""Version of the ZMQ/msgpack LMCache MP protocol envelope.

Version 1 is the existing unversioned wire format:
``[request_uid, request_type, *payloads]`` from DEALER clients and
``[identity, request_uid, request_type, *payloads]`` on the ROUTER server.
The constant is additive and does not add a required wire frame.
"""

# Type aliases for backwards compatibility
InstanceID = int
KeyType = IPCCacheEngineKey


def _type_name(cls: Any) -> str:
    if cls is None:
        return "None"
    if isinstance(cls, type):
        return cls.__name__
    return str(cls).replace("typing.", "")


def get_protocol_schema() -> dict[str, Any]:
    """Return a JSON-serializable schema view of the current MP protocol."""
    return {
        "protocol_version": LMCACHE_MP_PROTOCOL_VERSION,
        "serialization_format": "msgpack via msgspec",
        "request_envelope": [
            "request_uid:uint",
            "request_type:RequestType",
            "payloads:list[msgpack]",
        ],
        "response_envelope": [
            "request_uid:uint",
            "request_type:RequestType",
            "response?:msgpack",
        ],
        "error_schema": (
            "Handlers log exceptions and may omit a response in the current "
            "protocol; there is no typed wire error frame in version 1."
        ),
        "backward_compatibility": (
            "Protocol version 1 preserves the existing unversioned envelope. "
            "New fields in msgspec structs must remain backward-compatible "
            "through defaults or new request types."
        ),
        "request_types": {
            req_type.name: {
                "value": req_type.value,
                "payload_schema": [
                    _type_name(cls) for cls in definition.payload_classes
                ],
                "response_schema": _type_name(definition.response_class),
                "handler_type": definition.handler_type.name,
            }
            for req_type, definition in sorted(
                _PROTOCOL_DEFINITIONS.items(), key=lambda item: item[0].value
            )
        },
    }


def get_payload_classes(req_type: RequestType) -> list[Any]:
    """
    Get the expected payload classes for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        List of expected payload classes in order

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.payload_classes
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_response_class(req_type: RequestType) -> Optional[Any]:
    """
    Get the expected response class for a request type.

    Args:
        req_type: The request type to look up

    Returns:
        Expected response class, or None if no response

    Raises:
        ValueError: If the request type is not recognized
    """
    if pd := _PROTOCOL_DEFINITIONS.get(req_type, None):
        return pd.response_class
    else:
        raise ValueError(f"Invalid request type: {req_type}")


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
