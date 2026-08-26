# SPDX-License-Identifier: Apache-2.0
"""Base types and helpers for the multiprocess gRPC protocol."""

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import enum


class HandlerType(enum.Enum):
    """
    Defines how a protocol handler should be executed.

    - SYNC: Handler runs directly in the main loop (fast, non-blocking operations)
    - BLOCKING: Handler may block, run in a thread pool (I/O, slow operations)
    - NON_BLOCKING: Not supported yet (for future async handlers)
    """

    SYNC = enum.auto()
    BLOCKING = enum.auto()
    NON_BLOCKING = enum.auto()


RpcMethodName = str


def request_name_to_method_name(request_name: str) -> RpcMethodName:
    """Convert a legacy ALL_CAPS request name to the gRPC method name."""
    parts = request_name.split("_")
    return "".join(
        "P2P" if part == "P2P" else part[:1].upper() + part[1:].lower()
        for part in parts
    )


@dataclass
class ProtocolDefinition:
    """
    Defines the structure and behavior of a protocol request.

    Attributes:
        payload_classes: List of expected payload types in order
        response_class: Expected response type, or None if no response
        handler_type: How the handler should be executed (SYNC/BLOCKING/NON_BLOCKING)
        requires_client_affinity: Whether blocking calls must be routed to a
            stable per-client worker slot. Used for GPU/IPC calls whose stream
            ordering semantics depend on client affinity.
    """

    payload_classes: list[Any]
    response_class: Optional[Any]
    handler_type: HandlerType
    requires_client_affinity: bool = False
