# SPDX-License-Identifier: Apache-2.0
"""
Controller protocol definitions for cache management and configuration.

This module defines the protocol for:
- CLEAR: Clear all caches in the server
- GET_CHUNK_SIZE: Get the chunk size configuration from the server
- PING: Liveness probe carrying the sender's `instance_id`
"""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Wire-protocol constant shared by adapter and server for `PING`.
# Senders not registered as workers (currently only the scheduler-side
# adapter) pass this value as `instance_id`. The server treats it as a
# health probe — returns True without tracking liveness. We pin it to a
# value that `random.getrandbits(63)` will never produce so a real worker
# can never collide with it.
PING_SENTINEL_INSTANCE_ID: int = 0

# Define request names for this protocol group
REQUEST_NAMES = [
    "CLEAR",
    "GET_CHUNK_SIZE",
    "PING",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for controller operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Clear all caches
        # Payload: None
        # Returns: None
        "CLEAR": ProtocolDefinition(
            payload_classes=[],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get chunk size configuration
        # Payload: None
        # Returns: int - The chunk size value
        "GET_CHUNK_SIZE": ProtocolDefinition(
            payload_classes=[],
            response_class=int,
            handler_type=HandlerType.SYNC,
        ),
        # Ping
        # Payload: instance_id (int) — sender identity for liveness tracking.
        #   Workers pass their random 63-bit `instance_id`.
        #   The scheduler passes 0 as a sentinel (no liveness tracking).
        # Returns: bool - True if the server recognizes the sender (or for the
        #   sentinel id 0); False if the sender's instance_id is unknown
        #   (terminal: adapter must clear health and stop pinging).
        "PING": ProtocolDefinition(
            payload_classes=[int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
    }
