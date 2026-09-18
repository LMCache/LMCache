# SPDX-License-Identifier: Apache-2.0
"""
KV event channel protocol definitions.

This module defines the protocol for:
- POLL_KV_EVENTS: Read the server's cache-event log (host-cache store
  completions and evictions, L2 stores and deletes) after a client-held
  cursor, so an engine worker can republish them as KV events for KV-aware
  routing.
"""

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventPollResult
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "POLL_KV_EVENTS",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for the KV event channel.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Read the cache-event log after a cursor
        # Payload:
        #   - model_name: str - only records for this model's keys are returned
        #   - cursor: int - sequence number of the last record the client
        #     consumed (0 on first contact)
        #   - max_events: int - upper bound on returned records (>= 1)
        # Returns: KVEventPollResult - see lmcache.v1.multiprocess.custom_types
        # SYNC: the handler only copies records out of an in-memory log.
        "POLL_KV_EVENTS": ProtocolDefinition(
            payload_classes=[str, int, int],
            response_class=KVEventPollResult,
            handler_type=HandlerType.SYNC,
        ),
    }
