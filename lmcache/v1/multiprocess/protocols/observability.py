# SPDX-License-Identifier: Apache-2.0
"""
Observability protocol definitions.

This module defines protocols for:
- REPORT_BLOCK_ALLOCATION: Report vLLM GPU block allocation events
  (fire-and-forget, no response)
- POLL_KV_EVENTS: Read the server's cache-event log after a client-held
  cursor, so an engine worker can republish the records as KV events
"""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    KVEventPollResult,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "REPORT_BLOCK_ALLOCATION",
    "POLL_KV_EVENTS",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for observability operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Report vLLM block allocation
        # Payload:
        #   - instance_id: int - scheduler instance ID
        #   - model_name: str - model name from the adapter
        #   - records: list[BlockAllocationRecord] - allocation records
        # Returns: None (fire-and-forget)
        "REPORT_BLOCK_ALLOCATION": ProtocolDefinition(
            payload_classes=[int, str, list[BlockAllocationRecord]],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Read the cache-event log after a cursor
        # Payload:
        #   - model_name: str - only records for this model's keys are returned
        #   - cursor: int - sequence number of the last record consumed
        #     (0 on first contact)
        #   - max_events: int - upper bound on returned records (>= 1)
        # Returns: KVEventPollResult - see lmcache.v1.multiprocess.custom_types
        # SYNC: the handler only copies records out of an in-memory log.
        "POLL_KV_EVENTS": ProtocolDefinition(
            payload_classes=[str, int, int],
            response_class=KVEventPollResult,
            handler_type=HandlerType.SYNC,
        ),
    }
