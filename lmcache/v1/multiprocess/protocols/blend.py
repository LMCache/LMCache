# SPDX-License-Identifier: Apache-2.0
"""
Debug protocol definitions for testing and monitoring.

This module defines the protocol for:
- NOOP: No-operation command for testing connectivity and as a heartbeat
"""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "CB_LOOKUP_PRE_COMPUTED",
    "CB_STORE_PRE_COMPUTED",
    "CB_RETRIEVE_PRE_COMPUTED",
    "CB_STORE_FINAL",
    "CB_REGISTER_KV_CACHE",
    "CB_UNREGISTER_KV_CACHE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for debug operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Lookup pre-computed chunks
        # Payload:
        #   - token_ids: List[int] - List of input token IDs to look up
        # Returns: List of tuples (start, end) indicating the match ranges
        "CB_LOOKUP_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[list[int]],
            response_class=list[tuple[int, int]],
            handler_type=HandlerType.BLOCKING,
        ),
        # Store pre-computed chunks
        # Payload:
        #   - token_ids: List[int] - List of input token IDs of the request
        #   - offset: int - The starting offset in the CB KV cache buffer
        # Returns: None
        "CB_STORE_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[list[int], int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve pre-computed chunks
        # Payload:
        #   - token_ids: List[int] - List of input token IDs of the request
        #   - ranges: List[tuple[int, int]] - List of tuples (start, end) indicating
        #                                     the match ranges to retrieve
        #   - offset: int - The starting offset in the CB KV cache buffer
        # Returns: bool indicating the success of the retrieval
        "CB_RETRIEVE_PRE_COMPUTED": ProtocolDefinition(
            payload_classes=[list[int], list[tuple[int, int]], int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # Store final chunks after processing
        # Payload:
        #   - token_ids: List[int] - List of input token IDs of the request
        #   - offset: int - The starting offset in the CB KV cache buffer
        # Returns: None
        "CB_STORE_FINAL": ProtocolDefinition(
            payload_classes=[list[int], int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Register CB KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - kv_cache: KVCache - The CB KV cache configuration
        # Returns: None
        "CB_REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int, "KVCache"],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister CB KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "CB_UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
    }
