# SPDX-License-Identifier: Apache-2.0
"""
Engine protocol definitions for core KV cache operations.

This module defines the protocol for:
- REGISTER_KV_CACHE: Register a KV cache instance with the server
- UNREGISTER_KV_CACHE: Unregister a KV cache instance
- STORE: Store KV cache blocks to the server
- RETRIEVE: Retrieve KV cache blocks from the server
- LOOKUP: Check if keys exist in the cache
- END_SESSION: End a session and clean up associated resources
"""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey, KVCache
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "REGISTER_KV_CACHE",
    "UNREGISTER_KV_CACHE",
    "STORE",
    "RETRIEVE",
    "LOOKUP",
    "END_SESSION",
]

# Type alias for cache keys
KeyType = IPCCacheEngineKey


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for engine operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Register KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - kv_cache: KVCache - The KV cache configuration
        # Returns: None
        "REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int, KVCache],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Store KV cache blocks
        # Payload:
        #   - keys: list[KeyType] - Cache keys to store
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs containing the data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "STORE": ProtocolDefinition(
            payload_classes=[list[KeyType], int, list[int], bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve KV cache blocks
        # Payload:
        #   - keys: list[KeyType] - Cache keys to retrieve
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs to store retrieved data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        # Returns: tuple[bytes, list[bool]] - (CUDA event handle, list of success flags)
        "RETRIEVE": ProtocolDefinition(
            payload_classes=[list[KeyType], int, list[int], bytes],
            response_class=tuple[bytes, list[bool]],
            handler_type=HandlerType.BLOCKING,
        ),
        # Lookup keys in cache
        # Payload:
        #   - keys: list[KeyType] - Cache keys to look up
        # Returns: int - Number of keys found in cache
        "LOOKUP": ProtocolDefinition(
            payload_classes=[list[KeyType]],
            response_class=int,
            handler_type=HandlerType.BLOCKING,
        ),
        # End session
        # Payload:
        #   - request_id: str - Request ID of the session to end
        # Returns: None
        "END_SESSION": ProtocolDefinition(
            payload_classes=[str],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
    }
