# SPDX-License-Identifier: Apache-2.0
"""
Layer-wise transfer protocol definitions.

This module defines the protocol for the layer-wise KV retrieve path, which
is served by ``LMCacheLayerwiseTransferModule`` in
``lmcache.v1.multiprocess.modules.lmcache_driven_transfer_layerwise``.

Keeping these definitions out of ``protocols/engine.py`` means the default
(per-chunk) transfer protocol is byte-for-byte unchanged when layer-wise mode
is disabled:

- REGISTER_LAYERWISE_IPC_EVENT_POOL: Fetch the IPC event pool for a registered instance
- RETRIEVE_LAYERWISE: Retrieve KV cache blocks in layer-major order
"""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "REGISTER_LAYERWISE_IPC_EVENT_POOL",
    "RETRIEVE_LAYERWISE",
]

# Type alias for cache keys
KeyType = IPCCacheServerKey


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for layer-wise transfer operations.

    Returns:
        Dictionary mapping request names to their protocol definitions.
    """
    return {
        # Fetch the pre-allocated per-layer IPC event pool for an instance
        # that was already registered via REGISTER_KV_CACHE. Issued once,
        # right after registration, so REGISTER_KV_CACHE keeps its ``None``
        # response for every non-layer-wise deployment.
        # Payload:
        #   - instance_id: int - The GPU instance ID (such as PID)
        # Returns: tuple[int, list[bytes]] - (layerwise_batch, pool handles).
        #   The handle list is empty when layer-wise mode is disabled.
        "REGISTER_LAYERWISE_IPC_EVENT_POOL": ProtocolDefinition(
            payload_classes=[int],
            response_class=tuple[int, list[bytes]],
            handler_type=HandlerType.SYNC,
        ),
        # Layer-wise variant of RETRIEVE. Same payloads as RETRIEVE, but the
        # server signals a pooled IPC event as each layer batch lands on the
        # device and answers with one frame per batch so the worker can start
        # attention on layer 0 while later layers are still decompressing.
        # A dedicated request type keeps the plain RETRIEVE dispatch path
        # completely untouched.
        # Payload:
        #   - key: KeyType - Cache key (worker_id must not be None)
        #   - instance_id: int - The GPU instance ID (such as PID)
        #   - gpu_block_ids: list[list[int]] - Destination blocks per KV group
        #   - event_ipc_handle: bytes - Producer event handle to order against
        #   - skip_first_n_tokens: int - Tokens to skip at the range start
        # Returns: tuple[bytes, bool, bool] - (payload, is_final, succeeded).
        #   Intermediate frames carry a packed (first_layer, count, pool_index)
        #   batch descriptor with is_final=False; the closing frame carries the
        #   packed pool indices (empty if already reported) with is_final=True.
        #   Completion is decided by the future, not by the transport.
        "RETRIEVE_LAYERWISE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[list[int]], bytes, int],
            response_class=tuple[bytes, bool, bool],
            handler_type=HandlerType.STREAMING,
        ),
    }
