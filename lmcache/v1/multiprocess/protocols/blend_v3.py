# SPDX-License-Identifier: Apache-2.0
"""Blend V3 protocol definitions."""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "CB_REGISTER_ROPE_V3",
    "CB_UNREGISTER_ROPE_V3",
    "CB_RETRIEVE_PRE_COMPUTED_V3",
    "CB_UNIFIED_LOOKUP",
    # NOTE (Jiayi): qwen35 modification starts
    "AUX_PUT",
    "AUX_GET_BY_HASH_IPC",
    # NOTE (Jiayi): qwen35 modification ends
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return V3 blend protocol definitions."""
    return {
        # Register rope state on a previously-registered instance.
        # Payload: (instance_id, cos_sin_cache_ipc, head_size, is_neox_style).
        # Returns: None.
        "CB_REGISTER_ROPE_V3": ProtocolDefinition(
            payload_classes=[int, DeviceIPCWrapper, int, bool],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Drop rope state (paged KV cache lives on; use UNREGISTER_KV_CACHE).
        # Payload: (instance_id,). Returns: None.
        "CB_UNREGISTER_ROPE_V3": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Retrieve pre-computed chunks into the request's paged blocks.
        # Payload: (key, cb_match_result, gpu_block_ids, instance_id,
        #           event_ipc_handle).
        # Returns: (event_ipc_handle: bytes, success: bool).
        "CB_RETRIEVE_PRE_COMPUTED_V3": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                list[CBMatchResult],
                list[int],
                int,
                bytes,
            ],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Unified lookup: server runs prefix lookup + non-prefix fingerprint
        # match in one RPC, reconciles, and prefetches only the complement.
        # Payload:
        #   - key: IPCCacheServerKey carrying the query token IDs.
        #   - tp_size: tensor-parallel size (for MLA multi-reader locking,
        #     mirrors LOOKUP).
        # Returns: CBUnifiedLookupResult(prefix_coverage_tokens,
        #          non_prefix_segments).
        "CB_UNIFIED_LOOKUP": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, int],
            # Nullable: handler returns None to defer until both the prefix and
            # the sparse chunks are in L1 (mirrors dense QUERY_PREFETCH_STATUS).
            response_class=CBUnifiedLookupResult | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # NOTE (Jiayi): qwen35 modification starts
        # Generic opaque per-chunk blob store. One blob per cacheable chunk of
        # the request range, stored under the chunk's content hash in object
        # group ``group`` (disjoint from KV groups) so it is reusable across
        # requests. The server NEVER interprets the bytes: the caller packs
        # whatever it wants per chunk (e.g. all layers' compressed projections),
        # so one call covers all chunks (hence all layers) at once. ``sizes``
        # gives the per-chunk byte length used to split ``blob`` and to lay out
        # the read. Payload: (key, group, sizes, blob_ipc). Returns: success.
        "AUX_PUT": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                int,
                list[int],
                DeviceIPCWrapper,
            ],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # GPU-IPC retrieve: the worker exports a GPU receive buffer
        # (dst_ipc) and a forward-fence CUDA event; the server copies each
        # matched chunk straight into that buffer on its stream (same physical
        # GPU, no D2H/H2D/ZMQ bytes), then returns its completion-event handle.
        # Payload: (key, group, chunk_hashes, sizes, dst_ipc, instance_id,
        #           event_ipc_handle). Returns: (event_ipc_handle, ok).
        "AUX_GET_BY_HASH_IPC": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                int,
                list[bytes],
                list[int],
                DeviceIPCWrapper,
                int,
                bytes,
            ],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # NOTE (Jiayi): qwen35 modification ends
    }
