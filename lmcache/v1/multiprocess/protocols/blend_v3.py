# SPDX-License-Identifier: Apache-2.0
"""Blend V3 protocol — paged-aware CB pipeline.

RPCs:
* ``CB_REGISTER_ROPE_V3`` / ``CB_UNREGISTER_ROPE_V3`` — share / release the rope
  cos/sin cache onto a context already registered via ``REGISTER_KV_CACHE``.
* ``CB_RETRIEVE_PRE_COMPUTED_V3`` — scatter all matched chunks (prefix- and
  non-prefix-hit) into paged KV by per-token block ID; re-RoPE only the shifted
  (``old_st != cur_st``) subset.
* ``CB_UNIFIED_LOOKUP`` — the sole live lookup path: one RPC runs prefix +
  non-prefix match, reconcile, one sparse-coalesced prefetch, and per-TP-rank
  classify. ``(IPCCacheEngineKey, tp_size)`` → ``CBUnifiedLookupResult``.
"""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    CudaIPCWrapper,
    IPCCacheEngineKey,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "CB_REGISTER_ROPE_V3",
    "CB_UNREGISTER_ROPE_V3",
    "CB_RETRIEVE_PRE_COMPUTED_V3",
    "CB_UNIFIED_LOOKUP",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return V3 blend protocol definitions."""
    return {
        # Register rope state on a previously-registered instance.
        # Payload: (instance_id, cos_sin_cache_ipc, head_size, is_neox_style).
        # Returns: None.
        "CB_REGISTER_ROPE_V3": ProtocolDefinition(
            payload_classes=[int, CudaIPCWrapper, int, bool],
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
                IPCCacheEngineKey,
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
        #   - key: IPCCacheEngineKey carrying the query token IDs.
        #   - tp_size: tensor-parallel size (for MLA multi-reader locking,
        #     mirrors LOOKUP).
        # Returns: CBUnifiedLookupResult(prefix_coverage_tokens,
        #          non_prefix_segments).
        "CB_UNIFIED_LOOKUP": ProtocolDefinition(
            payload_classes=[IPCCacheEngineKey, int],
            # Nullable: handler returns None to defer until both the prefix and
            # the sparse chunks are in L1 (mirrors dense QUERY_PREFETCH_STATUS).
            response_class=CBUnifiedLookupResult | None,
            handler_type=HandlerType.BLOCKING,
        ),
    }
