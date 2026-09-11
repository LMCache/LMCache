# SPDX-License-Identifier: Apache-2.0
"""Blend protocol definitions: rope registration, unified lookup, retrieve."""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "CB_REGISTER_ROPE",
    "CB_UNREGISTER_ROPE",
    "CB_RETRIEVE_PRE_COMPUTED",
    "CB_UNIFIED_LOOKUP",
    "CB_PROTOCOL_HANDSHAKE",
]

BLEND_PROTOCOL_VERSION = 1


def handshake_response(client_version: int) -> tuple[int, bool]:
    """Answer a CB_PROTOCOL_HANDSHAKE with (server_version, client_compatible)."""
    return (BLEND_PROTOCOL_VERSION, client_version == BLEND_PROTOCOL_VERSION)


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return the blend protocol definitions."""
    return {
        # Register rope state on a previously-registered instance.
        # Payload: (instance_id, cos_sin_caches_ipc, head_size, is_neox_style,
        #           group_to_cache, group_rot).
        # cos_sin_caches_ipc: one IPC handle per distinct rope (dual-RoPE
        # models send two); group_to_cache maps engine group
        # idx -> cache idx (empty = all groups use cache 0).
        # group_rot: per-engine-group rope window [offset_elems, width_elems]
        # ([] entry = skip that group's re-RoPE; empty list = legacy
        # inference). MLA models must declare it — see cb_register_rope.
        # Returns: None.
        "CB_REGISTER_ROPE": ProtocolDefinition(
            handler_type=HandlerType.SYNC,
        ),
        # Drop rope state (paged KV cache lives on; use UNREGISTER_KV_CACHE).
        # Payload: (instance_id,). Returns: None.
        "CB_UNREGISTER_ROPE": ProtocolDefinition(
            handler_type=HandlerType.SYNC,
        ),
        # Retrieve pre-computed chunks into the request's paged blocks.
        # Payload: (key, cb_match_result, gpu_block_ids, instance_id,
        #           event_ipc_handle).
        # gpu_block_ids is per engine group (list[list[int]]).
        "CB_RETRIEVE_PRE_COMPUTED": ProtocolDefinition(
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
            handler_type=HandlerType.BLOCKING,
        ),
        "CB_PROTOCOL_HANDSHAKE": ProtocolDefinition(
            handler_type=HandlerType.SYNC,
        ),
    }
