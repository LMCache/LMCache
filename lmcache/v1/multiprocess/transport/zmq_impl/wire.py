# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for ZMQ operation identifiers."""

# Standard
from types import MappingProxyType
from typing import Any

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.rpc import RpcOperation

# These values were emitted by RequestType before operation-name routing. Keep
# this table frozen so current clients remain wire-compatible with old servers.
# New operations intentionally use their string name and must not be added here.
LEGACY_OPERATION_IDS = MappingProxyType(
    {
        "register_kv_cache": 1,
        "unregister_kv_cache": 2,
        "register_q_cache": 3,
        "unregister_q_cache": 4,
        "store_q": 5,
        "store": 6,
        "retrieve": 7,
        "lookup": 8,
        "query_prefetch_status": 9,
        "wait_prefetch_status": 10,
        "query_prefetch_lookup_hits": 11,
        "free_lookup_locks": 12,
        "end_session": 13,
        "register_kv_cache_engine_driven_context": 14,
        "unregister_kv_cache_engine_driven_context": 15,
        "prepare_store": 16,
        "commit_store": 17,
        "prepare_retrieve": 18,
        "commit_retrieve": 19,
        "clear": 20,
        "get_chunk_size": 21,
        "ping": 22,
        "report_block_allocation": 23,
        "noop": 24,
        "cb_register_rope": 25,
        "cb_unregister_rope": 26,
        "cb_retrieve_pre_computed": 27,
        "cb_unified_lookup": 28,
        "p2p_lookup_and_lock": 29,
        "p2p_query_lookup_results": 30,
        "p2p_unlock_objects": 31,
        "get_experimental": 32,
        "cb_protocol_handshake": 33,
    }
)
_LEGACY_ID_TO_OPERATION = {value: key for key, value in LEGACY_OPERATION_IDS.items()}


def encode_operation(operation: RpcOperation) -> bytes:
    """Encode an operation using its legacy ID when one exists.

    Args:
        operation: Stable snake-case RPC name.

    Returns:
        Msgpack bytes containing a legacy integer or the operation string.
    """
    wire_value: int | str = LEGACY_OPERATION_IDS.get(operation, operation)
    return msgspec.msgpack.encode(wire_value)


def decode_operation(data: bytes) -> RpcOperation:
    """Decode both legacy integer IDs and extensible operation names.

    Args:
        data: Msgpack-encoded operation identifier.

    Returns:
        Stable snake-case RPC name.

    Raises:
        ValueError: If a legacy integer ID is unknown.
        TypeError: If the wire value is neither an integer nor a string.
    """
    wire_value: Any = msgspec.msgpack.decode(data)
    if isinstance(wire_value, int):
        try:
            return _LEGACY_ID_TO_OPERATION[wire_value]
        except KeyError as exc:
            raise ValueError(f"Unknown legacy ZMQ operation id: {wire_value}") from exc
    if isinstance(wire_value, str):
        return wire_value
    raise TypeError(f"Invalid ZMQ operation identifier: {wire_value!r}")
