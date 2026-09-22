# SPDX-License-Identifier: Apache-2.0
"""Tests for descriptor-driven multiprocess RPC discovery."""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.rpc import get_rpc_spec, get_rpc_specs
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    LEGACY_OPERATION_IDS,
    decode_operation,
    encode_operation,
)

FROZEN_LEGACY_OPERATION_IDS = {
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


def test_legacy_zmq_operation_ids_are_frozen() -> None:
    assert dict(LEGACY_OPERATION_IDS) == FROZEN_LEGACY_OPERATION_IDS
    assert len(set(LEGACY_OPERATION_IDS.values())) == len(LEGACY_OPERATION_IDS)


@pytest.mark.parametrize(
    ("operation", "legacy_id"), sorted(FROZEN_LEGACY_OPERATION_IDS.items())
)
def test_zmq_legacy_operation_round_trip(operation: str, legacy_id: int) -> None:
    encoded = encode_operation(operation)
    assert msgspec.msgpack.decode(encoded) == legacy_id
    assert decode_operation(encoded) == operation


def test_new_zmq_operations_use_names_without_registry_changes() -> None:
    operation = "future_extension_rpc"
    encoded = encode_operation(operation)

    assert msgspec.msgpack.decode(encoded) == operation
    assert decode_operation(encoded) == operation


def test_chunk_event_store_uses_its_contract_name_on_zmq() -> None:
    operation = "store_with_chunk_events"
    spec = get_rpc_spec(operation)

    assert spec.response_type == tuple[bytes, list[tuple[bytes, int, int]], bool]
    assert msgspec.msgpack.decode(encode_operation(operation)) == operation


def test_unknown_legacy_zmq_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown legacy ZMQ operation id"):
        decode_operation(msgspec.msgpack.encode(999))


def test_request_contract_matches_generated_grpc_methods() -> None:
    descriptor_operations = {
        client_method_name(method.name) for _binding, method in iter_methods()
    }
    assert set(get_rpc_specs()) == descriptor_operations


def test_rpc_types_come_from_request_client_annotations() -> None:
    lookup = get_rpc_spec("lookup")
    assert len(lookup.payload_types) == 2
    assert lookup.payload_types[1] is int
    assert lookup.response_type is type(None)

    query = get_rpc_spec("query_prefetch_lookup_hits")
    assert query.payload_types == (str,)
    assert query.response_type == int | None

    clear = get_rpc_spec("clear")
    assert clear.payload_types == (bool,)
    assert clear.bind_payloads((), {}) == (False,)
    assert clear.bind_payloads((), {"force": True}) == (True,)
    assert clear.response_type is type(None)
