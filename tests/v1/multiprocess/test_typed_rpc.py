# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the typed-rpc migration path on the mp-mode mq.

These tests exercise the very first typed rpc (``PING``) and lock in
the invariants that:

1. The typed proto messages (``PingRequest`` / ``PingResponse``) hit
   the wire directly, so the payload bytes are NOT a msgspec envelope.
2. Business handlers registered via ``add_handler`` keep the exact
   same Python signature they had on the legacy path.
3. Legacy rpcs (``NOOP`` here as representative) still work on the
   same server, i.e. the two paths coexist in the same servicer.
"""

# Standard
from typing import Any, Optional
import inspect
import socket
import subprocess
import sys
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import (
    _TYPED_RPCS,
    MessageQueueClient,
    MessageQueueServer,
    request_type_to_method_name,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)

# See mq.py: message classes are dynamic; rebind through Any so
# static analysis stops complaining about attribute lookups.
lmcache_mq_pb2: Any = _pb2_typed


def test_mq_import_does_not_load_grpc_runtime() -> None:
    """Importing MQ helpers must not initialize gRPC native libraries."""
    script = """
import sys
from lmcache.v1.multiprocess import mq

assert "grpc" not in sys.modules
assert "grpc_tools" not in sys.modules
assert mq.grpc is None
assert mq.request_type_to_method_name(mq.RequestType.PING) == "Ping"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ping_is_registered_as_typed_rpc() -> None:
    # The whole point of Phase 1: PING is the reference typed rpc.
    assert RequestType.PING in _TYPED_RPCS
    spec = _TYPED_RPCS[RequestType.PING]
    assert spec.request_message is lmcache_mq_pb2.PingRequest
    assert spec.response_message is lmcache_mq_pb2.PingResponse


def test_ping_method_name_matches_proto() -> None:
    # If the CamelCase mapping ever drifts from the .proto file, gRPC
    # would 404 the method at handshake time -- catch that here.
    assert request_type_to_method_name(RequestType.PING) == "Ping"


def test_ping_typed_roundtrip() -> None:
    """Full client-to-server roundtrip over a real gRPC channel.

    The handler is deliberately introspected to make sure its Python
    signature (``instance_id: Optional[int]) -> bool``) matches
    ``get_payload_classes(PING)`` / ``get_response_class(PING)``, so
    the typed adapter can't paper over a mis-registered handler.
    """
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen_calls: list[Optional[int]] = []

    def ping_handler(instance_id: Optional[int]) -> bool:
        seen_calls.append(instance_id)
        return True

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.PING,
        get_payload_classes(RequestType.PING),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_normal_thread_pool([RequestType.PING], max_workers=2)
    server.start()

    try:
        client = MessageQueueClient(server_url)

        # Case 1: real instance id.  Should reach the handler as int.
        fut: MessagingFuture[bool] = client.submit_request(RequestType.PING, [42])
        assert fut.result(timeout=5.0) is True
        assert seen_calls[-1] == 42

        # Case 2: untracked prober.  The wire sentinel is -1 but the
        # handler should still see Python ``None``.
        fut = client.submit_request(RequestType.PING, [None])
        assert fut.result(timeout=5.0) is True  # type: ignore[union-attr]
        assert seen_calls[-1] is None

        # Handler was hit exactly twice.
        assert len(seen_calls) == 2

        # Typed rpc has no msgspec envelope; make sure the response
        # class the server reports for PING is still ``bool``, so the
        # legacy call sites don't get a surprise.
        assert get_response_class(RequestType.PING) is bool

        client.close()
    finally:
        server.close()


def test_typed_and_legacy_coexist() -> None:
    """Sanity check: a typed rpc and a legacy rpc served by the same
    ``MessageQueueServer`` instance both work independently.

    NOOP is chosen as the legacy witness because it's the smallest
    surviving legacy rpc (no payload, no response).
    """
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    ping_hits = threading.Event()

    def ping_handler(instance_id: Optional[int]) -> bool:
        del instance_id
        ping_hits.set()
        return True

    def noop_handler() -> str:
        return "ok"

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.PING,
        get_payload_classes(RequestType.PING),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RequestType.NOOP,
        get_payload_classes(RequestType.NOOP),
        get_handler_type(RequestType.NOOP),
        noop_handler,
    )
    server.add_normal_thread_pool([RequestType.PING], max_workers=1)
    server.start()

    try:
        client = MessageQueueClient(server_url)

        fut_typed: MessagingFuture[bool] = client.submit_request(RequestType.PING, [7])
        fut_legacy: MessagingFuture[str] = client.submit_request(RequestType.NOOP, [])

        assert fut_typed.result(timeout=5.0) is True
        assert fut_legacy.result(timeout=5.0) == "ok"
        assert ping_hits.is_set()

        client.close()
    finally:
        server.close()


@pytest.mark.parametrize("instance_id", [0, 1, 42, 2**31 - 1, None])
def test_ping_wire_encoding_boundary_values(instance_id: Optional[int]) -> None:
    """Exercise a handful of instance_id values through the typed
    encode / decode pair to catch signed-int overflow or ``None``
    handling regressions before they reach the wire."""
    spec = _TYPED_RPCS[RequestType.PING]
    proto_req = spec.python_to_request(instance_id)
    (round_tripped,) = spec.request_to_python(proto_req)
    assert round_tripped == instance_id

    proto_resp = spec.python_to_response(True)
    assert spec.response_to_python(proto_resp) is True

    # Sanity: the wire really is a PingRequest, not BytesRequest.
    assert isinstance(proto_req, lmcache_mq_pb2.PingRequest)


# ---------------------------------------------------------------------
# LOOKUP: second typed rpc, first consumer of IpcCacheServerKey.
# ---------------------------------------------------------------------


def _sample_key(
    *,
    worker_id: Optional[int] = 3,
    token_ids: tuple[int, ...] = (10, 20, 30, 40),
    cache_salt: str = "",
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="mymodel",
        world_size=4,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id="req-42",
        cache_salt=cache_salt,
    )


def test_lookup_is_registered_as_typed_rpc() -> None:
    assert RequestType.LOOKUP in _TYPED_RPCS
    spec = _TYPED_RPCS[RequestType.LOOKUP]
    assert spec.request_message is lmcache_mq_pb2.LookupRequest
    assert spec.response_message is lmcache_mq_pb2.LookupResponse


@pytest.mark.parametrize(
    "worker_id,token_ids,cache_salt",
    [
        (None, (1,), ""),
        (0, (), ""),
        (7, (1, 2, 3), "tenant-a"),
        (999, tuple(range(2048)), "long-token-list"),
    ],
)
def test_lookup_key_roundtrip(
    worker_id: Optional[int],
    token_ids: tuple[int, ...],
    cache_salt: str,
) -> None:
    """The shared IpcCacheServerKey proto <-> dataclass helpers must
    preserve every field, including the None/optional worker_id and
    empty/large token_ids edge cases."""
    spec = _TYPED_RPCS[RequestType.LOOKUP]
    key = _sample_key(worker_id=worker_id, token_ids=token_ids, cache_salt=cache_salt)
    proto_req = spec.python_to_request(key, 4)
    assert isinstance(proto_req, lmcache_mq_pb2.LookupRequest)
    round_tripped_key, tp_size = spec.request_to_python(proto_req)
    assert round_tripped_key == key
    assert tp_size == 4


def test_lookup_typed_roundtrip() -> None:
    """Full gRPC roundtrip: server handler sees the exact key the
    client sent, and the (empty) response arrives as Python ``None``."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen: list[tuple[IPCCacheServerKey, int]] = []

    def lookup_handler(key: IPCCacheServerKey, tp_size: int) -> None:
        seen.append((key, tp_size))
        return None

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.LOOKUP,
        get_payload_classes(RequestType.LOOKUP),
        HandlerType.BLOCKING,
        lookup_handler,
    )
    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=2)
    server.start()

    try:
        client = MessageQueueClient(server_url)
        key = _sample_key(cache_salt="tenant-b")

        fut: MessagingFuture[None] = client.submit_request(RequestType.LOOKUP, [key, 8])
        assert fut.result(timeout=5.0) is None
        assert seen == [(key, 8)]

        client.close()
    finally:
        server.close()


# ---------------------------------------------------------------------
# Wave 2: bulk-migrated trivial rpcs.  One parametrized test covers all
# ten in the same shape so adding the next wave stays cheap.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.FREE_LOOKUP_LOCKS,
        RequestType.END_SESSION,
        RequestType.UNREGISTER_KV_CACHE,
        RequestType.UNREGISTER_Q_CACHE,
        RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
        RequestType.QUERY_PREFETCH_STATUS,
        RequestType.WAIT_PREFETCH_STATUS,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        RequestType.CLEAR,
        RequestType.GET_CHUNK_SIZE,
        RequestType.GET_EXPERIMENTAL,
        RequestType.NOOP,
    ],
)
def test_wave2_rpc_is_typed(request_type: RequestType) -> None:
    """Each Wave 2 rpc must be off the msgspec envelope."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message is not lmcache_mq_pb2.BytesRequest
    assert spec.response_message is not lmcache_mq_pb2.BytesResponse


def test_optional_chunk_count_none_roundtrip() -> None:
    """``int | None`` -> ``optional int64`` must preserve None."""
    for rt in (
        RequestType.QUERY_PREFETCH_STATUS,
        RequestType.WAIT_PREFETCH_STATUS,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
    ):
        spec = _TYPED_RPCS[rt]
        for value in (None, 0, 1, 42, 2**31):
            proto_resp = spec.python_to_response(value)
            assert spec.response_to_python(proto_resp) == value


def test_free_lookup_locks_typed_roundtrip() -> None:
    """Full gRPC roundtrip on the FreeLookupLocks rpc — proves that
    IpcCacheServerKey embedded in a second message pair also works."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen: list[tuple[IPCCacheServerKey, int]] = []

    def free_locks_handler(key: IPCCacheServerKey, tp_size: int) -> None:
        seen.append((key, tp_size))
        return None

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.FREE_LOOKUP_LOCKS,
        get_payload_classes(RequestType.FREE_LOOKUP_LOCKS),
        HandlerType.BLOCKING,
        free_locks_handler,
    )
    server.add_normal_thread_pool([RequestType.FREE_LOOKUP_LOCKS], max_workers=1)
    server.start()
    try:
        client = MessageQueueClient(server_url)
        key = _sample_key(cache_salt="wave2")
        fut: MessagingFuture[None] = client.submit_request(
            RequestType.FREE_LOOKUP_LOCKS, [key, 2]
        )
        assert fut.result(timeout=5.0) is None
        assert seen == [(key, 2)]
        client.close()
    finally:
        server.close()


def test_get_chunk_size_typed_roundtrip() -> None:
    """Empty-payload rpc with a non-empty response body."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    def chunk_size_handler() -> int:
        return 256

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.GET_CHUNK_SIZE,
        get_payload_classes(RequestType.GET_CHUNK_SIZE),
        HandlerType.SYNC,
        chunk_size_handler,
    )
    server.start()
    try:
        client = MessageQueueClient(server_url)
        fut: MessagingFuture[int] = client.submit_request(
            RequestType.GET_CHUNK_SIZE, []
        )
        assert fut.result(timeout=5.0) == 256
        client.close()
    finally:
        server.close()


def test_query_prefetch_status_typed_roundtrip() -> None:
    """optional int64 chunk_count comes back as int and as None."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    results: dict[str, Optional[int]] = {"req-1": 7, "req-2": None}

    def status_handler(request_id: str) -> Optional[int]:
        return results[request_id]

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.QUERY_PREFETCH_STATUS,
        get_payload_classes(RequestType.QUERY_PREFETCH_STATUS),
        HandlerType.BLOCKING,
        status_handler,
    )
    server.add_normal_thread_pool([RequestType.QUERY_PREFETCH_STATUS], max_workers=2)
    server.start()
    try:
        client = MessageQueueClient(server_url)
        fut1: MessagingFuture[Optional[int]] = client.submit_request(
            RequestType.QUERY_PREFETCH_STATUS, ["req-1"]
        )
        fut2: MessagingFuture[Optional[int]] = client.submit_request(
            RequestType.QUERY_PREFETCH_STATUS, ["req-2"]
        )
        assert fut1.result(timeout=5.0) == 7
        assert fut2.result(timeout=5.0) is None
        client.close()
    finally:
        server.close()


# ---------------------------------------------------------------------
# Wave 3: Store/Retrieve, CB v1/v2/v3 lookup+store+retrieve family,
# ReportBlockAllocation.  Adapter roundtrip tests only — full e2e
# happens in the vllm-driven suite.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.STORE_Q,
        RequestType.STORE,
        RequestType.RETRIEVE,
        RequestType.REPORT_BLOCK_ALLOCATION,
        RequestType.CB_UNREGISTER_KV_CACHE,
        RequestType.CB_UNREGISTER_ROPE_V3,
        RequestType.CB_LOOKUP_PRE_COMPUTED,
        RequestType.CB_STORE_PRE_COMPUTED,
        RequestType.CB_STORE_FINAL,
        RequestType.CB_RETRIEVE_PRE_COMPUTED,
        RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
        RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
        RequestType.CB_RETRIEVE_PRE_COMPUTED_V3,
        RequestType.CB_UNIFIED_LOOKUP,
    ],
)
def test_wave3_rpc_is_typed(request_type: RequestType) -> None:
    """Wave 3 rpcs must all be off the msgspec envelope."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message is not lmcache_mq_pb2.BytesRequest
    assert spec.response_message is not lmcache_mq_pb2.BytesResponse


def test_store_retrieve_roundtrip() -> None:
    """Store/Retrieve payload preservation across the wire boundary."""
    key = _sample_key()
    store_spec = _TYPED_RPCS[RequestType.STORE]
    proto_req = store_spec.python_to_request(key, 12, [[1, 2, 3], [4, 5]], b"event")
    assert isinstance(proto_req, lmcache_mq_pb2.StoreRequest)
    round_key, iid, blocks, event = store_spec.request_to_python(proto_req)
    assert round_key == key
    assert iid == 12
    assert blocks == [[1, 2, 3], [4, 5]]
    assert event == b"event"

    proto_resp = store_spec.python_to_response((b"eh", True))
    assert store_spec.response_to_python(proto_resp) == (b"eh", True)

    retrieve_spec = _TYPED_RPCS[RequestType.RETRIEVE]
    proto_req = retrieve_spec.python_to_request(key, 12, [[7], [8, 9]], b"event2", 128)
    round_key, iid, blocks, event, skip = retrieve_spec.request_to_python(proto_req)
    assert round_key == key
    assert iid == 12
    assert blocks == [[7], [8, 9]]
    assert event == b"event2"
    assert skip == 128


def test_report_block_allocation_roundtrip() -> None:
    """BlockAllocationRecord list survives the wire in order."""
    spec = _TYPED_RPCS[RequestType.REPORT_BLOCK_ALLOCATION]
    records = [
        BlockAllocationRecord(
            req_id="r1", new_block_ids=[1, 2], new_token_ids=[10, 20]
        ),
        BlockAllocationRecord(req_id="r2", new_block_ids=[3], new_token_ids=[30]),
    ]
    proto_req = spec.python_to_request(7, "opt-125m", records)
    iid, model, out = spec.request_to_python(proto_req)
    assert iid == 7
    assert model == "opt-125m"
    assert out == records


def test_cb_lookup_v1_and_v2_roundtrip() -> None:
    """(start,end) tuples and CBMatchResult both survive the wire."""
    key = _sample_key()

    v1 = _TYPED_RPCS[RequestType.CB_LOOKUP_PRE_COMPUTED]
    ranges = [(0, 8), (16, 24)]
    proto_resp = v1.python_to_response(ranges)
    assert v1.response_to_python(proto_resp) == ranges

    v2 = _TYPED_RPCS[RequestType.CB_LOOKUP_PRE_COMPUTED_V2]
    matches = [
        CBMatchResult(old_st=0, old_ed=8, cur_st=0, cur_ed=8, hash=b"h1"),
        CBMatchResult(old_st=8, old_ed=16, cur_st=16, cur_ed=24, hash=b"h2"),
    ]
    proto_req = v2.python_to_request(key)
    (round_key,) = v2.request_to_python(proto_req)
    assert round_key == key
    proto_resp = v2.python_to_response(matches)
    assert v2.response_to_python(proto_resp) == matches


def test_cb_unified_lookup_nullable() -> None:
    """None (still loading) and populated result must both roundtrip."""
    spec = _TYPED_RPCS[RequestType.CB_UNIFIED_LOOKUP]

    # Absent payload -> Python None.
    proto_resp = spec.python_to_response(None)
    assert spec.response_to_python(proto_resp) is None

    # Populated payload.
    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=64,
        non_prefix_segments=[
            CBMatchResult(old_st=0, old_ed=8, cur_st=8, cur_ed=16, hash=b"a"),
        ],
        segmented_prefix_segments=[
            CBMatchResult(old_st=32, old_ed=40, cur_st=32, cur_ed=40, hash=b"b"),
        ],
    )
    proto_resp = spec.python_to_response(result)
    round = spec.response_to_python(proto_resp)
    assert round == result


def test_store_typed_grpc_roundtrip() -> None:
    """Full gRPC roundtrip through a live channel for the STORE rpc."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen: list[tuple[IPCCacheServerKey, int, list[list[int]], bytes]] = []

    def store_handler(
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        seen.append((key, instance_id, gpu_block_ids, event_ipc_handle))
        return (b"handle-back", True)

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.STORE,
        get_payload_classes(RequestType.STORE),
        HandlerType.BLOCKING,
        store_handler,
    )
    server.add_normal_thread_pool([RequestType.STORE], max_workers=2)
    server.start()
    try:
        client = MessageQueueClient(server_url)
        key = _sample_key(cache_salt="wave3")
        fut: MessagingFuture[tuple[bytes, bool]] = client.submit_request(
            RequestType.STORE, [key, 42, [[1, 2], [3]], b"ev"]
        )
        assert fut.result(timeout=5.0) == (b"handle-back", True)
        assert seen == [(key, 42, [[1, 2], [3]], b"ev")]
        client.close()
    finally:
        server.close()


# ---------------------------------------------------------------------
# Wave 4: Register*/Commit*/Prepare* + P2P family.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
        RequestType.PREPARE_STORE,
        RequestType.COMMIT_STORE,
        RequestType.PREPARE_RETRIEVE,
        RequestType.COMMIT_RETRIEVE,
        RequestType.P2P_LOOKUP_AND_LOCK,
        RequestType.P2P_QUERY_LOOKUP_RESULTS,
        RequestType.P2P_UNLOCK_OBJECTS,
    ],
)
def test_wave4_rpc_is_typed(request_type: RequestType) -> None:
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message is not lmcache_mq_pb2.BytesRequest
    assert spec.response_message is not lmcache_mq_pb2.BytesResponse


def test_register_edc_roundtrip() -> None:
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    spec = _TYPED_RPCS[RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT]
    payload = RegisterEngineDrivenContextPayload(
        instance_id=7,
        model_name="opt-125m",
        world_size=4,
        block_size=16,
        num_layers=12,
        hidden_dim_size=768,
        dtype_str="float16",
        use_mla=False,
    )
    proto_req = spec.python_to_request(payload)
    (round_payload,) = spec.request_to_python(proto_req)
    assert round_payload == payload

    resp = RegisterEngineDrivenContextResponse(shm_name="/shm", pool_size=42)
    proto_resp = spec.python_to_response(resp)
    assert spec.response_to_python(proto_resp) == resp


def test_prepare_store_pickle_context_roundtrip() -> None:
    # First Party
    from lmcache.v1.multiprocess.protocols.engine import PrepareStoreResponse

    spec = _TYPED_RPCS[RequestType.PREPARE_STORE]
    key = _sample_key()
    proto_req = spec.python_to_request(key, 3)
    round_key, iid = spec.request_to_python(proto_req)
    assert round_key == key
    assert iid == 3

    # Empty dict must land as empty bytes on the wire.
    proto_resp = spec.python_to_response(PrepareStoreResponse(context={}))
    assert proto_resp.pickled_context == b""
    assert spec.response_to_python(proto_resp).context == {}

    # Non-empty dict survives pickle.
    ctx = {"slots": [1, 2, 3], "chunk_indices": [0, 5]}
    proto_resp = spec.python_to_response(PrepareStoreResponse(context=ctx))
    assert spec.response_to_python(proto_resp).context == ctx


def test_prepare_retrieve_pickle_context_roundtrip() -> None:
    # First Party
    from lmcache.v1.multiprocess.protocols.engine import (
        PrepareRetrieveResponse,
    )

    spec = _TYPED_RPCS[RequestType.PREPARE_RETRIEVE]
    resp = PrepareRetrieveResponse(success=True, data=b"hello", context={"slots": [7]})
    proto_resp = spec.python_to_response(resp)
    round = spec.response_to_python(proto_resp)
    assert round.success is True
    assert round.data == b"hello"
    assert round.context == {"slots": [7]}


def test_commit_store_and_retrieve_bool_roundtrip() -> None:
    key = _sample_key()

    cs = _TYPED_RPCS[RequestType.COMMIT_STORE]
    proto_req = cs.python_to_request(key, 4, b"ctx-bytes")
    r_key, r_iid, r_ctx = cs.request_to_python(proto_req)
    assert (r_key, r_iid, r_ctx) == (key, 4, b"ctx-bytes")
    assert cs.response_to_python(cs.python_to_response(True)) is True
    assert cs.response_to_python(cs.python_to_response(False)) is False

    cr = _TYPED_RPCS[RequestType.COMMIT_RETRIEVE]
    proto_req = cr.python_to_request(key, 5)
    r_key, r_iid = cr.request_to_python(proto_req)
    assert (r_key, r_iid) == (key, 5)
    assert cr.response_to_python(cr.python_to_response(True)) is True


def test_p2p_lookup_and_query_roundtrip() -> None:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.transfer_channel.api import (
        TransferChannelAddress,
    )

    lookup = _TYPED_RPCS[RequestType.P2P_LOOKUP_AND_LOCK]
    keys = [
        ObjectKey(
            chunk_hash=b"\x01\x02",
            model_name="opt",
            kv_rank=7,
            object_group_id=1,
            cache_salt="tenant",
        ),
        ObjectKey(chunk_hash=b"\x03\x04", model_name="opt", kv_rank=7),
    ]
    group_layouts = {
        0: MemoryLayoutDesc(
            shapes=[torch.Size([2, 8, 128]), torch.Size([2, 8, 128])],
            dtypes=[torch.float16, torch.bfloat16],
        ),
        7: MemoryLayoutDesc(
            shapes=[torch.Size([4, 64])],
            dtypes=[torch.uint8],
        ),
    }
    proto_req = lookup.python_to_request(keys, group_layouts)
    round_keys, round_group_layouts = lookup.request_to_python(proto_req)
    assert round_keys == keys
    assert round_group_layouts == group_layouts

    proto_resp = lookup.python_to_response(99)
    assert lookup.response_to_python(proto_resp) == 99

    # Query: None <-> present addr list.
    query = _TYPED_RPCS[RequestType.P2P_QUERY_LOOKUP_RESULTS]
    assert query.response_to_python(query.python_to_response(None)) is None

    addrs = [
        TransferChannelAddress(offset=0, size=1024),
        TransferChannelAddress(offset=1024, size=512),
    ]
    proto_resp = query.python_to_response(addrs)
    assert query.response_to_python(proto_resp) == addrs

    # Empty list is a real (non-None) empty match.
    proto_resp = query.python_to_response([])
    assert query.response_to_python(proto_resp) == []

    # Unlock: keys survive round-trip, response is None.
    unlock = _TYPED_RPCS[RequestType.P2P_UNLOCK_OBJECTS]
    proto_req = unlock.python_to_request(keys)
    (round_keys,) = unlock.request_to_python(proto_req)
    assert round_keys == keys
    assert unlock.response_to_python(unlock.python_to_response(None)) is None


# ---------------------------------------------------------------------
# Wave 5: Register* rpcs.  DeviceIPCWrapper subclasses are pickle-
# preserved on the wire, so a bare test double subclass suffices to
# prove the identity round-trip without touching the CUDA / SHM paths.
# ---------------------------------------------------------------------


class _FakeIPCWrapper(DeviceIPCWrapper):
    """Minimal wrapper subclass used only by these tests -- exercises
    the pickle identity guarantee end-to-end without needing a real
    device."""

    def __init__(self, tag: str = "fake") -> None:
        self.handle = ("fake-handle", tag)
        self.dtype = torch.float16
        self.shape = (2, 4)
        self.stride = (4, 1)
        self.storage_offset = 0
        self.device_uuid = "test-uuid-" + tag


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.REGISTER_KV_CACHE,
        RequestType.REGISTER_Q_CACHE,
        RequestType.CB_REGISTER_KV_CACHE,
        RequestType.CB_REGISTER_ROPE_V3,
    ],
)
def test_wave5_rpc_is_typed(request_type: RequestType) -> None:
    """Wave 5 finishes the migration: all request types use typed RPCs."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message is not lmcache_mq_pb2.BytesRequest
    assert spec.response_message is not lmcache_mq_pb2.BytesResponse


def test_typed_rpc_coverage_is_complete() -> None:
    """Enforce the migration invariant: every RequestType is typed."""
    missing = [rt.name for rt in RequestType if rt not in _TYPED_RPCS]
    assert not missing, f"legacy rpcs remain: {missing}"


@pytest.mark.parametrize("request_type", list(RequestType))
def test_typed_rpc_request_arity_matches_protocol(request_type: RequestType) -> None:
    """Keep typed adapters aligned with the public protocol payload contract."""
    signature = inspect.signature(_TYPED_RPCS[request_type].python_to_request)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
    ]
    assert len(positional) == len(get_payload_classes(request_type))


def test_register_kv_cache_roundtrip() -> None:
    """DeviceIPCWrapper subclass identity + LayoutHints + EngineGroupInfo
    all survive the wire."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    spec = _TYPED_RPCS[RequestType.REGISTER_KV_CACHE]
    kv_cache = [_FakeIPCWrapper("a"), _FakeIPCWrapper("b")]
    hints: dict = {"kv_layout": "NHD", "tokens_per_block": 16}
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1, 2),
            tokens_per_block=16,
            sw_size_tokens=-1,
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(3, 4),
            tokens_per_block=16,
            sw_size_tokens=128,
        ),
    ]
    proto_req = spec.python_to_request(
        7,
        kv_cache,
        "opt-125m",
        4,
        EngineType.VLLM,
        hints,
        groups,
    )
    assert isinstance(proto_req, lmcache_mq_pb2.RegisterKvCacheRequest)
    (
        iid,
        r_kv,
        model,
        world,
        etype,
        r_hints,
        r_groups,
    ) = spec.request_to_python(proto_req)
    assert iid == 7
    assert model == "opt-125m"
    assert world == 4
    assert etype is EngineType.VLLM
    assert r_hints == hints
    assert r_groups == groups
    # Subclass identity + tags survive.
    assert len(r_kv) == 2
    assert all(isinstance(w, _FakeIPCWrapper) for w in r_kv)
    assert r_kv[0].device_uuid == "test-uuid-a"
    assert r_kv[1].device_uuid == "test-uuid-b"

    # Empty LayoutHints -> empty wire bytes.
    proto_req = spec.python_to_request(1, [], "m", 1, EngineType.MOCK, {}, [])
    assert proto_req.pickled_layout_hints == b""


def test_cb_register_kv_cache_roundtrip() -> None:
    spec = _TYPED_RPCS[RequestType.CB_REGISTER_KV_CACHE]
    kv_cache = [_FakeIPCWrapper("cb")]
    proto_req = spec.python_to_request(9, kv_cache, "llama", 2)
    iid, r_kv, model, world = spec.request_to_python(proto_req)
    assert (iid, model, world) == (9, "llama", 2)
    assert isinstance(r_kv[0], _FakeIPCWrapper)
    assert r_kv[0].device_uuid == "test-uuid-cb"


def test_cb_register_rope_v3_roundtrip() -> None:
    spec = _TYPED_RPCS[RequestType.CB_REGISTER_ROPE_V3]
    caches = [_FakeIPCWrapper("rope-0"), _FakeIPCWrapper("rope-1")]
    group_rot = [[0, 64], [], [128, 32]]
    proto_req = spec.python_to_request(3, caches, 128, True, [0, 1, 0], group_rot)
    iid, r_caches, head, neox, mapping, r_group_rot = spec.request_to_python(proto_req)
    assert iid == 3
    assert head == 128
    assert neox is True
    assert mapping == [0, 1, 0]
    assert r_group_rot == group_rot
    assert [w.device_uuid for w in r_caches] == [
        "test-uuid-rope-0",
        "test-uuid-rope-1",
    ]


def test_register_kv_cache_grpc_roundtrip() -> None:
    """End-to-end gRPC hop for the biggest Wave 5 rpc, proving the
    pickle-in-proto pattern holds over the real transport."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen: list[tuple] = []

    def handler(
        instance_id: int,
        kv_cache: list[DeviceIPCWrapper],
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        seen.append(
            (
                instance_id,
                len(kv_cache),
                model_name,
                world_size,
                engine_type,
                layout_hints,
                engine_group_infos,
            )
        )
        return None

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.REGISTER_KV_CACHE,
        get_payload_classes(RequestType.REGISTER_KV_CACHE),
        HandlerType.SYNC,
        handler,
    )
    server.start()
    try:
        client = MessageQueueClient(server_url)
        kv = [_FakeIPCWrapper("e2e")]
        hints = {"kv_layout": "HND"}
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=32,
                sw_size_tokens=-1,
            ),
        ]
        fut: MessagingFuture = client.submit_request(
            RequestType.REGISTER_KV_CACHE,
            [5, kv, "opt-125m", 2, EngineType.VLLM, hints, groups],
        )
        assert fut.result(timeout=5.0) is None
        assert len(seen) == 1
        (
            iid,
            n_kv,
            model,
            world,
            etype,
            r_hints,
            r_groups,
        ) = seen[0]
        assert (iid, n_kv, model, world) == (5, 1, "opt-125m", 2)
        assert etype is EngineType.VLLM
        assert r_hints == hints
        assert r_groups == groups
        client.close()
    finally:
        server.close()


def test_grpc_request_waits_for_server_startup() -> None:
    """Requests submitted during daemon startup must not fail fast."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    client = MessageQueueClient(server_url)
    future: MessagingFuture[bool] = client.submit_request(RequestType.PING, [None])
    time.sleep(0.25)
    assert not future.query()

    server = MessageQueueServer(server_url)

    def handler(instance_id: Optional[int]) -> bool:
        return instance_id is None

    server.add_handler(
        RequestType.PING,
        get_payload_classes(RequestType.PING),
        HandlerType.SYNC,
        handler,
    )
    server.start()
    try:
        assert future.result(timeout=5.0) is True
    finally:
        client.close()
        server.close()


def test_grpc_request_stays_pending_when_server_stops_mid_call() -> None:
    """An in-flight request must retain legacy fault-tolerance semantics."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    handler_entered = threading.Event()
    release_handler = threading.Event()

    def handler(instance_id: Optional[int]) -> bool:
        del instance_id
        handler_entered.set()
        release_handler.wait(timeout=5.0)
        return True

    server = MessageQueueServer(server_url)
    server.add_handler(
        RequestType.PING,
        get_payload_classes(RequestType.PING),
        HandlerType.SYNC,
        handler,
    )
    server.start()
    client = MessageQueueClient(server_url)
    future: MessagingFuture[bool] = client.submit_request(RequestType.PING, [None])
    assert handler_entered.wait(timeout=5.0)

    server.close()
    release_handler.set()
    time.sleep(0.25)
    try:
        assert not future.query()
    finally:
        client.close()
