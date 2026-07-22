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
from typing import Optional
import socket
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
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
    lmcache_mq_pb2,
)


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
        RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
        RequestType.QUERY_PREFETCH_STATUS,
        RequestType.WAIT_PREFETCH_STATUS,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        RequestType.CLEAR,
        RequestType.GET_CHUNK_SIZE,
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
