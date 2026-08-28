# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for typed gRPC on the mp-mode message queue.

These tests lock in the invariants that:

1. The typed proto messages (``PingRequest`` / ``PingResponse``) hit
   the wire directly, so the payload bytes are NOT a msgspec envelope.
2. Business handlers registered via ``add_handler`` keep their public
   Python signatures.
3. Every ``RpcMethod`` maps to one descriptor-derived typed RPC.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
import socket
import subprocess
import sys
import tempfile
import threading
import time

# Third Party
import grpc
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
    MultiprocessGrpcClient,
    MultiprocessGrpcServer,
    request_type_to_method_name,
)
from lmcache.v1.multiprocess.protocol import (
    RPC,
    HandlerType,
    RpcMethod,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2_grpc,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    TYPED_RPCS as _TYPED_RPCS,
)

# Generated message classes are dynamic; rebind through Any so
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
assert mq.request_type_to_method_name("PING") == "Ping"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ping_is_registered_as_typed_rpc() -> None:
    assert RPC.Ping in _TYPED_RPCS
    spec = _TYPED_RPCS[RPC.Ping]
    assert spec.request_message is lmcache_mq_pb2.PingRequest
    assert spec.response_message is lmcache_mq_pb2.PingResponse


def test_ping_method_name_matches_proto() -> None:
    # If the CamelCase mapping ever drifts from the .proto file, gRPC
    # would 404 the method at handshake time -- catch that here.
    assert request_type_to_method_name(RPC.Ping) == "Ping"


def test_service_descriptor_covers_every_typed_rpc() -> None:
    """The registry is derived from, and stays aligned with, gRPC services."""
    services = lmcache_mq_pb2.DESCRIPTOR.services_by_name
    service_methods = {
        method.name: (service.name, method)
        for service in services.values()
        for method in service.methods
    }
    expected_methods = {
        request_type_to_method_name(request_type) for request_type in RpcMethod
    }
    assert set(service_methods) == expected_methods

    for request_type, spec in _TYPED_RPCS.items():
        service_name, method = service_methods[
            request_type_to_method_name(request_type)
        ]
        assert service_name == request_type.service_name
        assert spec.service_name == service_name
        assert method.input_type is spec.request_message.DESCRIPTOR
        assert method.output_type is spec.response_message.DESCRIPTOR
        assert spec.payload_types == tuple(get_payload_classes(request_type))
        assert spec.response_type == get_response_class(request_type)


def test_client_installs_function_style_rpc_methods() -> None:
    """Every protocol method is directly callable on the client."""
    for request_type in RpcMethod:
        method = getattr(MultiprocessGrpcClient, request_type.name.lower(), None)
        assert callable(method), request_type.name


def test_service_descriptor_has_no_legacy_batch_rpc() -> None:
    """The transport contract is unary-only; Batch is no longer part of it."""
    services = lmcache_mq_pb2.DESCRIPTOR.services_by_name
    assert "MessageQueue" not in services
    assert all("Batch" not in service.methods_by_name for service in services.values())


def test_concurrent_mixed_requests_roundtrip_over_unary_rpcs() -> None:
    """Concurrent typed methods retain their individual results."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    first_ping_entered = threading.Event()
    release_first_ping = threading.Event()

    def ping_handler(instance_id: Optional[int]) -> bool:
        if instance_id == 0:
            first_ping_entered.set()
            release_first_ping.wait(timeout=5.0)
        return True

    def noop_handler() -> str:
        return "ok"

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RPC.Noop,
        get_payload_classes(RPC.Noop),
        HandlerType.SYNC,
        noop_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        first: MessagingFuture[bool] = client.submit_request(RPC.Ping, [0])
        assert first_ping_entered.wait(timeout=5.0)
        futures: list[MessagingFuture[Any]] = [
            client.submit_request(RPC.Ping, [1]),
            client.submit_request(RPC.Noop, []),
            client.submit_request(RPC.Ping, [2]),
            client.submit_request(RPC.Noop, []),
        ]
        release_first_ping.set()

        assert first.result(timeout=5.0) is True
        assert [future.result(timeout=5.0) for future in futures] == [
            True,
            "ok",
            True,
            "ok",
        ]
    finally:
        release_first_ping.set()
        client.close()
        server.close()


def test_unix_clients_keep_distinct_grpc_affinity() -> None:
    """Stable client metadata preserves affinity over Unix sockets."""
    with tempfile.TemporaryDirectory(prefix="lmcache-mq-test-") as directory:
        server_url = f"grpc+unix://{directory}/mq.sock"
        threads_by_client: dict[int, set[str]] = {1: set(), 2: set()}
        seen_lock = threading.Lock()

        def ping_handler(instance_id: Optional[int]) -> bool:
            assert instance_id is not None
            client_id = instance_id // 100
            with seen_lock:
                threads_by_client[client_id].add(threading.current_thread().name)
            time.sleep(0.005)
            return True

        server = MultiprocessGrpcServer(server_url)
        server.add_handler(
            RPC.Ping,
            get_payload_classes(RPC.Ping),
            HandlerType.BLOCKING,
            ping_handler,
        )
        server.add_affinity_thread_pool([RPC.Ping], max_workers=2)
        server.start()
        clients = [
            MultiprocessGrpcClient(server_url),
            MultiprocessGrpcClient(server_url),
        ]

        try:
            futures: list[MessagingFuture[bool]] = []
            for index in range(8):
                futures.append(clients[0].submit_request(RPC.Ping, [100 + index]))
                futures.append(clients[1].submit_request(RPC.Ping, [200 + index]))
            assert all(future.result(timeout=5.0) is True for future in futures)
            assert all(len(names) == 1 for names in threads_by_client.values())
            assert threads_by_client[1] != threads_by_client[2]
        finally:
            for client in clients:
                client.close()
            server.close()


def test_unary_item_error_surfaces_as_grpc_failure() -> None:
    """A handler failure surfaces as the matching unary gRPC failure only."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    first_ping_entered = threading.Event()
    release_first_ping = threading.Event()

    def ping_handler(instance_id: Optional[int]) -> bool:
        if instance_id == 0:
            first_ping_entered.set()
            release_first_ping.wait(timeout=5.0)
        return True

    def failing_noop_handler() -> str:
        raise ValueError("expected unary failure")

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RPC.Noop,
        get_payload_classes(RPC.Noop),
        HandlerType.SYNC,
        failing_noop_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        first: MessagingFuture[bool] = client.submit_request(RPC.Ping, [0])
        assert first_ping_entered.wait(timeout=5.0)
        failed: MessagingFuture[str] = client.submit_request(RPC.Noop, [])
        succeeded: MessagingFuture[bool] = client.submit_request(RPC.Ping, [1])
        release_first_ping.set()

        assert first.result(timeout=5.0) is True
        with pytest.raises(grpc.RpcError, match="expected unary failure") as exc_info:
            failed.result(timeout=5.0)
        assert exc_info.value.code() is grpc.StatusCode.UNKNOWN
        assert succeeded.result(timeout=5.0) is True
    finally:
        release_first_ping.set()
        client.close()
        server.close()


def test_client_works_with_minimal_unary_servicer() -> None:
    """The client only depends on the typed unary method it invokes."""
    port = _find_free_port()
    target = f"127.0.0.1:{port}"
    server_url = f"grpc://{target}"
    first_ping_entered = threading.Event()
    release_first_ping = threading.Event()

    class LegacyServicer(lmcache_mq_pb2_grpc.ControllerServiceServicer):
        def Ping(self, request: Any, context: Any) -> Any:
            del context
            if request.instance_id == 0:
                first_ping_entered.set()
                release_first_ping.wait(timeout=5.0)
            return lmcache_mq_pb2.PingResponse(ok=True)

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    lmcache_mq_pb2_grpc.add_ControllerServiceServicer_to_server(
        LegacyServicer(), server
    )
    server.add_insecure_port(target)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        first: MessagingFuture[bool] = client.submit_request(RPC.Ping, [0])
        assert first_ping_entered.wait(timeout=5.0)
        second: MessagingFuture[bool] = client.submit_request(RPC.Ping, [1])
        release_first_ping.set()

        assert first.result(timeout=5.0) is True
        assert second.result(timeout=5.0) is True
    finally:
        release_first_ping.set()
        client.close()
        server.stop(grace=None).wait()


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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()

    try:
        client = MultiprocessGrpcClient(server_url)

        # Case 1: real instance id.  Should reach the handler as int.
        fut: MessagingFuture[bool] = client.submit_request(RPC.Ping, [42])
        assert fut.result(timeout=5.0) is True
        assert seen_calls[-1] == 42

        # Case 2: untracked prober.  The wire sentinel is -1 but the
        # handler should still see Python ``None``.
        fut = client.submit_request(RPC.Ping, [None])
        assert fut.result(timeout=5.0) is True  # type: ignore[union-attr]
        assert seen_calls[-1] is None

        # Handler was hit exactly twice.
        assert len(seen_calls) == 2

        # Typed rpc has no msgspec envelope; make sure the response
        # class the server reports for PING is still ``bool``, so the
        # legacy call sites don't get a surprise.
        assert get_response_class(RPC.Ping) is bool

        client.close()
    finally:
        server.close()


def test_distinct_typed_rpcs_coexist() -> None:
    """Distinct typed methods served by one server do not interfere."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    ping_hits = threading.Event()

    def ping_handler(instance_id: Optional[int]) -> bool:
        del instance_id
        ping_hits.set()
        return True

    def noop_handler() -> str:
        return "ok"

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RPC.Noop,
        get_payload_classes(RPC.Noop),
        get_handler_type(RPC.Noop),
        noop_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=1)
    server.start()

    try:
        client = MultiprocessGrpcClient(server_url)

        fut_typed: MessagingFuture[bool] = client.submit_request(RPC.Ping, [7])
        fut_noop: MessagingFuture[str] = client.submit_request(RPC.Noop, [])

        assert fut_typed.result(timeout=5.0) is True
        assert fut_noop.result(timeout=5.0) == "ok"
        assert ping_hits.is_set()

        client.close()
    finally:
        server.close()


@pytest.mark.parametrize("instance_id", [0, 1, 42, 2**31 - 1, None])
def test_ping_wire_encoding_boundary_values(instance_id: Optional[int]) -> None:
    """Exercise a handful of instance_id values through the typed
    encode / decode pair to catch signed-int overflow or ``None``
    handling regressions before they reach the wire."""
    spec = _TYPED_RPCS[RPC.Ping]
    proto_req = spec.python_to_request(instance_id)
    (round_tripped,) = spec.request_to_python(proto_req)
    assert round_tripped == instance_id

    proto_resp = spec.python_to_response(True)
    assert spec.response_to_python(proto_resp) is True

    # Sanity: the concrete PingRequest is the wire message.
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
    assert RPC.Lookup in _TYPED_RPCS
    spec = _TYPED_RPCS[RPC.Lookup]
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
    spec = _TYPED_RPCS[RPC.Lookup]
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Lookup,
        get_payload_classes(RPC.Lookup),
        HandlerType.BLOCKING,
        lookup_handler,
    )
    server.add_normal_thread_pool([RPC.Lookup], max_workers=2)
    server.start()

    try:
        client = MultiprocessGrpcClient(server_url)
        key = _sample_key(cache_salt="tenant-b")

        fut: MessagingFuture[None] = client.submit_request(RPC.Lookup, [key, 8])
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
        RPC.FreeLookupLocks,
        RPC.EndSession,
        RPC.UnregisterKvCache,
        RPC.UnregisterQCache,
        RPC.UnregisterKvCacheEngineDrivenContext,
        RPC.QueryPrefetchStatus,
        RPC.WaitPrefetchStatus,
        RPC.QueryPrefetchLookupHits,
        RPC.Clear,
        RPC.GetChunkSize,
        RPC.GetExperimental,
        RPC.Noop,
    ],
)
def test_wave2_rpc_is_typed(request_type: RpcMethod) -> None:
    """Each Wave 2 rpc must be off the msgspec envelope."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR
    assert spec.response_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR


def test_optional_chunk_count_none_roundtrip() -> None:
    """``int | None`` -> ``optional int64`` must preserve None."""
    for rt in (
        RPC.QueryPrefetchStatus,
        RPC.WaitPrefetchStatus,
        RPC.QueryPrefetchLookupHits,
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.FreeLookupLocks,
        get_payload_classes(RPC.FreeLookupLocks),
        HandlerType.BLOCKING,
        free_locks_handler,
    )
    server.add_normal_thread_pool([RPC.FreeLookupLocks], max_workers=1)
    server.start()
    try:
        client = MultiprocessGrpcClient(server_url)
        key = _sample_key(cache_salt="wave2")
        fut: MessagingFuture[None] = client.free_lookup_locks(key, 2)
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.GetChunkSize,
        get_payload_classes(RPC.GetChunkSize),
        HandlerType.SYNC,
        chunk_size_handler,
    )
    server.start()
    try:
        client = MultiprocessGrpcClient(server_url)
        fut: MessagingFuture[int] = client.get_chunk_size()
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.QueryPrefetchStatus,
        get_payload_classes(RPC.QueryPrefetchStatus),
        HandlerType.BLOCKING,
        status_handler,
    )
    server.add_normal_thread_pool([RPC.QueryPrefetchStatus], max_workers=2)
    server.start()
    try:
        client = MultiprocessGrpcClient(server_url)
        fut1: MessagingFuture[Optional[int]] = client.query_prefetch_status("req-1")
        fut2: MessagingFuture[Optional[int]] = client.query_prefetch_status("req-2")
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
        RPC.StoreQ,
        RPC.Store,
        RPC.Retrieve,
        RPC.ReportBlockAllocation,
        RPC.CbUnregisterKvCache,
        RPC.CbUnregisterRopeV3,
        RPC.CbLookupPreComputed,
        RPC.CbStorePreComputed,
        RPC.CbStoreFinal,
        RPC.CbRetrievePreComputed,
        RPC.CbLookupPreComputedV2,
        RPC.CbRetrievePreComputedV2,
        RPC.CbRetrievePreComputedV3,
        RPC.CbUnifiedLookup,
    ],
)
def test_wave3_rpc_is_typed(request_type: RpcMethod) -> None:
    """Wave 3 rpcs must all be off the msgspec envelope."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR
    assert spec.response_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR


def test_store_retrieve_roundtrip() -> None:
    """Store/Retrieve payload preservation across the wire boundary."""
    key = _sample_key()
    store_spec = _TYPED_RPCS[RPC.Store]
    proto_req = store_spec.python_to_request(key, 12, [[1, 2, 3], [4, 5]], b"event")
    assert isinstance(proto_req, lmcache_mq_pb2.StoreRequest)
    round_key, iid, blocks, event = store_spec.request_to_python(proto_req)
    assert round_key == key
    assert iid == 12
    assert blocks == [[1, 2, 3], [4, 5]]
    assert event == b"event"

    proto_resp = store_spec.python_to_response((b"eh", True))
    assert store_spec.response_to_python(proto_resp) == (b"eh", True)

    retrieve_spec = _TYPED_RPCS[RPC.Retrieve]
    proto_req = retrieve_spec.python_to_request(key, 12, [[7], [8, 9]], b"event2", 128)
    round_key, iid, blocks, event, skip = retrieve_spec.request_to_python(proto_req)
    assert round_key == key
    assert iid == 12
    assert blocks == [[7], [8, 9]]
    assert event == b"event2"
    assert skip == 128


def test_report_block_allocation_roundtrip() -> None:
    """BlockAllocationRecord list survives the wire in order."""
    spec = _TYPED_RPCS[RPC.ReportBlockAllocation]
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

    v1 = _TYPED_RPCS[RPC.CbLookupPreComputed]
    ranges = [(0, 8), (16, 24)]
    proto_resp = v1.python_to_response(ranges)
    assert v1.response_to_python(proto_resp) == ranges

    v2 = _TYPED_RPCS[RPC.CbLookupPreComputedV2]
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
    spec = _TYPED_RPCS[RPC.CbUnifiedLookup]

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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Store,
        get_payload_classes(RPC.Store),
        HandlerType.BLOCKING,
        store_handler,
    )
    server.add_normal_thread_pool([RPC.Store], max_workers=2)
    server.start()
    try:
        client = MultiprocessGrpcClient(server_url)
        key = _sample_key(cache_salt="wave3")
        fut: MessagingFuture[tuple[bytes, bool]] = client.submit_request(
            RPC.Store, [key, 42, [[1, 2], [3]], b"ev"]
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
        RPC.RegisterKvCacheEngineDrivenContext,
        RPC.PrepareStore,
        RPC.CommitStore,
        RPC.PrepareRetrieve,
        RPC.CommitRetrieve,
        RPC.P2PLookupAndLock,
        RPC.P2PQueryLookupResults,
        RPC.P2PUnlockObjects,
    ],
)
def test_wave4_rpc_is_typed(request_type: RpcMethod) -> None:
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR
    assert spec.response_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR


def test_register_edc_roundtrip() -> None:
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    spec = _TYPED_RPCS[RPC.RegisterKvCacheEngineDrivenContext]
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

    spec = _TYPED_RPCS[RPC.PrepareStore]
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

    spec = _TYPED_RPCS[RPC.PrepareRetrieve]
    resp = PrepareRetrieveResponse(success=True, data=b"hello", context={"slots": [7]})
    proto_resp = spec.python_to_response(resp)
    round = spec.response_to_python(proto_resp)
    assert round.success is True
    assert round.data == b"hello"
    assert round.context == {"slots": [7]}


def test_commit_store_and_retrieve_bool_roundtrip() -> None:
    key = _sample_key()

    cs = _TYPED_RPCS[RPC.CommitStore]
    proto_req = cs.python_to_request(key, 4, b"ctx-bytes")
    r_key, r_iid, r_ctx = cs.request_to_python(proto_req)
    assert (r_key, r_iid, r_ctx) == (key, 4, b"ctx-bytes")
    assert cs.response_to_python(cs.python_to_response(True)) is True
    assert cs.response_to_python(cs.python_to_response(False)) is False

    cr = _TYPED_RPCS[RPC.CommitRetrieve]
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

    lookup = _TYPED_RPCS[RPC.P2PLookupAndLock]
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
    query = _TYPED_RPCS[RPC.P2PQueryLookupResults]
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
    unlock = _TYPED_RPCS[RPC.P2PUnlockObjects]
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
        RPC.RegisterKvCache,
        RPC.RegisterQCache,
        RPC.CbRegisterKvCache,
        RPC.CbRegisterRopeV3,
    ],
)
def test_wave5_rpc_is_typed(request_type: RpcMethod) -> None:
    """Wave 5 finishes the migration: all request types use typed RPCs."""
    assert request_type in _TYPED_RPCS
    spec = _TYPED_RPCS[request_type]
    assert spec.request_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR
    assert spec.response_message.DESCRIPTOR.file is lmcache_mq_pb2.DESCRIPTOR


def test_typed_rpc_coverage_is_complete() -> None:
    """Enforce the migration invariant: every RpcMethod is typed."""
    missing = [rt.name for rt in RpcMethod if rt not in _TYPED_RPCS]
    assert not missing, f"legacy rpcs remain: {missing}"


@pytest.mark.parametrize("request_type", list(RpcMethod))
def test_typed_rpc_request_arity_matches_protocol(request_type: RpcMethod) -> None:
    """Keep generated bindings aligned with the public protocol contract."""
    spec = _TYPED_RPCS[request_type]
    assert spec.payload_types == tuple(get_payload_classes(request_type))
    assert spec.response_type == get_response_class(request_type)


def test_register_kv_cache_roundtrip() -> None:
    """DeviceIPCWrapper subclass identity + LayoutHints + EngineGroupInfo
    all survive the wire."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    spec = _TYPED_RPCS[RPC.RegisterKvCache]
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
            extra_object_group_tag=2,
            recurrent_state=True,
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
    spec = _TYPED_RPCS[RPC.CbRegisterKvCache]
    kv_cache = [_FakeIPCWrapper("cb")]
    proto_req = spec.python_to_request(9, kv_cache, "llama", 2)
    iid, r_kv, model, world = spec.request_to_python(proto_req)
    assert (iid, model, world) == (9, "llama", 2)
    assert isinstance(r_kv[0], _FakeIPCWrapper)
    assert r_kv[0].device_uuid == "test-uuid-cb"


def test_cb_register_rope_v3_roundtrip() -> None:
    spec = _TYPED_RPCS[RPC.CbRegisterRopeV3]
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.RegisterKvCache,
        get_payload_classes(RPC.RegisterKvCache),
        HandlerType.SYNC,
        handler,
    )
    server.start()
    try:
        client = MultiprocessGrpcClient(server_url)
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
            RPC.RegisterKvCache,
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
    client = MultiprocessGrpcClient(server_url)
    future: MessagingFuture[bool] = client.submit_request(RPC.Ping, [None])
    time.sleep(0.25)
    assert not future.query()

    server = MultiprocessGrpcServer(server_url)

    def handler(instance_id: Optional[int]) -> bool:
        return instance_id is None

    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
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

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        get_payload_classes(RPC.Ping),
        HandlerType.SYNC,
        handler,
    )
    server.start()
    client = MultiprocessGrpcClient(server_url)
    future: MessagingFuture[bool] = client.submit_request(RPC.Ping, [None])
    assert handler_entered.wait(timeout=5.0)

    server.close()
    release_handler.set()
    time.sleep(0.25)
    try:
        assert not future.query()
    finally:
        client.close()
