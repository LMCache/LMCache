# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for descriptor-driven gRPC on the mp-mode message queue."""

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
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import (
    MultiprocessGrpcClient,
    MultiprocessGrpcServer,
    request_type_to_method_name,
)
from lmcache.v1.multiprocess.protocol import RPC, HandlerType, RpcMethod
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2_grpc,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    compile_request_decoder,
    encode_request_from_call,
    get_request_message_class,
    get_response_message_class,
)

# Generated message classes are dynamic; rebind through Any so static analysis
# stops complaining about attribute lookups.
lmcache_mq_pb2: Any = _pb2_typed


def test_mq_import_does_not_load_grpc_runtime() -> None:
    """Importing MQ helpers must not initialize gRPC native libraries."""
    script = """
import sys
from lmcache.v1.multiprocess import mq

assert "grpc" not in sys.modules
assert "grpc_tools" not in sys.modules
assert mq.grpc is None
assert mq.request_type_to_method_name("Ping") == "Ping"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sample_key(
    *,
    worker_id: Optional[int] = 3,
    token_ids: tuple[int, ...] = (1, 2, 3, 4),
    cache_salt: str = "",
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="test-model",
        world_size=4,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id="req-42",
        cache_salt=cache_salt,
        num_kv_readers=1,
    )


def test_service_descriptor_covers_every_protocol_rpc() -> None:
    """The protocol method namespace is derived from generated services."""
    services = lmcache_mq_pb2.DESCRIPTOR.services_by_name
    service_methods = {
        method.name: (service.name, method)
        for service in services.values()
        for method in service.methods
    }
    expected_methods = {request_type_to_method_name(item) for item in RpcMethod}
    assert set(service_methods) == expected_methods

    for rpc_method in RpcMethod:
        service_name, method = service_methods[request_type_to_method_name(rpc_method)]
        assert service_name == rpc_method.service_name
        assert get_request_message_class(rpc_method).DESCRIPTOR is method.input_type
        assert get_response_message_class(rpc_method).DESCRIPTOR is method.output_type


def test_client_installs_function_style_rpc_methods() -> None:
    """Every protocol method is directly callable on the client."""
    for request_type in RpcMethod:
        method = getattr(MultiprocessGrpcClient, request_type.client_method_name, None)
        assert callable(method), request_type.name


def test_service_descriptor_has_no_legacy_batch_rpc() -> None:
    """The transport contract is unary-only; Batch is no longer part of it."""
    services = lmcache_mq_pb2.DESCRIPTOR.services_by_name
    assert "MessageQueue" not in services
    assert all("Batch" not in service.methods_by_name for service in services.values())


def test_ping_method_name_matches_proto() -> None:
    assert request_type_to_method_name(RPC.Ping) == "Ping"


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
        [],
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RPC.Noop,
        [],
        HandlerType.SYNC,
        noop_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        first: MessagingFuture[bool] = client.ping(0)
        assert first_ping_entered.wait(timeout=5.0)
        futures: list[MessagingFuture[Any]] = [
            client.ping(1),
            client.noop(),
            client.ping(2),
            client.noop(),
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
            [],
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
                futures.append(clients[0].ping(100 + index))
                futures.append(clients[1].ping(200 + index))
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
        [],
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_handler(
        RPC.Noop,
        [],
        HandlerType.SYNC,
        failing_noop_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        first: MessagingFuture[bool] = client.ping(0)
        assert first_ping_entered.wait(timeout=5.0)
        failed: MessagingFuture[str] = client.noop()
        succeeded: MessagingFuture[bool] = client.ping(1)
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
        first: MessagingFuture[bool] = client.ping(0)
        assert first_ping_entered.wait(timeout=5.0)
        second: MessagingFuture[bool] = client.ping(1)
        release_first_ping.set()

        assert first.result(timeout=5.0) is True
        assert second.result(timeout=5.0) is True
    finally:
        release_first_ping.set()
        client.close()
        server.stop(grace=None).wait()


def test_ping_roundtrip_with_positional_and_keyword_fields() -> None:
    """The client can construct protobuf requests from function-style calls."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen_calls: list[Optional[int]] = []

    def ping_handler(instance_id: Optional[int]) -> bool:
        seen_calls.append(instance_id)
        return True

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        [],
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert client.ping(42).result(timeout=5.0) is True
        assert seen_calls[-1] == 42

        assert client.ping(instance_id=None).result(timeout=5.0) is True
        assert seen_calls[-1] is None

        assert len(seen_calls) == 2
    finally:
        client.close()
        server.close()


def test_lookup_request_codec_uses_handler_annotations() -> None:
    """Proto request fields decode to the Python handler signature."""

    def lookup_handler(key: IPCCacheServerKey, tp_size: int) -> None:
        del key, tp_size

    request_cls = get_request_message_class(RPC.Lookup)
    request = encode_request_from_call(request_cls, (_sample_key(), 4), {})
    decoder, payload_types = compile_request_decoder(request_cls, lookup_handler)

    assert isinstance(request, lmcache_mq_pb2.LookupRequest)
    assert payload_types == (IPCCacheServerKey, int)
    assert decoder(request) == (_sample_key(), 4)


def test_lookup_typed_roundtrip() -> None:
    """Server handler sees the exact key the client sent."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    seen: list[tuple[IPCCacheServerKey, int]] = []

    def lookup_handler(key: IPCCacheServerKey, tp_size: int) -> None:
        seen.append((key, tp_size))
        return None

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Lookup,
        [],
        HandlerType.BLOCKING,
        lookup_handler,
    )
    server.add_normal_thread_pool([RPC.Lookup], max_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        key = _sample_key(cache_salt="tenant-b")
        fut: MessagingFuture[None] = client.lookup(key, 8)
        assert fut.result(timeout=5.0) is None
        assert seen == [(key, 8)]
    finally:
        client.close()
        server.close()


def test_optional_chunk_count_response_roundtrip() -> None:
    """optional int64 chunk_count comes back as int and as None."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    results: dict[str, Optional[int]] = {"req-1": 7, "req-2": None}

    def status_handler(request_id: str) -> Optional[int]:
        return results[request_id]

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.QueryPrefetchStatus,
        [],
        HandlerType.BLOCKING,
        status_handler,
    )
    server.add_normal_thread_pool([RPC.QueryPrefetchStatus], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert client.query_prefetch_status("req-1").result(timeout=5.0) == 7
        assert client.query_prefetch_status("req-2").result(timeout=5.0) is None
    finally:
        client.close()
        server.close()


def test_store_typed_grpc_roundtrip() -> None:
    """Store response keeps the tuple shape used by device futures."""
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
        return b"evt", True

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Store,
        [],
        HandlerType.BLOCKING,
        store_handler,
    )
    server.add_normal_thread_pool([RPC.Store], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        key = _sample_key()
        result = client.store(key, 5, [[1, 2], [3]], b"ipc").result(timeout=5.0)
        assert result == (b"evt", True)
        assert seen == [(key, 5, [[1, 2], [3]], b"ipc")]
    finally:
        client.close()
        server.close()


def test_engine_driven_struct_responses_roundtrip() -> None:
    """Struct-like request/response objects remain Python objects to callers."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    def register_handler(
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        assert payload.instance_id == 9
        return RegisterEngineDrivenContextResponse(shm_name="shm-a", pool_size=1024)

    def prepare_retrieve_handler(
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        assert key.request_id == "req-42"
        assert instance_id == 9
        return PrepareRetrieveResponse(
            success=True,
            data=b"payload",
            context={"slots": [{"offset": 0}]},
        )

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(RPC.RegisterKvCacheEngineDrivenContext, register_handler)
    server.add_handler(
        RPC.PrepareRetrieve,
        [],
        HandlerType.BLOCKING,
        prepare_retrieve_handler,
    )
    server.add_normal_thread_pool([RPC.PrepareRetrieve], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        registration = client.register_kv_cache_engine_driven_context(
            RegisterEngineDrivenContextPayload(
                instance_id=9,
                model_name="m",
                world_size=1,
                block_size=16,
                num_layers=2,
                hidden_dim_size=8,
                dtype_str="float16",
                use_mla=False,
            )
        ).result(timeout=5.0)
        assert registration == RegisterEngineDrivenContextResponse(
            shm_name="shm-a", pool_size=1024
        )

        prepared = client.prepare_retrieve(_sample_key(), 9).result(timeout=5.0)
        assert prepared == PrepareRetrieveResponse(
            success=True,
            data=b"payload",
            context={"slots": [{"offset": 0}]},
        )
    finally:
        client.close()
        server.close()


def test_p2p_query_lookup_results_roundtrip() -> None:
    """Optional nested repeated protobuf responses map back to Python values."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    def query_handler(task_id: int) -> list[TransferChannelAddress] | None:
        if task_id == 1:
            return [
                TransferChannelAddress(offset=16, size=32),
                TransferChannelAddress(offset=64, size=128),
            ]
        return None

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.P2PQueryLookupResults,
        [],
        HandlerType.BLOCKING,
        query_handler,
    )
    server.add_normal_thread_pool([RPC.P2PQueryLookupResults], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert client.p2p_query_lookup_results(1).result(timeout=5.0) == [
            TransferChannelAddress(offset=16, size=32),
            TransferChannelAddress(offset=64, size=128),
        ]
        assert client.p2p_query_lookup_results(2).result(timeout=5.0) is None
    finally:
        client.close()
        server.close()


def test_cb_unified_lookup_response_roundtrip() -> None:
    """CacheBlend's optional structured response preserves nested dataclasses."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    result = CBUnifiedLookupResult(
        prefix_coverage_tokens=256,
        non_prefix_segments=[
            CBMatchResult(old_st=0, old_ed=64, cur_st=128, cur_ed=192, hash=b"a")
        ],
        segmented_prefix_segments=[
            CBMatchResult(old_st=64, old_ed=128, cur_st=64, cur_ed=128, hash=b"b")
        ],
    )

    def handler(key: IPCCacheServerKey, tp_size: int) -> CBUnifiedLookupResult | None:
        assert key.request_id == "req-42"
        assert tp_size == 2
        return result

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.CbUnifiedLookup,
        [],
        HandlerType.BLOCKING,
        handler,
    )
    server.add_normal_thread_pool([RPC.CbUnifiedLookup], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert client.cb_unified_lookup(_sample_key(), 2).result(timeout=5.0) == result
    finally:
        client.close()
        server.close()


class _FakeIPCWrapper(DeviceIPCWrapper):
    """Minimal wrapper subclass used to exercise pickle-in-proto payloads."""

    def __init__(self, tag: str = "fake") -> None:
        self.handle = ("fake-handle", tag)
        self.dtype = torch.float16
        self.shape = (2, 4)
        self.stride = (4, 1)
        self.storage_offset = 0
        self.device_uuid = "test-uuid-" + tag


def test_register_kv_cache_grpc_roundtrip() -> None:
    """DeviceIPCWrapper, LayoutHints, and EngineGroupInfo survive the wire."""
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
                kv_cache,
                model_name,
                world_size,
                engine_type,
                layout_hints,
                engine_group_infos,
            )
        )
        return None

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(RPC.RegisterKvCache, handler)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        kv = [_FakeIPCWrapper("e2e")]
        hints: dict[str, Any] = {"kv_layout": "HND"}
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=32,
                sw_size_tokens=-1,
            ),
        ]
        fut: MessagingFuture[None] = client.register_kv_cache(
            5, kv, "opt-125m", 2, EngineType.VLLM, hints, groups
        )
        assert fut.result(timeout=5.0) is None
        assert len(seen) == 1

        iid, r_kv, model, world, engine_type, r_hints, r_groups = seen[0]
        assert iid == 5
        assert model == "opt-125m"
        assert world == 2
        assert engine_type is EngineType.VLLM
        assert r_hints == hints
        assert r_groups == groups
        assert len(r_kv) == 1
        assert isinstance(r_kv[0], _FakeIPCWrapper)
        assert r_kv[0].device_uuid == "test-uuid-e2e"
    finally:
        client.close()
        server.close()


def test_p2p_lookup_request_roundtrip() -> None:
    """Map and nested-message fields decode through handler annotations."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    key = ObjectKey(chunk_hash=b"abc", model_name="m", kv_rank=1)
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3]), torch.Size([4])],
        dtypes=[torch.float16, torch.bfloat16],
    )

    def handler(
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        assert keys == [key]
        assert group_layout_descs == {7: layout}
        return 99

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.P2PLookupAndLock,
        [],
        HandlerType.BLOCKING,
        handler,
    )
    server.add_normal_thread_pool([RPC.P2PLookupAndLock], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert client.p2p_lookup_and_lock([key], {7: layout}).result(timeout=5.0) == 99
    finally:
        client.close()
        server.close()


def test_grpc_request_stays_pending_when_server_is_not_ready() -> None:
    """wait_for_ready keeps startup races pending instead of failing fast."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    client = MultiprocessGrpcClient(server_url)

    try:
        future: MessagingFuture[bool] = client.ping(1)
        assert not future.wait(timeout=0.1)
    finally:
        client.close()


def test_grpc_request_stays_pending_when_server_stops_mid_call() -> None:
    """UNAVAILABLE keeps in-flight calls pending for server restart."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    entered = threading.Event()
    release = threading.Event()

    def ping_handler(instance_id: Optional[int]) -> bool:
        del instance_id
        entered.set()
        release.wait(timeout=5.0)
        return True

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(
        RPC.Ping,
        [],
        HandlerType.BLOCKING,
        ping_handler,
    )
    server.add_normal_thread_pool([RPC.Ping], max_workers=1)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        future: MessagingFuture[bool] = client.ping(1)
        assert entered.wait(timeout=5.0)
        server.close()
        assert not future.wait(timeout=0.1)
    finally:
        release.set()
        client.close()
        server.close()
