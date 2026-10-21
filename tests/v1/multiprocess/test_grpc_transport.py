# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the generated-service gRPC transport."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import importlib
import queue
import subprocess
import sys
import threading
import time

# Third Party
import grpc
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
    KVEventBatch,
    KVEventRecord,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.ext_server_module import ExtServerModuleRouter
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.plugin_modules.qstore import QStoreModule
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.request_handler import (
    HandlerType,
    iter_request_handlers,
    request_handler,
)
from lmcache.v1.multiprocess.transport.grpc_impl import (
    proto_codec,
)
from lmcache.v1.multiprocess.transport.grpc_impl import (
    server_builder as grpc_server_builder,
)
from lmcache.v1.multiprocess.transport.grpc_impl.client import (
    CONNECT_TIMEOUT_MS,
    INITIAL_RECONNECT_BACKOFF_MS,
    MAX_RECONNECT_BACKOFF_MS,
    GrpcMultiprocessClient,
)
from lmcache.v1.multiprocess.transport.grpc_impl.codecs import (
    get_message_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    get_service_bindings,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)
from lmcache.v1.platform.devices.cpu.shm import CpuShmTensorWrapper

_GRPC_MODULE_TYPES = (
    LookupModule,
    ManagementModule,
    KVEventModule,
    P2PController,
    LMCacheDrivenTransferModule,
    EngineDrivenTransferModule,
    QStoreModule,
    BlendModule,
    ExtServerModuleRouter,
)

_GRPC_SERVICE_NAME_OVERRIDES = {
    EngineDrivenTransferModule: "EngineDrivenService",
    LMCacheDrivenTransferModule: "LMCacheDrivenService",
    P2PController: "P2PService",
    ExtServerModuleRouter: "ExtensionService",
}


def _expected_grpc_service_name(module_type: type[Any]) -> str:
    """Derive the gRPC service name owned by a server module.

    Args:
        module_type: Server module class whose service name is required.

    Returns:
        The conventional service name or its established abbreviated form.
    """
    return _GRPC_SERVICE_NAME_OVERRIDES.get(
        module_type,
        module_type.__name__.removesuffix("Module") + "Service",
    )


@dataclass
class _Calls:
    lookup: tuple[IPCCacheServerKey, int] | None = None
    allocation: tuple[int, str, list[BlockAllocationRecord]] | None = None
    clear_force: bool | None = None
    chunk_negotiation: int | None = None


@pytest.fixture(params=["sync", "aio"])
def grpc_client(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[GrpcMultiprocessClient, _Calls]]:
    calls = _Calls()

    class FakeModules:
        @request_handler(HandlerType.BLOCKING)
        def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
            calls.lookup = (key, tp_size)

        @request_handler(
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def store(
            self,
            key: IPCCacheServerKey,
            instance_id: int,
            block_ids: list[list[int]],
            event_ipc_handle: bytes,
        ) -> tuple[bytes, bool]:
            assert instance_id == 7
            assert block_ids == [[1, 2], [3]]
            assert event_ipc_handle == b"input-event"
            return b"output-event", key.model_name == "model"

        @request_handler(
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def prepare_store(
            self, key: IPCCacheServerKey, instance_id: int
        ) -> PrepareStoreResponse:
            assert key.request_configs == {"blend": True}
            assert instance_id == 7
            return PrepareStoreResponse(
                context={"slots": [{"offset": 8}], "chunk_indices": [2]}
            )

        @request_handler(
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def prepare_retrieve(
            self, key: IPCCacheServerKey, instance_id: int
        ) -> PrepareRetrieveResponse:
            assert key.request_configs == {"blend": True}
            assert instance_id == 7
            return PrepareRetrieveResponse(
                success=True,
                data=b"retrieved",
                context={"slot": 3},
            )

        @request_handler()
        def register_kv_cache_engine_driven_context(
            self, payload: RegisterEngineDrivenContextPayload
        ) -> RegisterEngineDrivenContextResponse:
            assert payload.num_physical_slots == 32
            return RegisterEngineDrivenContextResponse("shared-memory", 4096)

        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return instance_id == 7

        @request_handler(HandlerType.BLOCKING)
        def clear(self, force: bool = False) -> None:
            calls.clear_force = force

        @request_handler()
        def negotiate_chunk_size(self, required_chunk_alignment: int) -> int:
            calls.chunk_negotiation = required_chunk_alignment
            return 640

        @request_handler(operation="noop")
        def debug(self) -> str:
            return "ok"

        @request_handler(
            HandlerType.BLOCKING,
            operation="report_block_allocation",
        )
        def report_block_allocations(
            self,
            instance_id: int,
            model_name: str,
            records: list[BlockAllocationRecord],
        ) -> None:
            calls.allocation = (instance_id, model_name, records)

        @request_handler(HandlerType.BLOCKING)
        def cb_unified_lookup(
            self, key: IPCCacheServerKey, tp_size: int
        ) -> CBUnifiedLookupResult | None:
            assert key.model_name == "model"
            assert tp_size == 2
            return CBUnifiedLookupResult(
                prefix_coverage_tokens=16,
                non_prefix_segments=[CBMatchResult(0, 2, 4, 6, b"hash")],
            )

        @request_handler(HandlerType.BLOCKING)
        def p2p_lookup_and_lock(
            self,
            keys: list[ObjectKey],
            group_layout_descs: dict[int, MemoryLayoutDesc],
        ) -> int:
            assert keys[0].cache_salt == "tenant"
            assert group_layout_descs[0].shapes == [torch.Size([2, 4])]
            assert group_layout_descs[0].dtypes == [torch.float16]
            return 41

        @request_handler(HandlerType.BLOCKING)
        def p2p_query_lookup_results(
            self, task_id: int
        ) -> list[TransferChannelAddress] | None:
            assert task_id == 41
            return [TransferChannelAddress(offset=8, size=16)]

        @request_handler(HandlerType.BLOCKING)
        def server_module_call(
            self, request: ServerModuleCallRequest
        ) -> ServerModuleCallResponse:
            return ServerModuleCallResponse(
                success=request.method == "fake.echo",
                payload=b"grpc:" + request.payload,
                error="" if request.method == "fake.echo" else "missing",
            )

    modules: Any = FakeModules()
    config = MPServerConfig(
        transport="grpc",
        host="127.0.0.1",
        port=0,
        max_cpu_workers=2,
        max_gpu_workers=1,
        grpc_server_workers=4,
        grpc_server_mode=request.param,
    )
    server = build_grpc_request_server([modules], config)
    server.start()
    client = GrpcMultiprocessClient(  # type: ignore[abstract]
        f"grpc://127.0.0.1:{server.bound_port}"
    )
    try:
        yield client, calls
    finally:
        client.close()
        server.close()


def test_rpc_surface_is_derived_from_split_service_descriptors() -> None:
    """Every generated RPC has one codec derived from its gRPC service."""
    bindings = get_service_bindings()
    assert set(bindings) == {
        _expected_grpc_service_name(module_type) for module_type in _GRPC_MODULE_TYPES
    }

    registry = get_method_codec_registry()
    generated_methods = {method.full_name for _, method in iter_methods()}
    assert set(registry.by_full_name) == generated_methods

    lookup_codec = registry.by_full_name["lmcache.mp.LookupService.Lookup"]
    assert lookup_codec.operation == "lookup"
    assert lookup_codec.payload_types == (IPCCacheServerKey, int)
    assert lookup_codec.response_type is type(None)

    store_codec = registry.by_full_name["lmcache.mp.LMCacheDrivenService.Store"]
    assert store_codec.payload_types == (
        IPCCacheServerKey,
        int,
        list[list[int]],
        bytes,
    )
    assert store_codec.response_type == tuple[bytes, bool]

    clear_codec = registry.by_full_name["lmcache.mp.ManagementService.Clear"]
    assert clear_codec.operation == "clear"
    assert clear_codec.payload_types == (bool,)
    assert clear_codec.request_decoder(clear_codec.request_encoder((), {})) == (False,)
    assert clear_codec.request_decoder(
        clear_codec.request_encoder((), {"force": True})
    ) == (True,)

    negotiate_codec = registry.by_full_name[
        "lmcache.mp.ManagementService.NegotiateChunkSize"
    ]
    assert negotiate_codec.operation == "negotiate_chunk_size"
    assert negotiate_codec.payload_types == (int,)
    assert negotiate_codec.response_type is int

    registration_codec = registry.by_full_name[
        "lmcache.mp.EngineDrivenService.RegisterKvCacheEngineDrivenContext"
    ]
    registration_request = registration_codec.request_encoder(
        (),
        {
            "instance_id": 7,
            "model_name": "model",
            "world_size": 2,
            "block_size": 16,
            "num_layers": 32,
            "hidden_dim_size": 128,
            "dtype_str": "float16",
            "use_mla": False,
            "num_physical_slots": 32,
        },
    )
    assert registration_codec.request_decoder(registration_request) == (
        RegisterEngineDrivenContextPayload(
            instance_id=7,
            model_name="model",
            world_size=2,
            block_size=16,
            num_layers=32,
            hidden_dim_size=128,
            dtype_str="float16",
            use_mla=False,
            num_physical_slots=32,
        ),
    )
    server_module_codec = registry.by_full_name[
        "lmcache.mp.ExtensionService.ServerModuleCall"
    ]
    assert server_module_codec.operation == "server_module_call"
    assert server_module_codec.payload_types == (ServerModuleCallRequest,)
    assert server_module_codec.response_type is ServerModuleCallResponse


def test_module_annotations_cover_and_match_generated_grpc_methods() -> None:
    """Every generated RPC resolves to one compatible module annotation."""
    handlers = {
        registered.operation: registered.handler
        for module_type in _GRPC_MODULE_TYPES
        for registered in iter_request_handlers(module_type)
    }
    registry = get_method_codec_registry()
    codecs = tuple(registry.by_full_name.values())

    assert set(handlers) == {codec.operation for codec in codecs}
    for codec in codecs:
        codec.validate_handler(handlers[codec.operation])


def test_grpc_imports_do_not_load_zmq_runtime() -> None:
    """The gRPC transport imports without loading the ZMQ runtime."""
    script = r"""
import importlib.abc
import sys

banned = (
    "lmcache.v1.multiprocess.transport.zmq_impl",
)


class LegacyProtocolBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in banned):
            raise ImportError(f"gRPC imported legacy module: {fullname}")
        return None


sys.meta_path.insert(0, LegacyProtocolBlocker())
import lmcache.v1.multiprocess.transport.grpc_impl.client
import lmcache.v1.multiprocess.transport.grpc_impl.server
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_generated_module_import_does_not_load_runtime_dependencies() -> None:
    """Generated binding checks work in a PEP 517 build environment."""
    script = r"""
import importlib.abc
import sys

banned = (
    "cachetools",
    "lmcache.utils",
    "lmcache.v1.multiprocess.futures",
    "lmcache.v1.multiprocess.transport.base",
)


class RuntimeDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in banned):
            raise ImportError(
                f"generated module imported runtime dependency: {fullname}"
            )
        return None


sys.meta_path.insert(0, RuntimeDependencyBlocker())
import lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.common_pb2
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_generated_protobuf_type_stubs_are_available() -> None:
    """Generated protobuf modules include static type information."""
    module_names = (
        "common_pb2",
        "p2p_service_pb2",
    )
    package = "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen"

    for module_name in module_names:
        module = importlib.import_module(f"{package}.{module_name}")
        assert module.__file__ is not None
        module_path = Path(module.__file__)
        assert module_path.with_suffix(".pyi").is_file()


def test_client_channel_bounds_connect_timeout_and_reconnect_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client channel uses the same reconnect bounds as the ZMQ client."""
    captured: dict[str, Any] = {}
    insecure_channel = grpc.insecure_channel

    def record_channel(target: str, options: Any = None) -> grpc.Channel:
        captured.update(options)
        return insecure_channel(target, options=options)

    monkeypatch.setattr(grpc, "insecure_channel", record_channel)

    client = GrpcMultiprocessClient(  # type: ignore[abstract]
        "grpc://127.0.0.1:15706"
    )
    client.close()

    assert captured["grpc.min_reconnect_backoff_ms"] == CONNECT_TIMEOUT_MS
    assert captured["grpc.initial_reconnect_backoff_ms"] == (
        INITIAL_RECONNECT_BACKOFF_MS
    )
    assert captured["grpc.max_reconnect_backoff_ms"] == MAX_RECONNECT_BACKOFF_MS


def test_build_grpc_request_server_uses_configured_server_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gRPC server builder should not rely on constructor defaults."""
    modules: Any = [object()]

    class FakeGrpcServer:
        def __init__(
            self,
            bind_url: str,
            max_cpu_workers: int,
            max_gpu_workers: int,
            grpc_server_workers: int,
        ) -> None:
            self.args = (
                bind_url,
                max_cpu_workers,
                max_gpu_workers,
                grpc_server_workers,
            )
            self.modules: Any = None

        def add_modules(self, modules: Any) -> None:
            self.modules = modules

    monkeypatch.setattr(
        grpc_server_builder,
        "GrpcMultiprocessServer",
        FakeGrpcServer,
    )
    config = MPServerConfig(
        transport="grpc",
        host="127.0.0.1",
        port=6000,
        max_cpu_workers=2,
        max_gpu_workers=3,
        grpc_server_workers=7,
    )

    server = cast(
        Any,
        build_grpc_request_server(modules, config),
    )

    assert server.args == ("grpc://127.0.0.1:6000", 2, 3, 7)
    assert server.modules is modules


def test_grpc_server_registers_out_of_tree_services() -> None:
    """Server modules may attach package-owned gRPC services before start."""
    registered_servers: list[Any] = []

    class ServiceModule:
        def register_grpc_services(self, server: Any) -> None:
            registered_servers.append(server)

    server = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=1,
        max_gpu_workers=1,
        grpc_server_workers=1,
    )
    try:
        server.add_modules([cast(Any, ServiceModule())])
    finally:
        server.close()

    assert len(registered_servers) == 1


def test_service_message_codec_registry_round_trips_custom_types() -> None:
    """Service-owned codecs handle only their registered protobuf types."""
    registry = get_message_codec_registry()
    common_pb2 = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.common_pb2"
    )
    p2p_service_pb2 = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.p2p_service_pb2"
    )

    wrapper = CpuShmTensorWrapper(
        torch.empty((2, 4), dtype=torch.float16), "/lmcache_codec_test"
    )
    wrapper_message = common_pb2.DeviceIpcWrapper()
    wrapper_codec = registry.find(wrapper_message.DESCRIPTOR, type(wrapper))
    assert wrapper_codec is not None
    wrapper_codec.writer(wrapper_message, wrapper)
    decoded_wrapper = wrapper_codec.reader(wrapper_message)
    assert type(decoded_wrapper) is CpuShmTensorWrapper
    assert decoded_wrapper == wrapper

    shape = torch.Size([2, 4])
    shape_message = p2p_service_pb2.TensorShape()
    shape_codec = registry.find(shape_message.DESCRIPTOR, torch.Size)
    assert shape_codec is not None
    shape_codec.writer(shape_message, shape)
    assert shape_codec.reader(shape_message) == shape


def test_response_encoder_compiles_its_codec_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoding many responses compiles the structural codec only once."""
    kv_event_service_pb2 = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.kv_event_service_pb2"
    )
    compiled: list[Any] = []
    compile_message_codec = proto_codec._compile_message_codec

    def counting_compile(descriptor: Any, py_type: Any) -> Any:
        compiled.append(descriptor.full_name)
        return compile_message_codec(descriptor, py_type)

    monkeypatch.setattr(proto_codec, "_compile_message_codec", counting_compile)
    encode = proto_codec.compile_response_encoder_for_type(
        kv_event_service_pb2.KVEventBatch, KVEventBatch
    )
    record = KVEventRecord(1, "stored", "cpu", "model", [b"h"], None, [1, 2, 3])
    batch = KVEventBatch(True, 1, 2, False, [record])

    first = encode(batch).SerializeToString()
    after_first = len(compiled)
    for _ in range(3):
        assert encode(batch).SerializeToString() == first
    assert after_first > 0
    assert len(compiled) == after_first


def test_generated_grpc_services_communicate_end_to_end(
    grpc_client: tuple[GrpcMultiprocessClient, _Calls],
) -> None:
    client, calls = grpc_client
    key = IPCCacheServerKey(
        model_name="model",
        world_size=2,
        worker_id=None,
        token_ids=(1, 2, 3),
        start=0,
        end=3,
        request_id="request",
        cache_salt="tenant",
        request_configs={"blend": True},
        num_kv_readers=2,
    )

    assert client.lookup(key, 2).result(timeout=5) is None
    assert calls.lookup == (key, 2)
    assert client.store(key, 7, [[1, 2], [3]], b"input-event").result(5) == (
        b"output-event",
        True,
    )
    assert client.prepare_store(key, 7).result(5) == PrepareStoreResponse(
        context={"slots": [{"offset": 8}], "chunk_indices": [2]}
    )
    assert client.prepare_retrieve(key, 7).result(5) == PrepareRetrieveResponse(
        success=True,
        data=b"retrieved",
        context={"slot": 3},
    )
    registration = client.register_kv_cache_engine_driven_context(
        RegisterEngineDrivenContextPayload(
            instance_id=7,
            model_name="model",
            world_size=2,
            block_size=16,
            num_layers=32,
            hidden_dim_size=128,
            dtype_str="float16",
            use_mla=False,
            num_physical_slots=32,
        )
    ).result(5)
    assert registration == RegisterEngineDrivenContextResponse("shared-memory", 4096)
    assert client.ping(7).result(5) is True
    assert client.clear().result(5) is None
    assert calls.clear_force is False
    assert client.clear(force=True).result(5) is None
    assert calls.clear_force is True
    assert client.negotiate_chunk_size(640).result(5) == 640
    assert calls.chunk_negotiation == 640
    assert client.noop().result(5) == "ok"

    records = [BlockAllocationRecord("request", [4], [5, 6])]
    assert client.report_block_allocation(7, "model", records).result(5) is None
    assert calls.allocation == (7, "model", records)

    blend_result = client.cb_unified_lookup(key, 2).result(5)
    assert blend_result == CBUnifiedLookupResult(
        prefix_coverage_tokens=16,
        non_prefix_segments=[CBMatchResult(0, 2, 4, 6, b"hash")],
    )

    task_id = client.p2p_lookup_and_lock(
        [ObjectKey(b"chunk", "model", 0, cache_salt="tenant")],
        {0: MemoryLayoutDesc([torch.Size([2, 4])], [torch.float16])},
    ).result(5)
    assert task_id == 41
    assert client.p2p_query_lookup_results(task_id).result(5) == [
        TransferChannelAddress(offset=8, size=16)
    ]


def test_generated_grpc_server_module_call_round_trips(
    grpc_client: tuple[GrpcMultiprocessClient, _Calls],
) -> None:
    client, _calls = grpc_client

    response = client.server_module_call(
        ServerModuleCallRequest(method="fake.echo", payload=b"hello")
    ).result(timeout=1)

    assert response == ServerModuleCallResponse(
        success=True,
        payload=b"grpc:hello",
        error="",
    )


@pytest.mark.parametrize("grpc_server_mode", ["sync", "aio"])
def test_grpc_server_keeps_handler_concurrency(grpc_server_mode: str) -> None:
    """Normal handlers stay within max_cpu_workers, and a client's affinity
    handlers all run on one worker."""
    lock = threading.Lock()
    running = peak = 0
    affinity_threads: dict[int, set[int]] = {}

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.02)
            with lock:
                running -= 1
            return True

        @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
        def clear(self, force: bool = False) -> None:
            affinity_threads.setdefault(int(force), set()).add(threading.get_ident())

    config = MPServerConfig(
        transport="grpc",
        host="127.0.0.1",
        port=0,
        max_cpu_workers=2,
        max_gpu_workers=4,
        grpc_server_mode=cast(Any, grpc_server_mode),
    )
    server = build_grpc_request_server([cast(Any, Module())], config)
    server.start()
    url = f"grpc://127.0.0.1:{server.bound_port}"
    clients = [GrpcMultiprocessClient(url) for _ in range(2)]  # type: ignore[abstract]
    try:
        assert all(f.result(5) for f in [clients[0].ping(i) for i in range(8)])
        for client, force in zip(clients, (False, True), strict=True):
            for future in [client.clear(force=force) for _ in range(8)]:
                future.result(5)
    finally:
        for client in clients:
            client.close()
        server.close()

    assert peak == 2
    assert [len(threads) for threads in affinity_threads.values()] == [1, 1]


@pytest.mark.parametrize("grpc_server_mode", ["sync", "aio"])
def test_stalled_stream_subscriber_stops_server_reads(grpc_server_mode: str) -> None:
    """A subscriber that stops reading leaves later batches unread in the
    handler's stream instead of buffered on the server."""
    reads = 0

    class Module:
        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            def read(closed: threading.Event) -> KVEventBatch:
                nonlocal reads
                if closed.is_set():
                    raise StopIteration
                reads += 1
                record = KVEventRecord(
                    reads, "store", "cpu", model_name, [b"h"], None, [1] * 4096
                )
                return KVEventBatch(True, 1, reads, False, [record])

            return MessagingStream(read, lambda: None)

    config = MPServerConfig(
        transport="grpc",
        host="127.0.0.1",
        port=0,
        max_cpu_workers=2,
        max_gpu_workers=1,
        grpc_server_mode=cast(Any, grpc_server_mode),
    )
    server = build_grpc_request_server([cast(Any, Module())], config)
    server.start()
    client = GrpcMultiprocessClient(f"grpc://127.0.0.1:{server.bound_port}")  # type: ignore[abstract]
    try:
        stream = client.subscribe_kv_events(1, "model", 0, 16)
        next(stream)
        # Reads stop once the flow-control window is full.
        deadline = time.monotonic() + 10
        last = -1
        while reads != last:
            assert time.monotonic() < deadline, f"server kept reading: {reads}"
            last = reads
            time.sleep(0.5)
        stream.close()
    finally:
        client.close()
        server.close()


def test_aio_server_releases_threads_when_bind_fails() -> None:
    """A failed bind leaves no event loop or worker threads behind."""
    before = set(threading.enumerate())
    config = MPServerConfig(
        transport="grpc",
        host="nonexistent.invalid",
        port=0,
        max_cpu_workers=2,
        max_gpu_workers=2,
        grpc_server_mode="aio",
    )

    with pytest.raises(RuntimeError):
        build_grpc_request_server([], config)

    leaked = [
        thread.name
        for thread in set(threading.enumerate()) - before
        if thread.name.startswith("grpc-")
    ]
    assert leaked == []


@pytest.mark.parametrize("max_cpu_workers", [1, 2])
def test_grpc_normal_worker_limit_and_error_release(max_cpu_workers: int) -> None:
    """Normal handlers honor capacity and release it even on exceptions."""
    entered: queue.Queue[None] = queue.Queue()
    release = threading.Event()

    class BlockingModule:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            entered.put(None)
            assert release.wait(10)
            if instance_id == 0:
                raise RuntimeError("handler failed")
            return True

    server = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=max_cpu_workers,
        max_gpu_workers=1,
        grpc_server_workers=4,
    )
    server.add_modules([BlockingModule()])
    server.start()
    client = GrpcMultiprocessClient(  # type: ignore[abstract]
        f"grpc://127.0.0.1:{server.bound_port}"
    )
    try:
        futures = [client.ping(i) for i in range(max_cpu_workers + 1)]
        for _ in range(max_cpu_workers):
            entered.get(timeout=5)
        with pytest.raises(queue.Empty):
            entered.get(timeout=0.1)
        release.set()
        with pytest.raises(grpc.RpcError, match="handler failed"):
            futures[0].result(5)
        assert all(future.result(5) for future in futures[1:])
        assert client.ping(1).result(5) is True
    finally:
        release.set()
        client.close()
        server.close()


def test_blend_rope_windows_and_block_groups_round_trip_the_proto() -> None:
    """3-element view windows ([offset, width, scalar]) and ragged per-group
    block tables (HMA / compressed groups) survive the gRPC codec."""
    registry = get_method_codec_registry()
    rope = registry.by_full_name["lmcache.mp.BlendService.CbRegisterRope"]
    group_rot = [[], [224, 64, 15], [0, 64]]
    decoded = rope.request_decoder(
        rope.request_encoder(
            (),
            {
                "instance_id": 7,
                "cos_sin_caches_ipc": [],
                "head_size": 512,
                "is_neox_style": False,
                "group_to_cache": [0, 0, 1],
                "group_rot": group_rot,
                "group_head_size": [512, 584, 512],
                "kv_quant": "",
            },
        )
    )
    assert list(decoded[4]) == [0, 0, 1]
    assert [list(w) for w in decoded[5]] == group_rot
    assert list(decoded[6]) == [512, 584, 512]

    retrieve = registry.by_full_name["lmcache.mp.BlendService.CbRetrievePreComputed"]
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=(1, 2, 3),
        start=0,
        end=3,
        request_id="request",
        cache_salt="",
        request_configs={"blend": True},
        num_kv_readers=1,
    )
    groups = [[], [5], [2, 1, 235], [], [9, 9]]
    decoded = retrieve.request_decoder(
        retrieve.request_encoder(
            (),
            {
                "key": key,
                "cb_match_result": [CBMatchResult(0, 256, 512, 768, b"h")],
                "gpu_block_ids": groups,
                "instance_id": 7,
                "event_ipc_handle": b"ev",
            },
        )
    )
    assert [list(g) for g in decoded[2]] == groups
    assert decoded[1] == [CBMatchResult(0, 256, 512, 768, b"h")]
