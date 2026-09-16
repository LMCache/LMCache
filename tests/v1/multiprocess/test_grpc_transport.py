# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the generated-service gRPC transport."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import importlib
import subprocess
import sys

# Third Party
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
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.request_handler import (
    iter_request_handlers,
    request_handler,
)
from lmcache.v1.multiprocess.transport.grpc_impl import server as grpc_server_module
from lmcache.v1.multiprocess.transport.grpc_impl.client import (
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
    build_grpc_request_server,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


@dataclass
class _Calls:
    lookup: tuple[IPCCacheServerKey, int] | None = None
    allocation: tuple[int, str, list[BlockAllocationRecord]] | None = None


class _TestDeviceIPCWrapper(DeviceIPCWrapper):
    """Pickle-safe test wrapper for the shared custom codec."""

    def __init__(self) -> None:
        self.handle = b"handle"
        self.dtype = torch.float16
        self.shape = (2, 4)
        self.stride = (4, 1)
        self.storage_offset = 0
        self.device_uuid = "test-device"

    def to_tensor(self) -> torch.Tensor:
        """The codec test does not reconstruct a device tensor."""
        raise NotImplementedError


@pytest.fixture
def grpc_client() -> Iterator[tuple[GrpcMultiprocessClient, _Calls]]:
    calls = _Calls()

    class FakeModules:
        @request_handler(RequestType.LOOKUP, HandlerType.BLOCKING)
        def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
            calls.lookup = (key, tp_size)

        @request_handler(
            RequestType.STORE,
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
            RequestType.PREPARE_STORE,
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
            RequestType.PREPARE_RETRIEVE,
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

        @request_handler(RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT)
        def register_kv_cache_engine_driven_context(
            self, payload: RegisterEngineDrivenContextPayload
        ) -> RegisterEngineDrivenContextResponse:
            assert payload.num_physical_slots == 32
            return RegisterEngineDrivenContextResponse("shared-memory", 4096)

        @request_handler(RequestType.PING, HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return instance_id == 7

        @request_handler(RequestType.NOOP)
        def debug(self) -> str:
            return "ok"

        @request_handler(RequestType.REPORT_BLOCK_ALLOCATION, HandlerType.BLOCKING)
        def report_block_allocations(
            self,
            instance_id: int,
            model_name: str,
            records: list[BlockAllocationRecord],
        ) -> None:
            calls.allocation = (instance_id, model_name, records)

        @request_handler(RequestType.CB_UNIFIED_LOOKUP, HandlerType.BLOCKING)
        def cb_unified_lookup(
            self, key: IPCCacheServerKey, tp_size: int
        ) -> CBUnifiedLookupResult | None:
            assert key.model_name == "model"
            assert tp_size == 2
            return CBUnifiedLookupResult(
                prefix_coverage_tokens=16,
                non_prefix_segments=[CBMatchResult(0, 2, 4, 6, b"hash")],
            )

        @request_handler(RequestType.P2P_LOOKUP_AND_LOCK, HandlerType.BLOCKING)
        def p2p_lookup_and_lock(
            self,
            keys: list[ObjectKey],
            group_layout_descs: dict[int, MemoryLayoutDesc],
        ) -> int:
            assert keys[0].cache_salt == "tenant"
            assert group_layout_descs[0].shapes == [torch.Size([2, 4])]
            assert group_layout_descs[0].dtypes == [torch.float16]
            return 41

        @request_handler(RequestType.P2P_QUERY_LOOKUP_RESULTS, HandlerType.BLOCKING)
        def p2p_query_lookup_results(
            self, task_id: int
        ) -> list[TransferChannelAddress] | None:
            assert task_id == 41
            return [TransferChannelAddress(offset=8, size=16)]

    modules: Any = FakeModules()
    server = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=2,
        max_gpu_workers=1,
        grpc_server_workers=4,
    )
    server.add_modules([modules])
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
    assert {
        "LMCacheDrivenService",
        "EngineDrivenService",
        "LookupService",
        "QStoreService",
    }.issubset(bindings)
    assert "EngineService" not in bindings
    assert {method.name for _, method in iter_methods()} >= {
        "Store",
        "PrepareStore",
        "Lookup",
        "StoreQ",
    }
    registry = get_method_codec_registry()
    generated_methods = {method.full_name for _, method in iter_methods()}
    assert set(registry.by_full_name) == generated_methods

    lookup_codec = registry.by_full_name["lmcache.mp.LookupService.Lookup"]
    assert lookup_codec.request_type is RequestType.LOOKUP
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


def test_module_annotations_cover_and_match_generated_grpc_methods() -> None:
    """Every generated RPC resolves to one compatible module annotation."""
    module_types = (
        LookupModule,
        ManagementModule,
        P2PController,
        LMCacheDrivenTransferModule,
        EngineDrivenTransferModule,
        QStoreModule,
        BlendModule,
    )
    handlers = {
        registered.options.request_type: registered.handler
        for module_type in module_types
        for registered in iter_request_handlers(module_type)
    }
    registry = get_method_codec_registry()
    codecs = tuple(registry.by_full_name.values())

    assert set(handlers) == {codec.request_type for codec in codecs}
    for codec in codecs:
        codec.validate_handler(handlers[codec.request_type])


def test_grpc_imports_do_not_load_zmq_runtime() -> None:
    """The gRPC transport imports without loading the ZMQ runtime."""
    script = r"""
import importlib.abc
import sys

banned = (
    "lmcache.v1.multiprocess.mq",
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
        grpc_server_module,
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

    server = cast(Any, build_grpc_request_server(modules, config))

    assert server.args == ("grpc://127.0.0.1:6000", 2, 3, 7)
    assert server.modules is modules


def test_service_message_codec_registry_round_trips_custom_types() -> None:
    """Service-owned codecs handle only their registered protobuf types."""
    registry = get_message_codec_registry()
    common_pb2 = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.common_pb2"
    )
    p2p_service_pb2 = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.p2p_service_pb2"
    )

    wrapper = _TestDeviceIPCWrapper()
    wrapper_message = common_pb2.DeviceIpcWrapper()
    wrapper_codec = registry.find(wrapper_message.DESCRIPTOR, type(wrapper))
    assert wrapper_codec is not None
    wrapper_codec.writer(wrapper_message, wrapper)
    decoded_wrapper = wrapper_codec.reader(wrapper_message)
    assert type(decoded_wrapper) is _TestDeviceIPCWrapper
    assert decoded_wrapper == wrapper

    shape = torch.Size([2, 4])
    shape_message = p2p_service_pb2.TensorShape()
    shape_codec = registry.find(shape_message.DESCRIPTOR, torch.Size)
    assert shape_codec is not None
    shape_codec.writer(shape_message, shape)
    assert shape_codec.reader(shape_message) == shape


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
