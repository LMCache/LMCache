# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the generated-service gRPC transport."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
import subprocess
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
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
from lmcache.v1.multiprocess.rpc_messages import (
    CbUnifiedLookupRequest,
    CbUnifiedLookupResponse,
    EventIpcHandleResult,
    LookupRequest,
    LookupResponse,
    NoopRequest,
    NoopResponse,
    P2pLookupAndLockRequest,
    P2pLookupAndLockResponse,
    P2pQueryLookupResultsRequest,
    P2pQueryLookupResultsResponse,
    PingRequest,
    PingResponse,
    PrepareRetrieveRequest,
)
from lmcache.v1.multiprocess.rpc_messages import (
    PrepareRetrieveResponse as RpcPrepareRetrieveResponse,
)
from lmcache.v1.multiprocess.rpc_messages import (
    PrepareStoreRequest,
)
from lmcache.v1.multiprocess.rpc_messages import (
    PrepareStoreResponse as RpcPrepareStoreResponse,
)
from lmcache.v1.multiprocess.rpc_messages import (
    RegisterKvCacheEngineDrivenContextRequest,
    RegisterKvCacheEngineDrivenContextResponse,
    RegisterKvCacheRequest,
    ReportBlockAllocationRequest,
    ReportBlockAllocationResponse,
    StoreRequest,
    StoreResponse,
    deserialize_rpc_message,
    serialize_rpc_message,
)
from lmcache.v1.multiprocess.transport.grpc_impl.client import (
    GrpcMultiprocessClient,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    get_service_bindings,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


@dataclass
class _Calls:
    lookup: tuple[IPCCacheServerKey, int] | None = None
    allocation: tuple[int, str, list[BlockAllocationRecord]] | None = None


class _TestDeviceIPCWrapper(DeviceIPCWrapper):
    """Pickle-safe test wrapper for transport-neutral serialization."""

    def __init__(self) -> None:
        self.handle = b"handle"
        self.dtype = torch.float16
        self.shape = (2, 4)
        self.stride = (4, 1)
        self.storage_offset = 0
        self.device_uuid = "test-device"

    def to_tensor(self) -> torch.Tensor:
        """The adapter test does not reconstruct a device tensor."""
        raise NotImplementedError


@pytest.fixture
def grpc_client() -> Iterator[tuple[GrpcMultiprocessClient, _Calls]]:
    calls = _Calls()

    class FakeModules:
        @request_handler(RequestType.LOOKUP, HandlerType.BLOCKING)
        def lookup(self, request: LookupRequest) -> LookupResponse:
            calls.lookup = (request.key, request.tp_size)
            return LookupResponse()

        @request_handler(
            RequestType.STORE,
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def store(self, request: StoreRequest) -> StoreResponse:
            assert request.instance_id == 7
            assert request.gpu_block_ids == [[1, 2], [3]]
            assert request.event_ipc_handle == b"input-event"
            return StoreResponse(
                EventIpcHandleResult(b"output-event", request.key.model_name == "model")
            )

        @request_handler(
            RequestType.PREPARE_STORE,
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def prepare_store(
            self, request: PrepareStoreRequest
        ) -> RpcPrepareStoreResponse:
            assert request.key.request_configs == {"blend": True}
            assert request.instance_id == 7
            return RpcPrepareStoreResponse(
                context={"slots": [{"offset": 8}], "chunk_indices": [2]}
            )

        @request_handler(
            RequestType.PREPARE_RETRIEVE,
            HandlerType.BLOCKING,
            requires_client_affinity=True,
        )
        def prepare_retrieve(
            self, request: PrepareRetrieveRequest
        ) -> RpcPrepareRetrieveResponse:
            assert request.key.request_configs == {"blend": True}
            assert request.instance_id == 7
            return RpcPrepareRetrieveResponse(
                success=True,
                data=b"retrieved",
                context={"slot": 3},
            )

        @request_handler(RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT)
        def register_kv_cache_engine_driven_context(
            self, request: RegisterKvCacheEngineDrivenContextRequest
        ) -> RegisterKvCacheEngineDrivenContextResponse:
            assert request.num_physical_slots == 32
            return RegisterKvCacheEngineDrivenContextResponse("shared-memory", 4096)

        @request_handler(RequestType.PING, HandlerType.BLOCKING)
        def ping(self, request: PingRequest) -> PingResponse:
            return PingResponse(request.instance_id == 7)

        @request_handler(RequestType.NOOP)
        def debug(self, request: NoopRequest) -> NoopResponse:
            return NoopResponse("ok")

        @request_handler(RequestType.REPORT_BLOCK_ALLOCATION, HandlerType.BLOCKING)
        def report_block_allocations(
            self, request: ReportBlockAllocationRequest
        ) -> ReportBlockAllocationResponse:
            calls.allocation = (
                request.instance_id,
                request.model_name,
                request.records,
            )
            return ReportBlockAllocationResponse()

        @request_handler(RequestType.CB_UNIFIED_LOOKUP, HandlerType.BLOCKING)
        def cb_unified_lookup(
            self, request: CbUnifiedLookupRequest
        ) -> CbUnifiedLookupResponse:
            assert request.key.model_name == "model"
            assert request.tp_size == 2
            return CbUnifiedLookupResponse(
                CBUnifiedLookupResult(
                    prefix_coverage_tokens=16,
                    non_prefix_segments=[CBMatchResult(0, 2, 4, 6, b"hash")],
                )
            )

        @request_handler(RequestType.P2P_LOOKUP_AND_LOCK, HandlerType.BLOCKING)
        def p2p_lookup_and_lock(
            self, request: P2pLookupAndLockRequest
        ) -> P2pLookupAndLockResponse:
            assert request.keys[0].cache_salt == "tenant"
            assert request.group_layout_descs[0].shapes == [torch.Size([2, 4])]
            assert request.group_layout_descs[0].dtypes == [torch.float16]
            return P2pLookupAndLockResponse(41)

        @request_handler(RequestType.P2P_QUERY_LOOKUP_RESULTS, HandlerType.BLOCKING)
        def p2p_query_lookup_results(
            self, request: P2pQueryLookupResultsRequest
        ) -> P2pQueryLookupResultsResponse:
            assert request.task_id == 41
            return P2pQueryLookupResultsResponse(
                [TransferChannelAddress(offset=8, size=16)]
            )

    modules: Any = FakeModules()
    server = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=2,
        max_gpu_workers=1,
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
    """Every generated RPC has one Python contract derived by convention."""
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
    registry = get_method_registry()
    generated_methods = {method.full_name for _, method in iter_methods()}
    assert set(registry.by_full_name) == generated_methods
    for _, method in iter_methods():
        method_binding = registry.by_full_name[method.full_name]
        assert method_binding.python_request_class.__name__ == method.input_type.name
        assert method_binding.python_response_class.__name__ == method.output_type.name

    lookup_binding = registry.by_full_name["lmcache.mp.LookupService.Lookup"]
    assert lookup_binding.request_type is RequestType.LOOKUP
    assert lookup_binding.python_request_class is LookupRequest
    assert lookup_binding.python_response_class is LookupResponse

    store_binding = registry.by_full_name["lmcache.mp.LMCacheDrivenService.Store"]
    assert store_binding.python_request_class is StoreRequest
    assert store_binding.python_response_class is StoreResponse

    registration_binding = registry.by_full_name[
        "lmcache.mp.EngineDrivenService.RegisterKvCacheEngineDrivenContext"
    ]
    python_request = RegisterKvCacheEngineDrivenContextRequest(
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
    payload = serialize_rpc_message(python_request, type(python_request))
    assert deserialize_rpc_message(payload, type(python_request)) == python_request
    assert (
        registration_binding.python_response_class
        is RegisterKvCacheEngineDrivenContextResponse
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
    registry = get_method_registry()
    adapters = tuple(registry.by_full_name.values())

    assert set(handlers) == {adapter.request_type for adapter in adapters}
    for adapter in adapters:
        adapter.validate_handler(handlers[adapter.request_type])


def test_grpc_imports_do_not_load_zmq_runtime() -> None:
    """The gRPC transport imports without loading the ZMQ runtime."""
    script = r"""
import importlib.abc
import sys

banned = (
    "lmcache.v1.multiprocess.mq",
    "lmcache.v1.multiprocess.transport.zmq_impl",
    "lmcache.v1.multiprocess.transport.grpc_impl.message_conversion",
    "lmcache.v1.multiprocess.transport.grpc_impl.message_adapters",
)


class LegacyProtocolBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in banned):
            raise ImportError(f"gRPC imported legacy module: {fullname}")
        if fullname.endswith("_pb2_grpc"):
            raise ImportError(f"gRPC imported generated stub module: {fullname}")
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


def test_transport_neutral_serialization_round_trips_custom_types() -> None:
    """The shared Python message format handles non-primitive domain values."""
    wrapper = _TestDeviceIPCWrapper()
    registration = RegisterKvCacheRequest(
        instance_id=7,
        kv_cache=[wrapper],
        model_name="model",
        world_size=1,
        engine_type=EngineType.MOCK,
        layout_hints={},
        engine_group_infos=[],
    )
    decoded_registration = deserialize_rpc_message(
        serialize_rpc_message(registration, RegisterKvCacheRequest),
        RegisterKvCacheRequest,
    )
    decoded_wrapper = decoded_registration.kv_cache[0]
    assert type(decoded_wrapper) is _TestDeviceIPCWrapper
    assert decoded_wrapper.__dict__ == wrapper.__dict__

    p2p_request = P2pLookupAndLockRequest(
        keys=[ObjectKey(b"chunk", "model", 0)],
        group_layout_descs={0: MemoryLayoutDesc([torch.Size([2, 4])], [torch.float16])},
    )
    assert (
        deserialize_rpc_message(
            serialize_rpc_message(p2p_request, P2pLookupAndLockRequest),
            P2pLookupAndLockRequest,
        )
        == p2p_request
    )


def test_descriptor_derived_grpc_services_communicate_end_to_end(
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
