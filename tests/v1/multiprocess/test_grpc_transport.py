# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the generated-service gRPC transport."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transport.grpc_impl.client import (
    GrpcMultiprocessClient,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    get_service_bindings,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services import (
    BlendServiceImpl,
    ControllerServiceImpl,
    DebugServiceImpl,
    EngineDrivenServiceImpl,
    LMCacheDrivenServiceImpl,
    LookupServiceImpl,
    ObservabilityServiceImpl,
    P2PServiceImpl,
    QStoreServiceImpl,
)


@dataclass
class _Calls:
    lookup: tuple[IPCCacheServerKey, int] | None = None
    allocation: tuple[int, str, list[BlockAllocationRecord]] | None = None


@pytest.fixture
def grpc_client() -> Iterator[tuple[GrpcMultiprocessClient, _Calls]]:
    calls = _Calls()

    class FakeModules:
        def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
            calls.lookup = (key, tp_size)

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

        def prepare_store(
            self, key: IPCCacheServerKey, instance_id: int
        ) -> PrepareStoreResponse:
            assert key.request_configs == {"blend": True}
            assert instance_id == 7
            return PrepareStoreResponse(
                context={"slots": [{"offset": 8}], "chunk_indices": [2]}
            )

        def register_kv_cache_engine_driven_context(
            self, payload: RegisterEngineDrivenContextPayload
        ) -> RegisterEngineDrivenContextResponse:
            assert payload.num_physical_slots == 32
            return RegisterEngineDrivenContextResponse("shared-memory", 4096)

        def ping(self, instance_id: int | None) -> bool:
            return instance_id == 7

        def debug(self) -> str:
            return "ok"

        def report_block_allocations(
            self,
            instance_id: int,
            model_name: str,
            records: list[BlockAllocationRecord],
        ) -> None:
            calls.allocation = (instance_id, model_name, records)

        def cb_unified_lookup(
            self, key: IPCCacheServerKey, tp_size: int
        ) -> CBUnifiedLookupResult | None:
            assert key.model_name == "model"
            assert tp_size == 2
            return CBUnifiedLookupResult(
                prefix_coverage_tokens=16,
                non_prefix_segments=[CBMatchResult(0, 2, 4, 6, b"hash")],
            )

        def p2p_lookup_and_lock(
            self,
            keys: list[ObjectKey],
            group_layout_descs: dict[int, MemoryLayoutDesc],
        ) -> int:
            assert keys[0].cache_salt == "tenant"
            assert group_layout_descs[0].shapes == [torch.Size([2, 4])]
            assert group_layout_descs[0].dtypes == [torch.float16]
            return 41

    modules: Any = FakeModules()
    server = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=2,
        max_gpu_workers=1,
    )
    server.add_service("LMCacheDrivenService", LMCacheDrivenServiceImpl(None, modules))
    server.add_service("EngineDrivenService", EngineDrivenServiceImpl(modules))
    server.add_service("LookupService", LookupServiceImpl(modules))
    server.add_service("QStoreService", QStoreServiceImpl(None))
    server.add_service("ControllerService", ControllerServiceImpl(modules))
    server.add_service("DebugService", DebugServiceImpl(modules))
    server.add_service("ObservabilityService", ObservabilityServiceImpl(modules))
    server.add_service("P2PService", P2PServiceImpl(modules))
    server.add_service("BlendService", BlendServiceImpl(modules))
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
    """RPC discovery follows generated services without a request-type enum."""
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
