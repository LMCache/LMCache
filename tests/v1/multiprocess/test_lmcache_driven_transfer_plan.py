# SPDX-License-Identifier: Apache-2.0
"""Contract tests for shared-plan LMCache-driven transfer execution."""

# Standard
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Iterator, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupTransferMetadata,
    KVTransferMetadata,
    ObjectGroupPlan,
    ObjectGroupTransferMetadata,
    TransferPlanDirection,
    build_transfer_plan,
)


class _EventBackend:
    """Minimal event backend for executor handler contract tests."""

    def create_event(self, _device: torch.device) -> object:
        """Create an opaque completion event."""
        return object()

    def import_event(self, _handle: bytes, _device: torch.device) -> object:
        """Import an opaque producer event."""
        return object()

    def wait_event(self, _event: object, _stream: object) -> None:
        """Order the transfer stream after the producer."""

    def record_event(self, _event: object, _stream: object) -> None:
        """Record transfer completion."""

    def export_event(self, _event: object, _device: torch.device) -> bytes:
        """Export a stable opaque completion handle."""
        return b"event"


class _MemoryObject:
    """Small storage object with the executor's public size surface."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_size(self) -> int:
        """Return a deterministic object size."""
        return 1


class _CacheContext:
    """Record device bindings while exposing the transfer resource surface."""

    device = torch.device("cpu")
    stream = "transfer-stream"
    cupy_stream = "cupy-stream"
    max_batch_size = 2

    def __init__(self) -> None:
        self.staged_block_ids: list[list[list[int]]] = []

    def stage_block_ids(self, block_ids: list[list[int]]) -> list[torch.Tensor]:
        """Record and CPU-stage planned block IDs."""
        self.staged_block_ids.append(block_ids)
        return [torch.tensor(group, dtype=torch.int64) for group in block_ids]


class _StorageManager:
    """Provide prefetched objects for one retrieve handler invocation."""

    def __init__(self, memory_objs: list[_MemoryObject]) -> None:
        self._memory_objs = memory_objs

    def finish_write(self, _keys: list[object]) -> None:
        """No-op completion callback."""

    def finish_read_prefetched(self, _keys: list[object]) -> None:
        """No-op completion callback."""

    def reserve_write(
        self, keys: list[object], _layout: object, _mode: str
    ) -> dict[object, _MemoryObject]:
        """Reserve objects in original chunk order for a store."""
        return dict(zip(keys, self._memory_objs, strict=True))

    @contextmanager
    def read_prefetched_results(
        self, _keys: list[object]
    ) -> Iterator[list[_MemoryObject]]:
        """Yield all requested objects in original chunk order."""
        yield self._memory_objs


class _NoopDispatcher:
    """Avoid native callback setup in handler contract tests."""

    def register(self, _kind: str, _handler: object, payload_type: object) -> None:
        """Accept registration without a native thread."""

    def start(self) -> None:
        """Do not start a native thread."""


def _metadata() -> KVTransferMetadata:
    """Return two kernel groups that share an engine-group block-ID space."""
    # First Party
    import lmcache.c_ops as lmc_ops

    kernel_group_kwargs = {
        "engine_group_id": 3,
        "layer_indices": (0,),
        "blocks_per_chunk": 2,
        "slots_per_chunk_in_window": 4,
        "kv_size": 1,
        "num_layers": 1,
        "hidden_dim_size": 1,
        "slots_per_block": 1,
        "tokens_per_block": 2,
        "dtype": torch.float16,
        "engine_kv_format": lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
    }
    return KVTransferMetadata(
        num_chunks_in_sw=(2,),
        tokens_per_chunk=4,
        kernel_groups=(
            KernelGroupTransferMetadata(
                kernel_group_id=0,
                blocks_per_window=2,
                **kernel_group_kwargs,
            ),
            KernelGroupTransferMetadata(
                kernel_group_id=1,
                blocks_per_window=1,
                **kernel_group_kwargs,
            ),
        ),
        object_groups=(
            ObjectGroupTransferMetadata(
                object_group_id=0,
                kernel_group_ids=(1, 0),
                sw_size_chunks=2,
            ),
        ),
    )


def _module_context(storage_manager: _StorageManager) -> MPCacheServerContext:
    """Build the server context surface used by the retrieve handler."""
    return cast(
        MPCacheServerContext,
        SimpleNamespace(
            chunk_size=4,
            storage_manager=storage_manager,
            event_bus=SimpleNamespace(
                publish=lambda _event: None,
                publish_on_stream=lambda _stream, _event: None,
            ),
            resolve_obj_keys=lambda _key, _group_ids: [
                ["chunk-0", "chunk-1", "chunk-2"]
            ],
        ),
    )


def test_retrieve_binds_shared_plan_to_staging_and_matching_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve stages plan-ordered IDs and maps omitted chunks to objects."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    cache_context = _CacheContext()
    memory_objs = [_MemoryObject(f"chunk-{idx}") for idx in range(3)]
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", _NoopDispatcher
    )
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
        _module_context(_StorageManager(memory_objs))
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cache_context,  # type: ignore[arg-type]
        model_name="model",
        world_size=1,
        transfer_metadata=_metadata(),
        event_backend=_EventBackend(),  # type: ignore[arg-type]
    )
    captured: list[tuple[ObjectGroupPlan, list[_MemoryObject]]] = []

    def capture_transfer(
        _cache_context: object,
        _transfer_metadata: object,
        object_group_plan: ObjectGroupPlan,
        _block_ids_gpu: object,
        planned_memory_objs: list[_MemoryObject],
        *,
        batch_size: int,
        direction: object,
    ) -> None:
        """Capture the handler's resource-bound shared-plan execution."""
        captured.append((object_group_plan, planned_memory_objs))

    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "device", lambda _device: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "submit_callback_to_stream", lambda *_args: None
    )
    monkeypatch.setattr(
        module, "get_and_touch_context_entry", lambda _instance_id: entry
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "transfer_kv_per_object_group", capture_transfer
    )

    assert module.retrieve(
        SimpleNamespace(request_id="request", cache_salt=""),
        1,
        [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
        b"producer",
        skip_first_n_tokens=6,
    ) == (b"event", True)

    assert cache_context.staged_block_ids == [[[13, 15], [12, 13, 14, 15]]]
    plan, planned_memory_objs = captured[0]
    assert plan.chunk_indices == (1, 2)
    assert [group.kernel_group_id for group in plan.kernel_groups] == [1, 0]
    assert [group.skip_first_n_blocks for group in plan.kernel_groups] == [0, 1]
    assert planned_memory_objs == memory_objs[1:]


def test_store_binds_shared_plan_to_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store stages the shared plan without retrieve-only chunk omission."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    cache_context = _CacheContext()
    memory_objs = [_MemoryObject(f"chunk-{idx}") for idx in range(3)]
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", _NoopDispatcher
    )
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
        _module_context(_StorageManager(memory_objs))
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cache_context,  # type: ignore[arg-type]
        model_name="model",
        world_size=1,
        transfer_metadata=_metadata(),
        event_backend=_EventBackend(),  # type: ignore[arg-type]
    )
    captured: list[list[_MemoryObject]] = []

    def capture_transfer(
        _cache_context: object,
        _transfer_metadata: object,
        _object_group_plan: object,
        _block_ids_gpu: object,
        planned_memory_objs: list[_MemoryObject],
        *,
        batch_size: int,
        direction: object,
    ) -> None:
        """Capture the handler's plan-to-resource binding."""
        captured.append(planned_memory_objs)

    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "device", lambda _device: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "submit_callback_to_stream", lambda *_args: None
    )
    monkeypatch.setattr(
        module, "get_and_touch_context_entry", lambda _instance_id: entry
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "transfer_kv_per_object_group", capture_transfer
    )

    assert module.store(
        SimpleNamespace(request_id="request", cache_salt=""),
        1,
        [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
        b"producer",
    ) == (b"event", True)

    assert cache_context.staged_block_ids == [[[11, 13, 15], [10, 11, 12, 13, 14, 15]]]
    assert captured == [memory_objs]


def test_retrieve_rejects_conflicting_repeated_engine_group_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated kernel-group request IDs must agree for one engine group."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    cache_context = _CacheContext()
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", _NoopDispatcher
    )
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
        _module_context(_StorageManager([_MemoryObject("unused")]))
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cache_context,  # type: ignore[arg-type]
        model_name="model",
        world_size=1,
        transfer_metadata=_metadata(),
        event_backend=_EventBackend(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "device", lambda _device: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        module, "get_and_touch_context_entry", lambda _instance_id: entry
    )

    assert module.retrieve(
        SimpleNamespace(request_id="request", cache_salt=""),
        1,
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
        b"producer",
    ) == (b"event", False)
    assert cache_context.staged_block_ids == []


def test_fallback_executor_uses_plan_block_order_and_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback kernel launches consume plan IDs and planned skip geometry."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    import lmcache.c_ops as lmc_ops

    metadata = _metadata()
    plan = build_transfer_plan(
        metadata,
        {3: [10, 11, 12, 13, 14, 15]},
        num_chunks=3,
        direction=TransferPlanDirection.RETRIEVE,
        skip_first_n_tokens=6,
    ).object_groups[0]
    cache_context = _CacheContext()
    staged_block_ids = cache_context.stage_block_ids(
        [list(group.block_ids) for group in plan.kernel_groups]
    )
    launches: list[tuple[torch.Tensor, int]] = []

    class _Buffer:
        """Expose the staging buffer pointer surface."""

        def data_ptr(self) -> int:
            """Return a stable fake pointer."""
            return 1

    monkeypatch.setattr(
        lmcache_driven_transfer, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", False
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "lmcache_memcpy_async_h2d", lambda *_args: None
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.lmc_ops,
        "multi_layer_block_kv_transfer",
        lambda _paged, _buffers, block_ids, *_args: launches.append(
            (block_ids, _args[-1])
        ),
    )
    monkeypatch.setattr(
        cache_context,
        "get_temp_object_group_buffer",
        lambda _slot, _object_group_id: _Buffer(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_temp_kernel_group_buffer",
        lambda _slot, _kernel_group_id: _Buffer(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_kernel_group_kv_pointers",
        lambda _kernel_group_id: object(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_shape_desc",
        lambda _kernel_group_id: object(),
        raising=False,
    )

    lmcache_driven_transfer.transfer_kv_per_object_group(
        cache_context,  # type: ignore[arg-type]
        metadata,
        plan,
        staged_block_ids,
        cast(
            list[MemoryObj | None],
            [_MemoryObject("chunk-1"), _MemoryObject("chunk-2")],
        ),
        batch_size=2,
        direction=lmc_ops.TransferDirection.H2D,
    )

    assert [(ids.tolist(), skip) for ids, skip in launches] == [
        ([13, 15], 0),
        ([12, 13, 14, 15], 1),
    ]


def test_native_executor_uses_plan_block_order_and_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native batching consumes the same plan ordering and skip geometry."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    import lmcache.c_ops as lmc_ops

    metadata = _metadata()
    plan = build_transfer_plan(
        metadata,
        {3: [10, 11, 12, 13, 14, 15]},
        num_chunks=3,
        direction=TransferPlanDirection.RETRIEVE,
        skip_first_n_tokens=6,
    ).object_groups[0]
    cache_context = _CacheContext()
    staged_block_ids = cache_context.stage_block_ids(
        [list(group.block_ids) for group in plan.kernel_groups]
    )

    class _Buffer:
        """Expose the pointer interface required by native argument binding."""

        def data_ptr(self) -> int:
            """Return a stable fake pointer."""
            return 1

    class _KernelGroupSpec:
        """Capture native kernel-group binding arguments."""

        def __init__(self, *args: object) -> None:
            self.args = args

    class _LaunchVar:
        """Capture one native launch binding."""

        def __init__(self, *args: int) -> None:
            self.args = args

    class _BatchStep:
        """Capture one native batch."""

        def __init__(self, staging: object, launches: list[_LaunchVar]) -> None:
            self.staging = staging
            self.launches = launches

    executed_steps: list[list[_BatchStep]] = []
    monkeypatch.setattr(
        lmcache_driven_transfer, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", True
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "build_staging_copies", lambda *_args: object()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.lmc_ops, "KernelGroupSpec", _KernelGroupSpec
    )
    monkeypatch.setattr(lmcache_driven_transfer.lmc_ops, "LaunchVar", _LaunchVar)
    monkeypatch.setattr(lmcache_driven_transfer.lmc_ops, "BatchStep", _BatchStep)
    monkeypatch.setattr(
        lmcache_driven_transfer.lmc_ops,
        "execute_object_group_transfer",
        lambda _direction, _device, _pin_chunk_size, _specs, steps: (
            executed_steps.append(steps)
        ),
    )
    monkeypatch.setattr(
        cache_context,
        "get_temp_object_group_buffer",
        lambda _slot, _object_group_id: _Buffer(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_temp_kernel_group_buffer",
        lambda _slot, _kernel_group_id: _Buffer(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_kernel_group_kv_pointers",
        lambda _kernel_group_id: _Buffer(),
        raising=False,
    )
    monkeypatch.setattr(
        cache_context,
        "get_shape_desc",
        lambda _kernel_group_id: object(),
        raising=False,
    )

    lmcache_driven_transfer.transfer_kv_per_object_group(
        cache_context,  # type: ignore[arg-type]
        metadata,
        plan,
        staged_block_ids,
        cast(
            list[MemoryObj | None],
            [_MemoryObject("chunk-1"), _MemoryObject("chunk-2")],
        ),
        batch_size=2,
        direction=lmc_ops.TransferDirection.H2D,
    )

    assert [launch.args for launch in executed_steps[0][0].launches] == [
        (0, 0, 2, 2, 0),
        (1, 0, 4, 2, 1),
    ]
