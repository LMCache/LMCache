# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.dax_backend import DaxBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


@pytest.fixture
def loop_in_thread() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def disable_direct_gpu_ready(monkeypatch) -> None:
    monkeypatch.setattr(DaxBackend, "_ensure_direct_gpu_ready", lambda self: None)


def _create_metadata(
    chunk_size: int = 16,
    world_size: int = 1,
    role: str = "worker",
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, chunk_size, 2, 8),
        role=role,
    )


def _create_multi_group_metadata(chunk_size: int = 16) -> LMCacheMetadata:
    metadata = _create_metadata(chunk_size=chunk_size)
    metadata.kv_layer_groups_manager = KVLayerGroupsManager(
        kv_layer_groups=[
            KVLayerGroupInfo(
                layer_names=["layer0"],
                layer_indices=[0],
                shape=torch.Size([2, 1, chunk_size, 1, 8]),
                dtype=torch.bfloat16,
            ),
            KVLayerGroupInfo(
                layer_names=["layer1"],
                layer_indices=[1],
                shape=torch.Size([2, 1, chunk_size, 1, 16]),
                dtype=torch.bfloat16,
            ),
        ]
    )
    return metadata


def _create_config(
    *,
    chunk_size: int = 16,
    local_cpu: bool = True,
    max_local_cpu_size: float = 0.1,
    extra_config: dict | None = None,
    storage_plugins: list[str] | None = None,
) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=local_cpu,
        max_local_cpu_size=max_local_cpu_size,
        lmcache_instance_id="test_dax_backend",
    )
    if extra_config is not None:
        config.extra_config = extra_config
    if storage_plugins is not None:
        config.storage_plugins = storage_plugins
    return config


def _assert_no_dax_private_attrs(memory_obj) -> None:
    for attr in (
        "_lmcache_dax_handle",
        "_lmcache_dax_released",
        "_lmcache_dax_backed",
        "_lmcache_dax_offset",
    ):
        assert not hasattr(memory_obj, attr)


def test_dax_backend_tiered_roundtrip(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 1, torch.bfloat16)
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(5)

            futs = backend.batched_submit_put_task([key], [obj])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)

            out = backend.get_blocking(key)
            assert out is not None
            assert out.tensor is not None
            assert torch.equal(out.tensor, obj.tensor)
            out.ref_count_down()
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_rejects_tp_gt_1(loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 4 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16, world_size=2)

        with pytest.raises(ValueError, match="only supports TP=1"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=loop_in_thread,
                dst_device="cpu",
            )


def test_dax_backend_primary_requires_cuda_dst_device() -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="requires a CUDA dst_device"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cpu",
            )


@pytest.mark.parametrize("mode", ["primary", "tiered"])
def test_dax_backend_rejects_multi_group_metadata_at_init(
    mode: str,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=(mode == "tiered"),
            max_local_cpu_size=0.1 if mode == "tiered" else 0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": mode,
                "dax.arena_size_gb": 8 / 1024,
            },
        )
        metadata = _create_multi_group_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="single-group KV layout"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cuda:0" if mode == "primary" else "cpu",
            )


def test_storage_manager_can_use_dax_plugin_as_allocator(
    disable_direct_gpu_ready,
    monkeypatch,
) -> None:
    del disable_direct_gpu_ready
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.is_cuda_worker",
        lambda metadata: True,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.storage_manager.torch.cuda.Stream",
        lambda: object(),
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.is_cuda_worker",
        lambda metadata: True,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.torch.cuda.current_device",
        lambda: 0,
    )
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            storage_plugins=["dax"],
            extra_config={
                "storage_plugin.dax.module_path": (
                    "lmcache.v1.storage_backend.plugins.dax_backend"
                ),
                "storage_plugin.dax.class_name": "DaxBackend",
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 8 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16, role="worker")
        event_manager = EventManager()

        manager = StorageManager(
            config=config,
            metadata=metadata,
            event_manager=event_manager,
        )
        try:
            assert manager.allocator_backend is not None
            assert manager.allocator_backend.__class__.__name__ == "DaxBackend"
            assert isinstance(manager.allocator_backend, AllocatorBackendInterface)
            assert manager.memcheck()
            assert manager.allocator_backend.get_memory_allocator().memcheck()
        finally:
            manager.close()


def test_dax_backend_primary_view_survives_remove_until_release(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            key1 = CacheEngineKey("test_model", 1, 0, 601, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 602, torch.bfloat16)
            obj1 = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj1 is not None
            assert obj1.tensor is not None
            obj1.tensor.fill_(1)
            backend.batched_submit_put_task([key1], [obj1])
            obj1.ref_count_down()

            borrowed = backend.get_blocking(key1)
            assert borrowed is not None
            assert borrowed.tensor is not None
            assert torch.all(borrowed.tensor == 1)

            assert backend.remove(key1)

            obj2 = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj2 is not None
            assert obj2.tensor is not None
            obj2.tensor.fill_(9)
            backend.batched_submit_put_task([key2], [obj2])
            obj2.ref_count_down()

            assert torch.all(borrowed.tensor == 1)

            borrowed.ref_count_down()
            assert backend.remove(key2)

            recycled = backend.batched_allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                batch_size=2,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert recycled is not None
            for memory_obj in recycled:
                memory_obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_allocation_release_reuses_capacity(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            for _ in range(6):
                obj = backend.allocate(
                    torch.Size([2, 16, 8]),
                    torch.bfloat16,
                    fmt=MemoryFormat.KV_T2D,
                    eviction=False,
                )
                assert obj is not None
                obj.ref_count_down()

            objs = backend.batched_allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                batch_size=2,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert objs is not None
            for obj in objs:
                obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_keeps_memory_obj_tracking_internal(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 603, torch.bfloat16)
            obj = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(2)
            _assert_no_dax_private_attrs(obj)

            backend.batched_submit_put_task([key], [obj])

            borrowed = backend.get_blocking(key)
            assert borrowed is not None
            _assert_no_dax_private_attrs(borrowed)

            borrowed.ref_count_down()
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_direct_commit_put_and_release_do_not_deadlock(
    disable_direct_gpu_ready,
    monkeypatch,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        lookup_entered = threading.Event()
        allow_lookup = threading.Event()
        errors: dict[str, BaseException] = {}
        original = backend._get_memory_obj_state_locked

        def _blocking_get_state(memory_obj):
            lookup_entered.set()
            assert allow_lookup.wait(timeout=1)
            return original(memory_obj)

        monkeypatch.setattr(backend, "_get_memory_obj_state_locked", _blocking_get_state)

        try:
            key = CacheEngineKey("test_model", 1, 0, 604, torch.bfloat16)
            obj = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(11)

            def _put() -> None:
                try:
                    backend.batched_submit_put_task([key], [obj])
                except BaseException as e:
                    errors["put"] = e

            def _release() -> None:
                try:
                    obj.ref_count_down()
                except BaseException as e:
                    errors["release"] = e

            put_thread = threading.Thread(target=_put)
            release_thread = threading.Thread(target=_release)

            put_thread.start()
            assert lookup_entered.wait(timeout=1)

            release_thread.start()
            allow_lookup.set()

            put_thread.join(timeout=1)
            release_thread.join(timeout=1)

            assert not put_thread.is_alive()
            assert not release_thread.is_alive()
            assert "put" not in errors
            assert "release" not in errors

            out = backend.get_blocking(key)
            assert out is not None
            assert out.tensor is not None
            assert torch.all(out.tensor == 11)
            out.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_rejects_multi_tensor_allocate(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            rejected = backend.allocate(
                [torch.Size([2, 8, 8]), torch.Size([2, 8, 8])],
                [torch.bfloat16, torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert rejected is None

            valid = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert valid is not None
            valid.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_failed_init_does_not_leak_fds() -> None:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("/proc/self/fd is not available on this platform")

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        fd_before = len(os.listdir("/proc/self/fd"))
        for _ in range(3):
            local_cpu = LocalCPUBackend(
                config=config,
                metadata=metadata,
                dst_device="cpu",
                memory_allocator=AdHocMemoryAllocator(device="cpu"),
            )
            with pytest.raises(RuntimeError, match="exceeds device capacity"):
                DaxBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=None,
                    dst_device="cpu",
                )
        fd_after = len(os.listdir("/proc/self/fd"))
        assert fd_after == fd_before


def test_dax_backend_oversized_put_skips_without_indexing_or_leaking_slots(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            oversized_key = CacheEngineKey("test_model", 1, 0, 704, torch.bfloat16)
            oversized = alloc.allocate(
                [torch.Size([2, 1025, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert oversized is not None
            assert oversized.get_size() > backend.slot_bytes

            backend.batched_submit_put_task([oversized_key], [oversized])
            assert not backend.contains(oversized_key)
            assert backend.get_blocking(oversized_key) is None
            oversized.ref_count_down()

            reserved = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert reserved is not None
            direct_key = CacheEngineKey("test_model", 1, 0, 705, torch.bfloat16)

            with monkeypatch.context() as local_patch:
                local_patch.setattr(
                    type(reserved),
                    "get_size",
                    lambda self: backend.slot_bytes + 1,
                )
                backend.batched_submit_put_task([direct_key], [reserved])

            assert not backend.contains(direct_key)
            reclaimed = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert reclaimed is None
            reserved.ref_count_down()
            recycled = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert recycled is not None
            recycled.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_duplicate_direct_commit_reject_keeps_reserved_slot_until_release(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 64 / (1024 * 1024),  # two slots
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            committed_key = CacheEngineKey("test_model", 1, 0, 800, torch.bfloat16)
            initial = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert initial is not None
            backend.batched_submit_put_task([committed_key], [initial])
            initial.ref_count_down()
            assert backend.contains(committed_key)

            reserved = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert reserved is not None

            backend.batched_submit_put_task([committed_key], [reserved])
            assert backend.contains(committed_key)

            reclaimed = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert reclaimed is None

            reserved.ref_count_down()
            recycled = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert recycled is not None
            recycled.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_put_rejects_mismatched_key_and_obj_lengths(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 64 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None

            with pytest.raises(ValueError, match="same length"):
                backend.batched_submit_put_task(
                    [
                        CacheEngineKey("test_model", 1, 0, 801, torch.bfloat16),
                        CacheEngineKey("test_model", 1, 0, 802, torch.bfloat16),
                    ],
                    [obj],
                )
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_multi_tensor_put_skips_without_indexing_or_leaking_slots(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            multi_key = CacheEngineKey("test_model", 1, 0, 706, torch.bfloat16)
            multi = alloc.allocate(
                [torch.Size([2, 128, 8]), torch.Size([2, 128, 8])],
                [torch.bfloat16, torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert multi is not None

            backend.batched_submit_put_task([multi_key], [multi])
            assert not backend.contains(multi_key)
            assert backend.get_blocking(multi_key) is None
            multi.ref_count_down()

            reclaimed = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert reclaimed is not None
            reclaimed.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_tiered_get_blocking_releases_lock_during_read(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key1 = CacheEngineKey("test_model", 1, 0, 701, torch.bfloat16)
        key2 = CacheEngineKey("test_model", 1, 0, 702, torch.bfloat16)
        read_started = threading.Event()
        allow_read = threading.Event()
        remove_finished = threading.Event()
        reader_out: dict[str, object] = {}
        remove_out: dict[str, object] = {}
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            for key, fill_value in ((key1, 3), (key2, 4)):
                obj = alloc.allocate(
                    [torch.Size([2, 16, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(fill_value)
                backend.batched_submit_put_task([key], [obj])
                obj.ref_count_down()

            def _reader() -> None:
                reader_out["value"] = backend.get_blocking(key1)

            def _remover() -> None:
                remove_out["value"] = backend.remove(key2)
                remove_finished.set()

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            remover = threading.Thread(target=_remover)
            remover.start()
            assert remove_finished.wait(timeout=0.2)
            assert remove_out["value"] is True

            allow_read.set()
            reader.join(timeout=1)
            remover.join(timeout=1)
            assert not reader.is_alive()
            assert not remover.is_alive()

            result = reader_out["value"]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 3)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_tiered_remove_during_read_defers_slot_reclaim(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 32 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        key = CacheEngineKey("test_model", 1, 0, 703, torch.bfloat16)
        read_started = threading.Event()
        allow_read = threading.Event()
        reader_out: dict[str, object] = {}
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            def _reader() -> None:
                reader_out["value"] = backend.get_blocking(key)

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            assert backend.remove(key)
            blocked = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert blocked is None

            allow_read.set()
            reader.join(timeout=1)
            assert not reader.is_alive()

            result = reader_out["value"]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 7)
            result.ref_count_down()

            recycled = backend.allocate(
                torch.Size([2, 256, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert recycled is not None
            recycled.ref_count_down()
            assert backend.get_blocking(key) is None
        finally:
            backend.close()


def test_dax_backend_tiered_get_read_failure_releases_cpu_memory_obj(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 32 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        allocated_on_read: list[MemoryObj] = []
        original_allocate = local_cpu.allocate

        def _tracking_allocate(
            shape: torch.Size,
            dtype: torch.dtype,
            fmt: MemoryFormat,
        ) -> MemoryObj:
            memory_obj = original_allocate(shape, dtype, fmt)
            assert memory_obj is not None
            allocated_on_read.append(memory_obj)
            return memory_obj

        def _failing_do_read(
            self,
            offset: int,
            memory_obj: MemoryObj,
            size: int,
        ) -> None:
            del self, offset, memory_obj, size
            raise RuntimeError("simulated read failure")

        monkeypatch.setattr(local_cpu, "allocate", _tracking_allocate)
        monkeypatch.setattr(DaxBackend, "_do_read", _failing_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 707, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            with pytest.raises(RuntimeError, match="simulated read failure"):
                backend.get_blocking(key)

            assert len(allocated_on_read) == 1
            assert allocated_on_read[0].get_ref_count() == 0
            assert backend.contains(key)
        finally:
            backend.close()


def test_dax_backend_allocator_exhaustion_triggers_eviction(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 64 / (1024 * 1024),  # ~2 slots for test kv shape
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            keys = [
                CacheEngineKey("test_model", 1, 0, 101, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 102, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 103, torch.bfloat16),
            ]

            for i, key in enumerate(keys):
                obj = alloc.allocate(
                    [torch.Size([2, 256, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(i + 1)
                futs = backend.batched_submit_put_task([key], [obj])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
                obj.ref_count_down()

            assert backend.get_blocking(keys[0]) is None
            out1 = backend.get_blocking(keys[1])
            assert out1 is not None
            out1.ref_count_down()
            out2 = backend.get_blocking(keys[2])
            assert out2 is not None
            out2.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_pinned_key_is_not_evicted(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 64 / (1024 * 1024),
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            keys = [
                CacheEngineKey("test_model", 1, 0, 201, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 202, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 203, torch.bfloat16),
            ]

            for key in keys[:2]:
                obj = alloc.allocate(
                    [torch.Size([2, 256, 8])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                futs = backend.batched_submit_put_task([key], [obj])
                if futs:
                    for fut in futs:
                        fut.result(timeout=5)
                obj.ref_count_down()

            assert backend.pin(keys[0])

            obj3 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj3 is not None
            futs = backend.batched_submit_put_task([keys[2]], [obj3])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj3.ref_count_down()

            out0 = backend.get_blocking(keys[0])
            assert out0 is not None
            out0.ref_count_down()
            assert backend.get_blocking(keys[1]) is None
            out2 = backend.get_blocking(keys[2])
            assert out2 is not None
            out2.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_remove_inflight_reclaims_slot(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(1024 * 1024)

        config = _create_config(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 32 / (1024 * 1024),  # one slot
            },
        )
        metadata = _create_metadata(chunk_size=256)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 301, torch.bfloat16)
            obj1 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj1 is not None
            backend.batched_submit_put_task([key1], [obj1])
            obj1.ref_count_down()

            assert backend.remove(key1)
            assert backend.get_blocking(key1) is None

            key2 = CacheEngineKey("test_model", 1, 0, 302, torch.bfloat16)
            obj2 = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj2 is not None
            backend.batched_submit_put_task([key2], [obj2])
            obj2.ref_count_down()
            out = backend.get_blocking(key2)
            assert out is not None
            out.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_multithread_put_get_smoke(memory_allocator, loop_in_thread):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        def _worker(i: int) -> None:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 400 + i, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(i)
            futs = backend.batched_submit_put_task([key], [obj])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj.ref_count_down()
            out = backend.get_blocking(key)
            assert out is not None
            assert out.tensor is not None
            assert torch.equal(out.tensor, obj.tensor)
            out.ref_count_down()

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_worker, i) for i in range(20)]
                for fut in futures:
                    fut.result(timeout=10)
        finally:
            backend.close()


def test_dax_backend_sync_close_waits_for_active_put(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        write_started = threading.Event()
        allow_write = threading.Event()
        close_returned = threading.Event()
        observed: dict[str, object] = {}
        writer_exc: dict[str, BaseException] = {}
        original_do_write = DaxBackend._do_write

        def _blocking_do_write(self, offset, memory_obj, size) -> None:
            write_started.set()
            assert allow_write.wait(timeout=1)
            observed["mmap_is_none"] = self._mmap_obj is None
            observed["base_ptr"] = self._base_ptr
            original_do_write(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_write", _blocking_do_write)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            key = CacheEngineKey("test_model", 1, 0, 410, torch.bfloat16)

            def _writer() -> None:
                try:
                    backend.batched_submit_put_task([key], [obj])
                except BaseException as e:
                    writer_exc["error"] = e

            writer = threading.Thread(target=_writer)
            closer = threading.Thread(
                target=lambda: (backend.close(), close_returned.set())
            )

            writer.start()
            assert write_started.wait(timeout=1)
            closer.start()
            time.sleep(0.05)
            assert not close_returned.is_set()

            allow_write.set()
            writer.join(timeout=1)
            closer.join(timeout=1)

            assert not writer.is_alive()
            assert not closer.is_alive()
            assert "error" not in writer_exc
            assert close_returned.is_set()
            assert observed["mmap_is_none"] is False
            assert observed["base_ptr"] != 0
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_sync_close_waits_for_active_tiered_get(
    memory_allocator,
    loop_in_thread,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        read_started = threading.Event()
        allow_read = threading.Event()
        close_returned = threading.Event()
        reader_out: dict[str, object] = {}
        original_do_read = DaxBackend._do_read

        def _blocking_do_read(self, offset, memory_obj, size) -> None:
            read_started.set()
            assert allow_read.wait(timeout=1)
            original_do_read(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_read", _blocking_do_read)

        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(6)
            key = CacheEngineKey("test_model", 1, 0, 412, torch.bfloat16)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            def _reader() -> None:
                reader_out["value"] = backend.get_blocking(key)

            reader = threading.Thread(target=_reader)
            closer = threading.Thread(
                target=lambda: (backend.close(), close_returned.set())
            )

            reader.start()
            assert read_started.wait(timeout=1)
            closer.start()
            time.sleep(0.05)

            assert not close_returned.is_set()
            assert backend._mmap_obj is not None
            assert backend._base_ptr != 0

            allow_read.set()
            reader.join(timeout=1)
            closer.join(timeout=1)

            assert not reader.is_alive()
            assert not closer.is_alive()
            assert close_returned.is_set()

            result = reader_out["value"]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 6)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_close_is_best_effort_with_borrowed_view(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 413, torch.bfloat16)
            obj = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(8)
            backend.batched_submit_put_task([key], [obj])
            obj.ref_count_down()

            borrowed = backend.get_blocking(key)
            assert borrowed is not None
            assert borrowed.tensor is not None
            assert torch.all(borrowed.tensor == 8)

            start = time.monotonic()
            backend.close()
            elapsed = time.monotonic() - start

            assert elapsed < 0.5
            assert torch.all(borrowed.tensor == 8)
            borrowed.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_close_is_best_effort_with_reserved_allocation(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            obj = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(8)

            start = time.monotonic()
            backend.close()
            elapsed = time.monotonic() - start

            assert elapsed < 0.5
            assert torch.all(obj.tensor == 8)
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_late_release_after_close_is_safe(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            obj = backend.allocate(
                torch.Size([2, 16, 8]),
                torch.bfloat16,
                fmt=MemoryFormat.KV_T2D,
                eviction=False,
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(8)

            backend.close()

            assert torch.all(obj.tensor == 8)
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_primary_close_waits_for_inflight_allocate_but_not_release(
    disable_direct_gpu_ready,
    monkeypatch,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        entered = threading.Event()
        allow_finish = threading.Event()
        close_returned = threading.Event()
        result: dict[str, MemoryObj | None] = {}
        errors: dict[str, BaseException] = {}
        original = backend._set_memory_obj_handle

        def _delayed_set(memory_obj, handle, lease) -> None:
            entered.set()
            assert allow_finish.wait(timeout=1)
            original(memory_obj, handle, lease)

        monkeypatch.setattr(backend, "_set_memory_obj_handle", _delayed_set)

        def _allocate() -> None:
            try:
                result["obj"] = backend.allocate(
                    torch.Size([2, 16, 8]),
                    torch.bfloat16,
                    fmt=MemoryFormat.KV_T2D,
                    eviction=False,
                )
            except BaseException as e:
                errors["allocate"] = e

        allocator = threading.Thread(target=_allocate)
        allocator.start()
        assert entered.wait(timeout=1)

        closer = threading.Thread(target=lambda: (backend.close(), close_returned.set()))
        closer.start()
        time.sleep(0.05)
        assert not close_returned.is_set()

        allow_finish.set()
        allocator.join(timeout=1)
        closer.join(timeout=1)

        assert not allocator.is_alive()
        assert not closer.is_alive()
        assert "allocate" not in errors

        obj = result["obj"]
        assert obj is not None
        assert close_returned.is_set()
        assert obj.tensor is not None
        obj.tensor.fill_(3)
        assert torch.all(obj.tensor == 3)
        obj.ref_count_down()


def test_dax_backend_close_rejects_new_ops_after_shutdown(
    disable_direct_gpu_ready,
) -> None:
    del disable_direct_gpu_ready
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(4096)

        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=None,
            loop=None,
            dst_device="cuda:0",
        )

        try:
            backend.close()
            assert (
                backend.allocate(
                    torch.Size([2, 16, 8]),
                    torch.bfloat16,
                    fmt=MemoryFormat.KV_T2D,
                    eviction=False,
                )
                is None
            )
            assert backend.get_blocking(
                CacheEngineKey("test_model", 1, 0, 999, torch.bfloat16)
            ) is None
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
            with pytest.raises(RuntimeError, match="closing"):
                backend.batched_submit_put_task(
                    [CacheEngineKey("test_model", 1, 0, 1000, torch.bfloat16)],
                    [obj],
                )
            obj.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_async_close_waits_for_active_put(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
                "dax.async_put": True,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        close_returned = threading.Event()
        key = CacheEngineKey("test_model", 1, 0, 411, torch.bfloat16)

        try:
            assert backend._begin_put_task(key)
            closer = threading.Thread(
                target=lambda: (backend.close(), close_returned.set())
            )
            closer.start()
            time.sleep(0.05)
            assert not close_returned.is_set()
            backend._finish_put_task(key)
            closer.join(timeout=2)

            assert not closer.is_alive()
            assert close_returned.is_set()
        finally:
            backend.close()


def test_dax_backend_async_close_waits_for_put_completion(
    memory_allocator,
    loop_in_thread,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(16 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "tiered",
                "dax.arena_size_gb": 16 / 1024,
                "dax.async_put": True,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        gate = Future()

        def _finish_future() -> None:
            time.sleep(0.05)
            gate.set_result(None)

        releaser = threading.Thread(target=_finish_future, daemon=True)

        try:
            with backend._state_lock:
                backend._async_futures.add(gate)
            releaser.start()
            start = time.monotonic()
            backend.close()
            elapsed = time.monotonic() - start
            assert gate.done()
            assert elapsed >= 0.04
        finally:
            backend.close()
            releaser.join(timeout=1)


def test_dax_backend_direct_init_failure_fail_fast(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        def _raise_cudart(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("cudart not available")

        monkeypatch.setattr(
            "lmcache.v1.storage_backend.plugins.dax_backend.torch.cuda.cudart",
            _raise_cudart,
        )
        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 8 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        with pytest.raises(RuntimeError, match="DAX direct GPU copy failed"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cuda:0",
            )


def test_dax_backend_direct_init_failure_releases_open_arena_resources(
    monkeypatch,
) -> None:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("/proc/self/fd is not available on this platform")

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        def _raise_cudart(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("cudart not available")

        monkeypatch.setattr(
            "lmcache.v1.storage_backend.plugins.dax_backend.torch.cuda.cudart",
            _raise_cudart,
        )
        config = _create_config(
            chunk_size=16,
            local_cpu=False,
            max_local_cpu_size=0.0,
            extra_config={
                "dax.device_path": dev_path,
                "dax.mode": "primary",
                "dax.arena_size_gb": 8 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        fd_before = len(os.listdir("/proc/self/fd"))
        for _ in range(3):
            with pytest.raises(RuntimeError, match="DAX direct GPU copy failed"):
                DaxBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=None,
                    loop=None,
                    dst_device="cuda:0",
                )
        fd_after = len(os.listdir("/proc/self/fd"))
        assert fd_after == fd_before
