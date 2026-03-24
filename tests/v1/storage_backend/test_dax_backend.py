# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
import asyncio
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
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.dax_backend import DaxBackend


@pytest.fixture
def loop_in_thread() -> Generator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


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


def test_dax_backend_roundtrip(memory_allocator, loop_in_thread):
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
                "dax.max_dax_size": 16 / 1024,
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
                "dax.max_dax_size": 4 / 1024,
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


def test_dax_backend_requires_local_cpu_backend() -> None:
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
                "dax.max_dax_size": 4096 / 1024**3,
            },
        )
        metadata = _create_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="requires local_cpu_backend"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cpu",
            )


def test_dax_backend_rejects_multi_group_metadata_at_init() -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dax.bin")
        with open(dev_path, "wb") as fout:
            fout.truncate(8 * 1024 * 1024)

        config = _create_config(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=0.1,
            extra_config={
                "dax.device_path": dev_path,
                "dax.max_dax_size": 8 / 1024,
            },
        )
        metadata = _create_multi_group_metadata(chunk_size=16)

        with pytest.raises(ValueError, match="single-group KV layout"):
            DaxBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=None,
                loop=None,
                dst_device="cpu",
            )


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
                "dax.max_dax_size": 16 / 1024,
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


def test_dax_backend_oversized_put_raises_error(
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
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
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

            with pytest.raises(ValueError, match="exceeds slot size"):
                backend.batched_submit_put_task([oversized_key], [oversized])
            assert not backend.contains(oversized_key)
            oversized.ref_count_down()
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
                "dax.max_dax_size": 64 / (1024 * 1024),
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
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
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

            valid_key = CacheEngineKey("test_model", 1, 0, 708, torch.bfloat16)
            reclaimed = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert reclaimed is not None
            backend.batched_submit_put_task([valid_key], [reclaimed])
            out = backend.get_blocking(valid_key)
            assert out is not None
            out.ref_count_down()
            reclaimed.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_get_blocking_releases_lock_during_read(
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
                "dax.max_dax_size": 16 / 1024,
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
        reader_result: list[MemoryObj | None] = []
        remove_result: list[bool] = []
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
                reader_result.append(backend.get_blocking(key1))

            def _remover() -> None:
                remove_result.append(backend.remove(key2))
                remove_finished.set()

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            remover = threading.Thread(target=_remover)
            remover.start()
            assert remove_finished.wait(timeout=0.2)
            assert remove_result[0] is True

            allow_read.set()
            reader.join(timeout=1)
            remover.join(timeout=1)
            assert not reader.is_alive()
            assert not remover.is_alive()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 3)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_remove_during_read_defers_slot_reclaim(
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
                "dax.max_dax_size": 32 / (1024 * 1024),
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
        reader_result: list[MemoryObj | None] = []
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
                reader_result.append(backend.get_blocking(key))

            reader = threading.Thread(target=_reader)
            reader.start()
            assert read_started.wait(timeout=1)

            assert backend.remove(key)
            blocked_key = CacheEngineKey("test_model", 1, 0, 704, torch.bfloat16)
            blocked = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert blocked is not None
            with pytest.raises(RuntimeError, match="No free slots available"):
                backend.batched_submit_put_task([blocked_key], [blocked])
            blocked.ref_count_down()

            allow_read.set()
            reader.join(timeout=1)
            assert not reader.is_alive()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 7)
            result.ref_count_down()

            recycled_key = CacheEngineKey("test_model", 1, 0, 705, torch.bfloat16)
            recycled = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert recycled is not None
            backend.batched_submit_put_task([recycled_key], [recycled])
            recycled_out = backend.get_blocking(recycled_key)
            assert recycled_out is not None
            recycled_out.ref_count_down()
            recycled.ref_count_down()
            assert backend.get_blocking(key) is None
        finally:
            backend.close()


def test_dax_backend_get_read_failure_releases_cpu_memory_obj(
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
                "dax.max_dax_size": 32 / (1024 * 1024),
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


def test_dax_backend_allocator_exhaustion_triggers_eviction(
    memory_allocator,
    loop_in_thread,
):
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
                "dax.max_dax_size": 64 / (1024 * 1024),  # ~2 slots for test kv shape
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
                "dax.max_dax_size": 64 / (1024 * 1024),
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


def test_dax_backend_overlapping_pin_unpin(memory_allocator, loop_in_thread):
    """Regression: overlapping pin/unpin must use ref-counting, not a plain set."""
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
                "dax.max_dax_size": 64 / (1024 * 1024),  # 2 slots
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
                CacheEngineKey("test_model", 1, 0, 401, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 402, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 403, torch.bfloat16),
                CacheEngineKey("test_model", 1, 0, 404, torch.bfloat16),
            ]

            # Store keys A and B (fills both slots)
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

            # Pin key A twice (simulate two concurrent lookups)
            assert backend.pin(keys[0])
            assert backend.pin(keys[0])

            # Unpin key A once — pin_count should still be 1
            backend.unpin(keys[0])

            # Store key C — forces eviction; key A must survive (still pinned)
            obj_c = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj_c is not None
            futs = backend.batched_submit_put_task([keys[2]], [obj_c])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj_c.ref_count_down()

            # Key A is still retrievable (protected by remaining pin)
            assert backend.contains(keys[0]), (
                "key A should survive eviction (pin_count=1)"
            )

            # Key B was evicted (unpinned)
            assert not backend.contains(keys[1]), "key B should be evicted"

            # Unpin key A a second time — pin_count → 0
            backend.unpin(keys[0])

            # Store key D — forces eviction; key A can now be evicted
            obj_d = alloc.allocate(
                [torch.Size([2, 256, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj_d is not None
            futs = backend.batched_submit_put_task([keys[3]], [obj_d])
            if futs:
                for fut in futs:
                    fut.result(timeout=5)
            obj_d.ref_count_down()

            # Key A should now be evicted
            assert not backend.contains(keys[0]), (
                "key A should be evictable after full unpin"
            )
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
                "dax.max_dax_size": 32 / (1024 * 1024),  # one slot
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
                "dax.max_dax_size": 16 / 1024,
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
                "dax.max_dax_size": 16 / 1024,
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

            def _closer() -> None:
                backend.close()
                close_returned.set()

            writer = threading.Thread(target=_writer)
            closer = threading.Thread(target=_closer)

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


def test_dax_backend_sync_close_waits_for_active_get(
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
                "dax.max_dax_size": 16 / 1024,
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
        reader_result: list[MemoryObj | None] = []
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
                reader_result.append(backend.get_blocking(key))

            def _closer() -> None:
                backend.close()
                close_returned.set()

            reader = threading.Thread(target=_reader)
            closer = threading.Thread(target=_closer)

            reader.start()
            assert read_started.wait(timeout=1)
            closer.start()
            time.sleep(0.05)

            assert not close_returned.is_set()
            assert backend._mmap_obj is not None  # noqa: SLF001
            assert backend._base_ptr != 0  # noqa: SLF001

            allow_read.set()
            reader.join(timeout=1)
            closer.join(timeout=1)

            assert not reader.is_alive()
            assert not closer.is_alive()
            assert close_returned.is_set()

            result = reader_result[0]
            assert result is not None
            assert result.tensor is not None
            assert torch.all(result.tensor == 6)
            result.ref_count_down()
        finally:
            backend.close()


def test_dax_backend_close_rejects_new_ops_after_shutdown() -> None:
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
                "dax.max_dax_size": 16 / 1024,
            },
        )
        metadata = _create_metadata(chunk_size=16)
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=AdHocMemoryAllocator(device="cpu"),
        )
        backend = DaxBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=None,
            dst_device="cpu",
        )

        try:
            backend.close()
            assert (
                backend.get_blocking(
                    CacheEngineKey("test_model", 1, 0, 999, torch.bfloat16)
                )
                is None
            )
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
                "dax.max_dax_size": 16 / 1024,
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

        write_started = threading.Event()
        allow_write = threading.Event()
        close_returned = threading.Event()
        original_do_write = DaxBackend._do_write

        def _blocking_do_write(self, offset, memory_obj, size) -> None:
            write_started.set()
            assert allow_write.wait(timeout=2)
            original_do_write(self, offset, memory_obj, size)

        monkeypatch.setattr(DaxBackend, "_do_write", _blocking_do_write)

        # Start the event loop so the async_put path is exercised.
        loop_thread = threading.Thread(target=loop_in_thread.run_forever, daemon=True)
        loop_thread.start()
        try:
            alloc = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 411, torch.bfloat16)
            obj = alloc.allocate(
                [torch.Size([2, 16, 8])],
                [torch.bfloat16],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None

            def _closer() -> None:
                backend.close()
                close_returned.set()

            assert write_started.wait(timeout=2)
            closer = threading.Thread(target=_closer)
            closer.start()
            time.sleep(0.05)
            assert not close_returned.is_set()

            allow_write.set()
            closer.join(timeout=2)

            assert not closer.is_alive()
            assert close_returned.is_set()
            obj.ref_count_down()
        finally:
            loop_in_thread.call_soon_threadsafe(loop_in_thread.stop)
            loop_thread.join(timeout=2)
            backend.close()
