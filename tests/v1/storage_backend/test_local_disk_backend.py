# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os
import shutil
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
import lmcache.v1.storage_backend.local_disk_backend as local_disk_backend_module


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(
    disk_path: str,
    max_disk_size: float = 1.0,
    local_disk_path_sharding: str = "by_gpu",
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        local_disk_path_sharding=local_disk_path_sharding,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def create_memory_obj(
    local_cpu_backend: LocalCPUBackend,
    shape: torch.Size,
    dtype: torch.dtype,
    fill_value: int,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
):
    """Create a CPU memory object filled with a deterministic value."""
    memory_obj = local_cpu_backend.allocate(shape, dtype, fmt=fmt)
    assert memory_obj is not None
    assert memory_obj.tensor is not None
    memory_obj.tensor.fill_(fill_value)
    return memory_obj


def wait_for_disk_store(
    backend: LocalDiskBackend,
    key: CacheEngineKey,
    timeout: float = 5.0,
) -> None:
    """Wait until an asynchronous disk store becomes visible."""
    start = time.time()
    while backend.exists_in_put_tasks(key):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for disk store of {key}")
        time.sleep(0.01)

    while not backend.contains(key):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for disk cache entry {key}")
        time.sleep(0.01)


def list_manifest_files(path: str) -> list[str]:
    return [
        name
        for name in os.listdir(path)
        if name.startswith(".lmcache-local-disk-manifest-")
    ]


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def loop_in_thread():
    """Create a background event loop for async disk stores."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda:0",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )

        assert backend.dst_device == "cuda:0"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_recovers_after_restart_using_filename(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test local disk cache recovery across backend restarts."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend1 = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(11)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=7,
            fmt=MemoryFormat.KV_2LTD,
        )
        expected = bytes(memory_obj.byte_array)

        try:
            backend1.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend1, key)
            memory_obj.ref_count_down()
            backend1.close()

            backend2 = LocalDiskBackend(
                config=config,
                loop=loop_in_thread,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
                metadata=metadata,
            )
            try:
                assert key not in backend2.dict
                assert list_manifest_files(temp_disk_path) == []
                assert backend2.contains(key)
                assert key in backend2.dict
                restored = backend2.get_blocking(key)
                assert restored is not None
                assert bytes(restored.byte_array) == expected
                restored.ref_count_down()
            finally:
                backend2.close()
        finally:
            local_cpu_backend.memory_allocator.close()

    def test_contains_recovers_partial_chunk_after_restart_using_file_size(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test restart recovery infers partial chunk metadata from file size."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend1 = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(14)
        partial_tokens = metadata.chunk_size // 2
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes(partial_tokens)[0],
            metadata.get_dtypes()[0],
            fill_value=8,
            fmt=MemoryFormat.KV_2LTD,
        )
        expected = bytes(memory_obj.byte_array)

        try:
            backend1.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend1, key)
            memory_obj.ref_count_down()
            backend1.close()

            backend2 = LocalDiskBackend(
                config=config,
                loop=loop_in_thread,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
                metadata=metadata,
            )
            try:
                assert backend2.contains(key)
                restored = backend2.get_blocking(key)
                assert restored is not None
                assert restored.get_shape() == metadata.get_shapes(partial_tokens)[0]
                assert bytes(restored.byte_array) == expected
                restored.ref_count_down()
            finally:
                backend2.close()
        finally:
            local_cpu_backend.memory_allocator.close()

    def test_batched_async_contains_recovers_consecutive_files(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test batched lookup lazily recovers files until the first miss."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key1 = create_test_key(21)
        key2 = create_test_key(22)
        missing_key = create_test_key(23)
        memory_obj1 = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=1,
            fmt=MemoryFormat.KV_2LTD,
        )
        memory_obj2 = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=2,
            fmt=MemoryFormat.KV_2LTD,
        )
        memory_obj1_released = False
        memory_obj2_released = False

        try:
            with open(backend._key_to_path(key1), "wb") as f:
                f.write(memory_obj1.byte_array)
            with open(backend._key_to_path(key2), "wb") as f:
                f.write(memory_obj2.byte_array)
            memory_obj1.ref_count_down()
            memory_obj1_released = True
            memory_obj2.ref_count_down()
            memory_obj2_released = True

            assert len(backend.dict) == 0
            future = asyncio.run_coroutine_threadsafe(
                backend.batched_async_contains(
                    "lookup",
                    [key1, key2, missing_key],
                ),
                loop_in_thread,
            )
            hits = future.result(timeout=5)

            assert hits == 2
            assert key1 in backend.dict
            assert key2 in backend.dict
            assert missing_key not in backend.dict
            assert list_manifest_files(temp_disk_path) == []
        finally:
            backend.close()
            if not memory_obj1_released:
                memory_obj1.ref_count_down()
            if not memory_obj2_released:
                memory_obj2.ref_count_down()
            local_cpu_backend.memory_allocator.close()

    def test_batched_async_contains_pin_true_does_not_leak_disk_pin(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test async disk lookup pins are released after prefetch cleanup."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(26)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=4,
            fmt=MemoryFormat.KV_2LTD,
        )
        loaded_mem_objs: list[MemoryObj] = []
        memory_obj_released = False
        PinMonitor.GetOrCreate(config, metadata)

        try:
            backend.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend, key)
            memory_obj.ref_count_down()
            memory_obj_released = True

            assert backend.dict[key].pin_count == 0
            contains_future = asyncio.run_coroutine_threadsafe(
                backend.batched_async_contains("lookup", [key], pin=True),
                loop_in_thread,
            )
            assert contains_future.result(timeout=5) == 1
            assert backend.dict[key].pin_count == 0

            get_future = asyncio.run_coroutine_threadsafe(
                backend.batched_get_non_blocking("lookup", [key]),
                loop_in_thread,
            )
            loaded_mem_objs = get_future.result(timeout=5)
            assert len(loaded_mem_objs) == 1

            for loaded_mem_obj in loaded_mem_objs:
                if loaded_mem_obj.is_pinned:
                    loaded_mem_obj.unpin()
                loaded_mem_obj.ref_count_down()
            loaded_mem_objs = []

            assert backend.dict[key].pin_count == 0
        finally:
            backend.close()
            for loaded_mem_obj in loaded_mem_objs:
                if loaded_mem_obj.is_pinned:
                    loaded_mem_obj.unpin()
                loaded_mem_obj.ref_count_down()
            if not memory_obj_released:
                memory_obj.ref_count_down()
            PinMonitor.DestroyInstance()
            local_cpu_backend.memory_allocator.close()

    def test_contains_handles_file_disappearing_before_getsize(
        self, temp_disk_path, loop_in_thread, memory_allocator, monkeypatch
    ):
        """Test stale files racing with lazy recovery degrade to a miss."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(24)
        path = backend._key_to_path(key)

        try:
            with open(path, "wb") as f:
                f.write(b"stale")

            original_getsize = local_disk_backend_module.os.path.getsize

            def disappearing_getsize(target_path: str) -> int:
                if target_path == path:
                    os.remove(path)
                    raise FileNotFoundError(path)
                return original_getsize(target_path)

            monkeypatch.setattr(
                local_disk_backend_module.os.path,
                "getsize",
                disappearing_getsize,
            )

            assert not backend.contains(key)
            assert key not in backend.dict
        finally:
            backend.close()
            local_cpu_backend.memory_allocator.close()

    def test_contains_rejects_uninferrable_file_size(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test file sizes that cannot map to KV metadata remain misses."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(25)

        try:
            with open(backend._key_to_path(key), "wb") as f:
                f.write(b"stale")

            assert not backend.contains(key)
            assert key not in backend.dict
        finally:
            backend.close()
            local_cpu_backend.memory_allocator.close()

    def test_get_blocking_prunes_missing_file(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test missing disk files degrade to a cache miss."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(12)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=9,
            fmt=MemoryFormat.KV_2LTD,
        )

        try:
            backend.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend, key)
            memory_obj.ref_count_down()

            path = backend.dict[key].path
            os.remove(path)

            assert backend.get_blocking(key) is None
            assert key not in backend.dict
            assert backend.current_cache_size == 0
            assert backend.usage == 0
            assert list_manifest_files(temp_disk_path) == []
        finally:
            backend.close()
            local_cpu_backend.memory_allocator.close()

    def test_contains_ignores_file_while_put_task_is_active(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test in-progress disk writes do not recover as cache hits."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(13)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=5,
            fmt=MemoryFormat.KV_2LTD,
        )
        write_started = threading.Event()
        finish_write = threading.Event()
        memory_obj_released = False
        original_write_file = backend.write_file
        original_remove_put_task = backend.disk_worker.remove_put_task
        written_paths: list[str] = []
        remove_put_task_started = threading.Event()
        finish_remove_put_task = threading.Event()

        def blocking_write_file(buffer: memoryview, path: str) -> None:
            written_paths.append(path)
            with open(path, "wb"):
                pass
            write_started.set()
            assert finish_write.wait(timeout=5)
            original_write_file(buffer, path)

        def blocking_remove_put_task(task_key: CacheEngineKey) -> None:
            remove_put_task_started.set()
            assert finish_remove_put_task.wait(timeout=5)
            original_remove_put_task(task_key)

        backend.write_file = blocking_write_file
        backend.disk_worker.remove_put_task = blocking_remove_put_task

        try:
            backend.submit_put_task(key, memory_obj)
            assert write_started.wait(timeout=5)
            assert backend.exists_in_put_tasks(key)
            assert written_paths
            assert os.path.isfile(written_paths[0])

            assert not backend.contains(key)

            finish_write.set()
            assert remove_put_task_started.wait(timeout=5)
            assert backend.exists_in_put_tasks(key)
            assert backend.contains(key)

            finish_remove_put_task.set()
            wait_for_disk_store(backend, key)
            memory_obj.ref_count_down()
            memory_obj_released = True
            assert backend.contains(key)
        finally:
            finish_write.set()
            finish_remove_put_task.set()
            backend.close()
            if not memory_obj_released:
                memory_obj.ref_count_down()
            local_cpu_backend.memory_allocator.close()

    def test_no_manifest_file_created_for_put_remove_close(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test local disk operations do not persist a manifest file."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = LocalDiskBackend(
            config=config,
            loop=loop_in_thread,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
            metadata=metadata,
        )
        key = create_test_key(31)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=3,
            fmt=MemoryFormat.KV_2LTD,
        )

        try:
            backend.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend, key)
            memory_obj.ref_count_down()
            assert backend.remove(key)
            backend.close()
            assert list_manifest_files(temp_disk_path) == []
        finally:
            backend.close()
            local_cpu_backend.memory_allocator.close()


class TestMultiPathDiskBackend:
    """Test cases for multi-path (multi-device) LocalDiskBackend."""

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Test initialisation with comma-separated paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            # Path selected by device_id (0 % 2 = 0 -> dir_a)
            assert backend.path == dir_a
            # Both directories should exist
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            # Block size is a plain int for the selected path
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_gpu_affinity_selects_path(self, async_loop, local_cpu_backend):
        """Different cuda devices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in ("cuda:0", "cuda:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu["cuda:0"] == dir_a
            assert dirs_by_gpu["cuda:1"] == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_all_directories_created(self, async_loop, local_cpu_backend):
        """All paths in the list get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            combined = ",".join(paths)
            config = create_test_config(combined)
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            for p in paths:
                assert os.path.isdir(p), f"{p} should exist"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_backward_compat(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """A single path (no commas) works exactly as before."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_default(self, temp_disk_path, async_loop, local_cpu_backend):
        """Default local_disk_path_sharding is 'by_gpu' (backend inits OK)."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_explicit_by_gpu(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Explicitly setting local_disk_path_sharding='by_gpu' works."""
        config = create_test_config(temp_disk_path, local_disk_path_sharding="by_gpu")
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises ValueError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

    def test_cpu_dst_device_defaults_to_first_path(self, async_loop, local_cpu_backend):
        """dst_device='cpu' should fall back to device_id=0."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )
            # device_id=0 -> 0 % 2 = 0 -> dir_a
            assert backend.path == dir_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()


class TestParseLocalDisk:
    """Unit tests for the _parse_local_disk config parser."""

    def test_none(self):
        assert _parse_local_disk(None) is None

    def test_single_raw_path(self):
        assert _parse_local_disk("/mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri_no_trailing_slash(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache") == "/mnt/nvme0/cache"

    def test_comma_separated_raw(self):
        result = _parse_local_disk("/mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_comma_separated_file_uris(self):
        result = _parse_local_disk("file:///mnt/nvme0/,file:///mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_mixed_uri_and_raw(self):
        result = _parse_local_disk("file:///mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_whitespace_around_paths(self):
        result = _parse_local_disk("  /mnt/nvme0/ , /mnt/nvme1/  ")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_empty_string(self):
        assert _parse_local_disk("") is None


class TestGetBlockingCachePolicyUpdate:
    """Regression tests for phantom cache hit in get_blocking() (issue #3015).

    ``get_blocking()`` must call ``cache_policy.update_on_hit()`` only when
    ``load_bytes_from_disk()`` returns a valid ``MemoryObj``.  Calling it
    before confirming load success records a phantom hit that skews future
    eviction decisions.
    """

    def _inject_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        """Insert a key into backend.dict without writing anything to disk."""
        meta = DiskCacheMetadata(
            path="/nonexistent/path.pt",
            size=0,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=MemoryFormat.KV_2LTD,
            pin_count=0,
        )
        with backend.disk_lock:
            backend.dict[key] = meta
            backend.cache_policy.update_on_put(key)

    def test_no_phantom_hit_when_load_fails(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must NOT be called when load_bytes_from_disk returns None."""
        key = create_test_key(101)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=None
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_updates_cache_policy_on_successful_load(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must be called exactly once when the load succeeds."""
        key = create_test_key(102)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        fake_memory_obj = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=fake_memory_obj
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is fake_memory_obj
        mock_update.assert_called_once_with(key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_absent_returns_none_without_policy_update(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """get_blocking must return None immediately when the key is not cached."""
        key = create_test_key(103)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()
