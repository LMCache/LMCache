# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import json
import os
import shutil
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
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

    def test_manifest_roundtrip_persists_across_restart(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test local disk cache restoration across backend restarts."""
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
                assert os.path.exists(backend2.manifest_path)
                assert backend2.contains(key)
                restored = backend2.get_blocking(key)
                assert restored is not None
                assert bytes(restored.byte_array) == expected
                restored.ref_count_down()
            finally:
                backend2.close()
        finally:
            local_cpu_backend.memory_allocator.close()

    def test_close_skips_manifest_write_when_cache_dir_missing(
        self, temp_disk_path, loop_in_thread, memory_allocator, capsys
    ):
        """Test close() skips manifest persistence after cache dir deletion."""
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

        try:
            shutil.rmtree(temp_disk_path)
            backend.close()
            captured = capsys.readouterr()
            assert "Failed to persist local disk manifest" not in captured.err
        finally:
            local_cpu_backend.memory_allocator.close()

    def test_get_blocking_prunes_missing_file(
        self, temp_disk_path, loop_in_thread, memory_allocator
    ):
        """Test missing disk files degrade to a cache miss and manifest prune."""
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

            with open(backend.manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
            assert key.to_string() not in manifest_data["entries"]
        finally:
            backend.close()
            local_cpu_backend.memory_allocator.close()

    def test_manifest_write_cleans_temp_file_when_close_fails(
        self, temp_disk_path, loop_in_thread, memory_allocator, monkeypatch
    ):
        """Test close() removes manifest temp files when persistence fails."""
        config = create_test_config(temp_disk_path)
        metadata = create_test_metadata()
        local_cpu_backend = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        local_disk_backend = LocalDiskBackend(
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
        created_tmp_paths: list[str] = []

        def failing_dump(obj, fp, *args, **kwargs):
            created_tmp_paths.append(fp.name)
            fp.write("{")
            fp.flush()
            raise OSError("disk full")

        monkeypatch.setattr(local_disk_backend_module.json, "dump", failing_dump)

        try:
            local_disk_backend.submit_put_task(key, memory_obj)
            wait_for_disk_store(local_disk_backend, key)
            memory_obj.ref_count_down()
            local_disk_backend.close()

            assert created_tmp_paths
            assert not os.path.exists(created_tmp_paths[0])
            assert not os.path.exists(local_disk_backend.manifest_path)
        finally:
            local_disk_backend.close()
            local_cpu_backend.memory_allocator.close()

    def test_manifest_restore_skips_usage_recompute_stat_race(
        self, temp_disk_path, loop_in_thread, memory_allocator, monkeypatch
    ):
        """Test restart restore tolerates getsize racing with file removal."""
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
        key = create_test_key(32)
        memory_obj = create_memory_obj(
            local_cpu_backend,
            metadata.get_shapes()[0],
            metadata.get_dtypes()[0],
            fill_value=4,
            fmt=MemoryFormat.KV_2LTD,
        )

        try:
            backend1.submit_put_task(key, memory_obj)
            wait_for_disk_store(backend1, key)
            memory_obj.ref_count_down()
            backend1.close()

            path = os.path.join(
                temp_disk_path, key.to_string().replace("/", "-") + ".pt"
            )
            original_getsize = local_disk_backend_module.os.path.getsize
            getsize_calls = 0

            def flaky_getsize(target_path: str) -> int:
                nonlocal getsize_calls
                if target_path == path:
                    getsize_calls += 1
                    if getsize_calls >= 2:
                        raise FileNotFoundError(target_path)
                return original_getsize(target_path)

            monkeypatch.setattr(
                local_disk_backend_module.os.path,
                "getsize",
                flaky_getsize,
            )

            backend2 = LocalDiskBackend(
                config=config,
                loop=loop_in_thread,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
                metadata=metadata,
            )
            try:
                assert backend2.contains(key)
                assert backend2.usage == 0
            finally:
                backend2.close()
        finally:
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
        """Default local_disk_path_sharding is 'by_gpu'."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.local_disk_path_sharding == "by_gpu"
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
        assert backend.local_disk_path_sharding == "by_gpu"
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises AssertionError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(
            AssertionError, match="Unsupported local_disk_path_sharding"
        ):
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
