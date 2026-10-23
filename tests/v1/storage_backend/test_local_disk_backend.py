# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, call, patch
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.inprocess.v1.cache_controller.message import OpType
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


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
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def put_on_disk(backend: LocalDiskBackend, key: CacheEngineKey, num_tokens: int) -> str:
    """Store ``key`` through submit_put_task() and wait for the write to land.

    Requires the backend's event loop to be running (see ``running_loop``).

    Returns:
        The path of the file the put wrote.
    """
    memory_obj = backend.local_cpu_backend.allocate(
        torch.Size([2, num_tokens, 8, 128]),  # KV_T2D: tokens on dim 1
        torch.bfloat16,
        MemoryFormat.KV_T2D,
    )
    assert memory_obj is not None
    written = threading.Event()
    backend.submit_put_task(
        key, memory_obj, on_complete_callback=lambda _: written.set()
    )
    memory_obj.ref_count_down()
    assert written.wait(10), "disk write did not complete"
    return backend.dict[key].path


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
def running_loop(async_loop):
    """Run ``async_loop`` on a thread so disk puts actually complete."""
    thread = threading.Thread(target=async_loop.run_forever, daemon=True)
    thread.start()
    yield async_loop
    async_loop.call_soon_threadsafe(async_loop.stop)
    thread.join(timeout=10)


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
        dst_device=f"{torch_device_type}:0",
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
            dst_device=f"{torch_device_type}:0",
        )

        assert backend.dst_device == f"{torch_device_type}:0"
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
            dst_device=f"{torch_device_type}:0",
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
                dst_device=f"{torch_device_type}:0",
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
        """Different accelerator device indices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in (f"{torch_device_type}:0", f"{torch_device_type}:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu[f"{torch_device_type}:0"] == dir_a
            assert dirs_by_gpu[f"{torch_device_type}:1"] == dir_b
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
                dst_device=f"{torch_device_type}:0",
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
            dst_device=f"{torch_device_type}:0",
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
            dst_device=f"{torch_device_type}:0",
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
            dst_device=f"{torch_device_type}:0",
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
                dst_device=f"{torch_device_type}:0",
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


class TestBatchedGetBlocking:
    """Tests for the concurrent batched_get_blocking() override."""

    _SHAPE = torch.Size([28, 2, 256, 8, 128])
    _DTYPE = torch.bfloat16

    def _write_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
    ) -> bytes:
        """Write random data to disk for *key* and register it in the backend.

        Returns the raw bytes that were written so callers can verify reads.
        """
        path = backend._key_to_path(key)
        nbytes = 1
        for s in self._SHAPE:
            nbytes *= s
        nbytes *= self._DTYPE.itemsize
        data = os.urandom(nbytes)
        with open(path, "wb") as f:
            f.write(data)
        backend.insert_key(
            key,
            size=nbytes,
            shape=self._SHAPE,
            dtype=self._DTYPE,
            fmt=MemoryFormat.KV_2LTD,
        )
        return data

    def test_all_keys_missing(self, local_disk_backend: LocalDiskBackend) -> None:
        """batched_get_blocking returns [None, …] when no keys are cached."""
        keys = [create_test_key(i) for i in range(200, 204)]
        results = local_disk_backend.batched_get_blocking(keys)

        assert len(results) == len(keys)
        assert all(r is None for r in results)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_reads_match_written_data(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """Data loaded by batched_get_blocking matches what was written."""
        keys = [create_test_key(i) for i in range(210, 214)]
        expected_data = {}
        for key in keys:
            expected_data[key] = self._write_key(local_disk_backend, key)

        results = local_disk_backend.batched_get_blocking(keys)

        assert len(results) == len(keys)
        for key, mem_obj in zip(keys, results, strict=True):
            assert mem_obj is not None, f"Expected data for {key}"
            actual = bytes(mem_obj.byte_array)
            assert actual == expected_data[key]
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_mixed_hit_and_miss(self, local_disk_backend: LocalDiskBackend) -> None:
        """Handles a mix of cached and missing keys correctly."""
        present_key = create_test_key(220)
        missing_key = create_test_key(221)
        expected = self._write_key(local_disk_backend, present_key)

        results = local_disk_backend.batched_get_blocking([present_key, missing_key])

        assert len(results) == 2
        assert results[0] is not None
        assert bytes(results[0].byte_array) == expected
        assert results[1] is None
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_cache_policy_updated_only_for_hits(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit is called only for keys that were successfully loaded."""
        hit_key = create_test_key(230)
        miss_key = create_test_key(231)
        self._write_key(local_disk_backend, hit_key)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            results = local_disk_backend.batched_get_blocking([hit_key, miss_key])

        assert results[0] is not None
        assert results[1] is None
        mock_update.assert_called_once_with(hit_key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_single_key_delegates_to_get_blocking(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """A single-key batch falls through to get_blocking (fast path)."""
        key = create_test_key(240)
        sentinel = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "get_blocking", return_value=sentinel
        ) as mock_get:
            results = local_disk_backend.batched_get_blocking([key])

        assert results == [sentinel]
        mock_get.assert_called_once_with(key)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_concurrent_reads_use_multiple_threads(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """Verify that reads actually fan out across threads."""
        keys = [create_test_key(i) for i in range(250, 254)]
        for key in keys:
            self._write_key(local_disk_backend, key)

        thread_ids: list[int] = []
        lock = threading.Lock()
        original_read_file = local_disk_backend.read_file

        def tracking_read_file(key, buffer, path):
            with lock:
                thread_ids.append(threading.current_thread().ident)
            return original_read_file(key, buffer, path)

        with patch.object(
            local_disk_backend, "read_file", side_effect=tracking_read_file
        ):
            results = local_disk_backend.batched_get_blocking(keys)

        assert all(r is not None for r in results)
        # With 4 keys and 4 threads, we should see more than 1 unique thread.
        assert len(set(thread_ids)) > 1, (
            f"Expected multiple threads, got {set(thread_ids)}"
        )
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_empty_keys_list(self, local_disk_backend: LocalDiskBackend) -> None:
        """An empty key list returns an empty result list."""
        results = local_disk_backend.batched_get_blocking([])
        assert results == []
        local_disk_backend.local_cpu_backend.memory_allocator.close()


class TestSubmitPutTask:
    """Tests for LocalDiskBackend.submit_put_task()."""

    @pytest.mark.parametrize(
        "put_in_progress",
        [False, True],
        ids=["resident_only", "resident_and_inflight"],
    )
    def test_already_stored_key_is_not_rewritten(self, put_in_progress: bool) -> None:
        key = create_test_key(1)
        resident_metadata = MagicMock()
        backend = object.__new__(LocalDiskBackend)
        backend.disk_lock = threading.Lock()
        backend.dict = {key: resident_metadata}
        backend.cache_policy = MagicMock()
        backend.disk_worker = MagicMock()
        backend.current_cache_size = 4096
        backend.usage = 4096

        memory_obj = MagicMock(spec=MemoryObj)
        memory_obj.tensor = torch.empty(1, dtype=torch.bfloat16)
        callback = MagicMock()

        with patch.object(
            backend, "exists_in_put_tasks", return_value=put_in_progress
        ) as mock_exists_in_put_tasks:
            result = backend.submit_put_task(
                key,
                memory_obj,
                on_complete_callback=callback,
            )

        assert result is None
        assert backend.current_cache_size == 4096
        assert backend.usage == 4096
        backend.cache_policy.update_on_hit.assert_called_once_with(key, backend.dict)
        mock_exists_in_put_tasks.assert_not_called()
        backend.disk_worker.insert_put_task.assert_not_called()
        backend.disk_worker.submit_task.assert_not_called()
        memory_obj.get_physical_size.assert_not_called()
        memory_obj.ref_count_up.assert_not_called()
        callback.assert_not_called()


class TestLocalDiskBackendClear:
    """Contract tests for LocalDiskBackend.clear() (issue #5835)."""

    def test_clear_removes_unpinned_entries_and_files(
        self, local_disk_backend: LocalDiskBackend, running_loop
    ) -> None:
        try:
            keys = [create_test_key(i) for i in range(3)]
            paths = [put_on_disk(local_disk_backend, k, num_tokens=16) for k in keys]

            assert local_disk_backend.clear() == 3 * 16
            assert not any(local_disk_backend.contains(k) for k in keys)
            assert not any(os.path.exists(path) for path in paths)
            assert local_disk_backend.usage == 0
            assert local_disk_backend.current_cache_size == 0
        finally:
            local_disk_backend.close()

    def test_clear_keeps_pinned_entries(
        self, local_disk_backend: LocalDiskBackend, running_loop
    ) -> None:
        try:
            pinned_key, free_key = create_test_key(10), create_test_key(11)
            pinned_path = put_on_disk(local_disk_backend, pinned_key, num_tokens=16)
            free_path = put_on_disk(local_disk_backend, free_key, num_tokens=8)
            pinned_size = local_disk_backend.dict[pinned_key].size
            assert local_disk_backend.pin(pinned_key)

            assert local_disk_backend.clear() == 8
            assert local_disk_backend.contains(pinned_key)
            assert os.path.exists(pinned_path)
            assert not local_disk_backend.contains(free_key)
            assert not os.path.exists(free_path)
            assert local_disk_backend.usage == pinned_size
            assert local_disk_backend.current_cache_size == pinned_size
        finally:
            local_disk_backend.close()

    def test_clear_treats_missing_file_as_removed(
        self, temp_disk_path, running_loop, local_cpu_backend
    ) -> None:
        """A file deleted outside LMCache must not abort clear() part-way.

        Uses LFU: if a key left the dict without update_on_force_evict, its
        frequency entry would go stale and the next capacity eviction would
        raise KeyError.
        """
        entry_bytes = 2 * 8 * 8 * 128 * torch.bfloat16.itemsize  # num_tokens=8
        config = create_test_config(
            temp_disk_path, max_disk_size=2.5 * entry_bytes / 1024**3
        )
        config.cache_policy = "LFU"
        backend = LocalDiskBackend(
            config=config,
            loop=running_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device=f"{torch_device_type}:0",
        )
        try:
            keys = [create_test_key(i) for i in range(20, 22)]
            paths = [put_on_disk(backend, key, num_tokens=8) for key in keys]
            os.remove(paths[1])

            assert backend.clear() == 2 * 8
            assert not any(backend.contains(key) for key in keys)
            assert not any(os.path.exists(path) for path in paths)

            # Room for two entries: the third put has to evict one.
            refill = [create_test_key(i) for i in range(30, 33)]
            for key in refill:
                put_on_disk(backend, key, num_tokens=8)
            assert sum(backend.contains(key) for key in refill) == 2
        finally:
            backend.close()

    def test_clear_failure_keeps_earlier_removals_consistent(
        self, local_disk_backend: LocalDiskBackend, running_loop
    ) -> None:
        """An unlink error stops clear(); entries cleared before it stay cleared
        and announced, and the failing entry stays tracked."""
        try:
            first, second = create_test_key(50), create_test_key(51)
            put_on_disk(local_disk_backend, first, num_tokens=8)
            second_path = put_on_disk(local_disk_backend, second, num_tokens=8)
            second_size = local_disk_backend.dict[second].size
            sender = MagicMock()
            local_disk_backend.batched_msg_sender = sender
            real_remove = os.remove

            def fail_on_second(path: str) -> None:
                if path == second_path:
                    raise PermissionError(path)
                real_remove(path)

            with patch(
                "lmcache.v1.storage_backend.local_disk_backend.os.remove",
                side_effect=fail_on_second,
            ):
                with pytest.raises(PermissionError):
                    local_disk_backend.clear()

            assert not local_disk_backend.contains(first)
            sender.add_kv_op.assert_called_once_with(
                op_type=OpType.EVICT, key=first.chunk_hash
            )
            assert local_disk_backend.contains(second)
            assert os.path.exists(second_path)
            assert local_disk_backend.usage == second_size
        finally:
            local_disk_backend.close()

    def test_clear_sends_evict_message_per_cleared_key(
        self, local_disk_backend: LocalDiskBackend, running_loop
    ) -> None:
        try:
            keys = [create_test_key(i) for i in range(40, 42)]
            for key in keys:
                put_on_disk(local_disk_backend, key, num_tokens=8)
            sender = MagicMock()
            local_disk_backend.batched_msg_sender = sender

            local_disk_backend.clear()

            sender.add_kv_op.assert_has_calls(
                [call(op_type=OpType.EVICT, key=key.chunk_hash) for key in keys],
                any_order=True,
            )
            assert sender.add_kv_op.call_count == len(keys)
        finally:
            local_disk_backend.close()

    def test_clear_on_empty_cache_returns_zero(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        assert local_disk_backend.clear() == 0
        local_disk_backend.local_cpu_backend.memory_allocator.close()
