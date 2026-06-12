# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
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
        assert backend.max_cache_size == int(config.max_local_disk_size * 1024**3)
        assert backend.path_max_cache_sizes[temp_disk_path] == backend.max_cache_size
        assert backend.path_current_cache_sizes[temp_disk_path] == 0

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

    def test_equal_quota_is_split_across_assigned_paths(
        self, async_loop, local_cpu_backend
    ):
        """Assigned paths share the backend budget evenly."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            combined = ",".join(dirs)
            size = 512 * 1024
            config = create_test_config(combined, max_disk_size=(4 * size) / 1024**3)
            metadata = LMCacheMetadata(
                model_name="test_model",
                world_size=2,
                local_world_size=2,
                worker_id=1,
                local_worker_id=1,
                kv_dtype=torch.bfloat16,
                kv_shape=(28, 2, 256, 8, 128),
            )
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:1",
                metadata=metadata,
            )
            assert backend.assigned_paths == dirs[2:4]
            assert backend.path_max_cache_sizes == {dirs[2]: size, dirs[3]: size}
            assert backend.max_cache_size == 2 * size
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_init_fails_when_assigned_path_available_space_is_too_small(
        self, async_loop, local_cpu_backend
    ):
        """Init fails if assigned filesystems cannot satisfy their quota."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            combined = ",".join(dirs)
            size = 512 * 1024
            config = create_test_config(combined, max_disk_size=(4 * size) / 1024**3)
            metadata = LMCacheMetadata(
                model_name="test_model",
                world_size=2,
                local_world_size=2,
                worker_id=1,
                local_worker_id=1,
                kv_dtype=torch.bfloat16,
                kv_shape=(28, 2, 256, 8, 128),
            )

            real_stats = {path: os.statvfs(path) for path in dirs}

            class FakeStat:
                def __init__(self, *, f_bsize: int, f_frsize: int, f_bavail: int):
                    self.f_bsize = f_bsize
                    self.f_frsize = f_frsize
                    self.f_bavail = f_bavail

            def fake_statvfs(path):
                if path == dirs[2]:
                    return FakeStat(f_bsize=4096, f_frsize=4096, f_bavail=1)
                return real_stats[path]

            with patch(
                "lmcache.v1.storage_backend.local_disk_backend.os.statvfs",
                side_effect=fake_statvfs,
            ):
                with pytest.raises(ValueError, match="available filesystem space"):
                    LocalDiskBackend(
                        config=config,
                        loop=async_loop,
                        local_cpu_backend=local_cpu_backend,
                        dst_device="cuda:1",
                        metadata=metadata,
                    )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_init_fails_when_same_filesystem_lacks_combined_quota(
        self, async_loop, local_cpu_backend
    ):
        """Paths on one filesystem are validated against their combined quota."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            combined = ",".join(dirs)
            size = 512 * 1024
            config = create_test_config(combined, max_disk_size=(4 * size) / 1024**3)
            metadata = LMCacheMetadata(
                model_name="test_model",
                world_size=2,
                local_world_size=2,
                worker_id=1,
                local_worker_id=1,
                kv_dtype=torch.bfloat16,
                kv_shape=(28, 2, 256, 8, 128),
            )

            real_stats = {path: os.statvfs(path) for path in dirs}
            real_stat_results = {path: os.stat(path) for path in dirs}

            class FakeStatVFS:
                def __init__(self, *, f_bsize: int, f_frsize: int, f_bavail: int):
                    self.f_bsize = f_bsize
                    self.f_frsize = f_frsize
                    self.f_bavail = f_bavail

            class FakeStat:
                def __init__(self, *, st_dev: int):
                    self.st_dev = st_dev

            def fake_statvfs(path):
                if path in {dirs[2], dirs[3]}:
                    return FakeStatVFS(
                        f_bsize=4096,
                        f_frsize=4096,
                        f_bavail=(size + size // 2) // 4096,
                    )
                return real_stats[path]

            def fake_stat(path):
                if path in {dirs[2], dirs[3]}:
                    return FakeStat(st_dev=99)
                return real_stat_results[path]

            with (
                patch(
                    "lmcache.v1.storage_backend.local_disk_backend.os.statvfs",
                    side_effect=fake_statvfs,
                ),
                patch(
                    "lmcache.v1.storage_backend.local_disk_backend.os.stat",
                    side_effect=fake_stat,
                ),
            ):
                with pytest.raises(ValueError, match="available filesystem space"):
                    LocalDiskBackend(
                        config=config,
                        loop=async_loop,
                        local_cpu_backend=local_cpu_backend,
                        dst_device="cuda:1",
                        metadata=metadata,
                    )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_path_local_eviction_uses_only_target_path(
        self, async_loop, local_cpu_backend
    ):
        """Quota pressure evicts only entries from the target path."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            size = 512 * 1024
            config = create_test_config(
                f"{dir_a},{dir_b}", max_disk_size=(2 * size) / 1024**3
            )
            metadata = LMCacheMetadata(
                model_name="test_model",
                world_size=1,
                local_world_size=1,
                worker_id=0,
                local_worker_id=0,
                kv_dtype=torch.bfloat16,
                kv_shape=(28, 2, 256, 8, 128),
            )
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
                metadata=metadata,
            )

            key_a = create_test_key(0)
            key_b = create_test_key(1)
            path_a = os.path.dirname(backend._key_to_path(key_a))
            path_b = os.path.dirname(backend._key_to_path(key_b))
            assert path_a != path_b

            backend.dict[key_a] = DiskCacheMetadata(
                backend._key_to_path(key_a), size, None, None, None, None, 0
            )
            backend.dict[key_b] = DiskCacheMetadata(
                backend._key_to_path(key_b), size, None, None, None, None, 0
            )
            backend.path_current_cache_sizes[path_a] = size
            backend.path_current_cache_sizes[path_b] = size
            backend.current_cache_size = size * 2
            backend.path_dicts[path_a][key_a] = backend.dict[key_a]
            backend.path_dicts[path_b][key_b] = backend.dict[key_b]
            backend.path_cache_policies[path_a].update_on_put(key_a)
            backend.path_cache_policies[path_b].update_on_put(key_b)

            memory_obj = MagicMock(spec=MemoryObj)
            memory_obj.tensor = torch.zeros(1)
            memory_obj.get_physical_size.return_value = size
            memory_obj.ref_count_up.return_value = None

            backend.disk_worker.submit_task = MagicMock(return_value=asyncio.sleep(0))

            with patch(
                "lmcache.v1.storage_backend.local_disk_backend."
                "asyncio.run_coroutine_threadsafe"
            ) as mock_schedule:

                def _close_coro(coro, _loop):
                    coro.close()
                    return MagicMock()

                mock_schedule.side_effect = _close_coro
                backend.submit_put_task(key_a, memory_obj)

            assert key_a not in backend.dict
            assert key_b in backend.dict
            assert backend.path_current_cache_sizes[path_a] == size
            assert backend.path_current_cache_sizes[path_b] == size
            assert backend.disk_worker.submit_task.call_args.kwargs["key"] == key_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_worker_aware_subset_assignment(self, async_loop, local_cpu_backend):
        """With metadata, a worker is assigned a contiguous subset of paths."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            combined = ",".join(dirs)
            config = create_test_config(combined)
            # 2 local workers, 4 paths: rank 1 owns the second half.
            metadata = LMCacheMetadata(
                model_name="test_model",
                world_size=2,
                local_world_size=2,
                worker_id=1,
                local_worker_id=1,
                kv_dtype=torch.bfloat16,
                kv_shape=(28, 2, 256, 8, 128),
            )
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:1",
                metadata=metadata,
            )
            assert backend.assigned_paths == dirs[2:4]
            assert backend.path == dirs[2]
            # _key_to_path must land under one of the assigned dirs.
            key = create_test_key(7)
            assert os.path.dirname(backend._key_to_path(key)) in dirs[2:4]
            # Block size map keyed on the assigned paths only.
            assert set(backend.path_block_sizes) == set(dirs[2:4])
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_no_metadata_single_path_legacy(self, async_loop, local_cpu_backend):
        """Without metadata, the legacy single-path mapping is used."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            combined = ",".join(dirs)
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:2",
                metadata=None,
            )
            # device_id=2 -> 2 % 4 = 2 -> dirs[2], single path.
            assert backend.assigned_paths == [dirs[2]]
            assert backend.path == dirs[2]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
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
        path = backend._key_to_path(key)
        cache_path = os.path.dirname(path)
        meta = DiskCacheMetadata(
            path=path,
            size=0,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=MemoryFormat.KV_2LTD,
            pin_count=0,
        )
        with backend.disk_lock:
            backend.dict[key] = meta
            backend.path_dicts[cache_path][key] = meta
            backend.path_cache_policies[cache_path].update_on_put(key)

    def test_no_phantom_hit_when_load_fails(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must NOT be called when load_bytes_from_disk returns None."""
        key = create_test_key(101)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        cache_path = local_disk_backend._get_cache_path_for_key(key)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=None
        ):
            with patch.object(
                local_disk_backend.path_cache_policies[cache_path], "update_on_hit"
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
        cache_path = local_disk_backend._get_cache_path_for_key(key)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=fake_memory_obj
        ):
            with patch.object(
                local_disk_backend.path_cache_policies[cache_path], "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is fake_memory_obj
        mock_update.assert_called_once_with(
            key, local_disk_backend.path_dicts[cache_path]
        )
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_absent_returns_none_without_policy_update(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """get_blocking must return None immediately when the key is not cached."""
        key = create_test_key(103)

        with patch.object(
            local_disk_backend.path_cache_policies[local_disk_backend.path],
            "update_on_hit",
        ) as mock_update:
            result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()
