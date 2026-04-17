# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from typing import Callable

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend import local_disk_backend as local_disk_backend_module
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


def wait_for_metadata(
    meta_path: str,
    predicate: Callable[[dict[str, object]], bool],
    timeout_s: float = 2.0,
) -> dict[str, object]:
    """Poll a metadata sidecar until it matches the expected state."""
    deadline = time.time() + timeout_s
    last_payload = None
    while time.time() < deadline:
        with open(meta_path, "r", encoding="utf-8") as f:
            last_payload = json.load(f)
        if predicate(last_payload):
            return last_payload
        time.sleep(0.01)

    pytest.fail(f"Timed out waiting for metadata update in {meta_path}: {last_payload}")


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
    """Create a running asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


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
    backend = LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda",
    )
    yield backend
    local_cpu_backend.memory_allocator.close()


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        assert backend.dst_device == "cuda"
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
            dst_device="cuda",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

    def test_get_blocking_returns_none_when_cpu_staging_is_exhausted(
        self, local_disk_backend, monkeypatch
    ):
        """Disk load should degrade to miss instead of busy-looping forever."""
        key = create_test_key(21)
        payload = b"staging"
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(payload)

        local_disk_backend.insert_key(
            key,
            size=len(payload),
            shape=torch.Size([len(payload)]),
            dtype=torch.uint8,
            fmt=MemoryFormat.BINARY,
        )

        monkeypatch.setattr(
            local_disk_backend.local_cpu_backend,
            "allocate",
            lambda *args, **kwargs: None,
        )

        assert local_disk_backend.get_blocking(key) is None

    def test_recover_persisted_index_restores_lru_order(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Recovered entries should preserve LRU eviction order."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        key_old = create_test_key(10)
        key_new = create_test_key(11)
        payload = b"lmcache"
        for key, created_ts, last_access_ts, hit_count in [
            (key_old, 10.0, 20.0, 2),
            (key_new, 11.0, 30.0, 5),
        ]:
            data_path = backend._key_to_path(key)
            with open(data_path, "wb") as f:
                f.write(payload)
            with open(backend._key_to_meta_path(key), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "key": key.to_string(),
                        "size": len(payload),
                        "shape": None,
                        "dtype": None,
                        "fmt": None,
                        "cached_positions": None,
                        "shapes": None,
                        "dtypes": None,
                        "created_ts": created_ts,
                        "last_access_ts": last_access_ts,
                        "hit_count": hit_count,
                    },
                    f,
                )

        recovered_backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        recovered_keys = list(recovered_backend.dict.keys())
        assert recovered_keys == [key_old, key_new]
        evict_candidates = recovered_backend.cache_policy.get_evict_candidates(
            recovered_backend.dict,
            num_candidates=1,
        )
        assert evict_candidates == [key_old]

        local_cpu_backend.memory_allocator.close()

    def test_touch_cache_persists_hit_metadata(self, local_disk_backend):
        """Pinned cache hits should eventually flush updated access metadata."""
        key = create_test_key(30)
        local_disk_backend.insert_key(
            key,
            size=16,
            shape=torch.Size([16]),
            dtype=torch.uint8,
            fmt=MemoryFormat.BINARY,
        )

        meta_path = local_disk_backend._key_to_meta_path(key)
        with open(meta_path, "r", encoding="utf-8") as f:
            initial_payload = json.load(f)

        assert local_disk_backend.contains(key, pin=True)
        local_disk_backend.touch_cache()

        updated_payload = wait_for_metadata(
            meta_path,
            lambda payload: payload["hit_count"] == 2,
        )
        assert updated_payload["last_access_ts"] >= initial_payload["last_access_ts"]

    def test_recover_persisted_index_restores_lru_reuse_tracking(
        self, temp_disk_path, async_loop, local_cpu_backend, monkeypatch
    ):
        """Recovered LRU entries should count the first post-restart hit as reuse."""
        config = create_test_config(temp_disk_path)
        key = create_test_key(31)
        payload = b"reuse"

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )
        data_path = backend._key_to_path(key)
        with open(data_path, "wb") as f:
            f.write(payload)
        with open(backend._key_to_meta_path(key), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key.to_string(),
                    "size": len(payload),
                    "shape": None,
                    "dtype": None,
                    "fmt": None,
                    "cached_positions": None,
                    "shapes": None,
                    "dtypes": None,
                    "created_ts": 10.0,
                    "last_access_ts": 20.0,
                    "hit_count": 2,
                },
                f,
            )

        recovered_backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        reuse_intervals: list[float] = []
        monkeypatch.setattr(
            recovered_backend.cache_policy.stats_monitor,
            "on_chunk_reuse",
            lambda interval: reuse_intervals.append(interval),
        )

        assert recovered_backend.contains(key, pin=True)
        recovered_backend.touch_cache()

        assert len(reuse_intervals) == 1
        assert reuse_intervals[0] > 0

    def test_recover_persisted_index_enforces_max_cache_size(
        self, temp_disk_path, async_loop, local_cpu_backend, monkeypatch
    ):
        """Recovery should evict old entries if the configured disk budget shrinks."""
        payload = b"x" * 1024
        config = create_test_config(
            temp_disk_path,
            max_disk_size=len(payload) / float(1024**3),
        )
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        key_old = create_test_key(40)
        key_new = create_test_key(41)
        for key, created_ts, last_access_ts in [
            (key_old, 10.0, 20.0),
            (key_new, 11.0, 30.0),
        ]:
            data_path = backend._key_to_path(key)
            with open(data_path, "wb") as f:
                f.write(payload)
            with open(backend._key_to_meta_path(key), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "key": key.to_string(),
                        "size": len(payload),
                        "shape": None,
                        "dtype": None,
                        "fmt": None,
                        "cached_positions": None,
                        "shapes": None,
                        "dtypes": None,
                        "created_ts": created_ts,
                        "last_access_ts": last_access_ts,
                        "hit_count": 1,
                    },
                    f,
                )

        logged_messages = []

        def capture_info(message, *args):
            logged_messages.append(message % args)

        monkeypatch.setattr(local_disk_backend_module.logger, "info", capture_info)
        recovered_backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        assert list(recovered_backend.dict.keys()) == [key_new]
        assert recovered_backend.current_cache_size == len(payload)
        assert recovered_backend.usage == len(payload)
        assert not os.path.exists(recovered_backend._key_to_path(key_old))
        assert not os.path.exists(recovered_backend._key_to_meta_path(key_old))
        assert any(
            message.startswith(
                f"Recovered 1 persisted disk cache entries ({len(payload)} bytes)"
            )
            for message in logged_messages
        )

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Comma-separated disk paths should remain supported."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            config = create_test_config(f"{dir_a},{dir_b}")
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            assert backend.path == dir_a
            assert backend.local_disk_path_sharding == "by_gpu"
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_multi_path_uses_device_affinity(self, async_loop, local_cpu_backend):
        """Different CUDA devices should shard onto different disk paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            config = create_test_config(f"{dir_a},{dir_b}")
            backend0 = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            backend1 = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:1",
            )

            assert backend0.path == dir_a
            assert backend1.path == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_works_as_before(
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
            temp_disk_path,
            local_disk_path_sharding="round_robin",
        )

        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

        local_cpu_backend.memory_allocator.close()
