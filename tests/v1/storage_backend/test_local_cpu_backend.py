# SPDX-License-Identifier: Apache-2.0
# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


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
    local_cpu: bool = True, use_layerwise: bool = False, enable_blending: bool = False
):
    """Create a test configuration for LocalCPUBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=local_cpu,
        use_layerwise=use_layerwise,
        enable_blending=enable_blending,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, hash(key_id))


def create_test_memory_obj(shape=(2, 16, 8, 128), dtype=torch.bfloat16) -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = create_test_config()
    return LocalCPUBackend(config=config, memory_allocator=memory_allocator)


@pytest.fixture
def local_cpu_backend_disabled(memory_allocator):
    """Create a LocalCPUBackend with local_cpu disabled."""
    config = create_test_config(local_cpu=False)
    return LocalCPUBackend(config=config, memory_allocator=memory_allocator)


class TestLocalCPUBackend:
    """Test cases for LocalCPUBackend."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_init(self, memory_allocator):
        """Test LocalCPUBackend initialization."""
        config = create_test_config()
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        assert backend.use_hot is True
        assert backend.lookup_server is None
        assert backend.memory_allocator == memory_allocator
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert len(backend.hot_cache) == 0
        assert backend.layerwise is False
        assert backend.enable_blending is False

        memory_allocator.close()

    def test_init_with_lookup_server_and_worker(self, memory_allocator):
        """Test LocalCPUBackend initialization with lookup server and worker."""
        config = create_test_config()
        lookup_server = MockLookupServer()
        lmcache_worker = MockLMCacheWorker()

        backend = LocalCPUBackend(
            config=config,
            memory_allocator=memory_allocator,
            lookup_server=lookup_server,
            lmcache_worker=lmcache_worker,
        )

        assert backend.lookup_server == lookup_server
        assert backend.lmcache_worker == lmcache_worker

        memory_allocator.close()

    def test_init_with_layerwise_config(self, memory_allocator):
        """Test LocalCPUBackend initialization with layerwise configuration."""
        config = create_test_config(use_layerwise=True, enable_blending=True)
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        assert backend.layerwise is True
        assert backend.enable_blending is True

        memory_allocator.close()

    def test_str(self, local_cpu_backend):
        """Test string representation."""
        assert str(local_cpu_backend) == "LocalCPUBackend"

        local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_cpu_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key("nonexistent")
        assert not local_cpu_backend.contains(key)
        assert not local_cpu_backend.contains(key, pin=True)

        local_cpu_backend.memory_allocator.close()

    def test_contains_key_exists(self, local_cpu_backend):
        """Test contains() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        assert local_cpu_backend.contains(key)
        assert local_cpu_backend.contains(key, pin=True)

        local_cpu_backend.memory_allocator.close()

    def test_exists_in_put_tasks(self, local_cpu_backend):
        """Test exists_in_put_tasks()."""
        key = create_test_key("test_key")
        # LocalCPUBackend always returns False for exists_in_put_tasks
        assert not local_cpu_backend.exists_in_put_tasks(key)
        local_cpu_backend.memory_allocator.close()

    def test_submit_put_task(self, local_cpu_backend):
        """Test submit_put_task()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        future = local_cpu_backend.submit_put_task(key, memory_obj)

        # LocalCPUBackend returns None for submit_put_task
        assert future is None
        assert key in local_cpu_backend.hot_cache
        assert local_cpu_backend.hot_cache[key] == memory_obj
        assert (
            memory_obj.get_ref_count() == 2
        )  # 1 from creation + 1 from submit_put_task
        local_cpu_backend.memory_allocator.close()

    def test_submit_put_task_reinsert(self, local_cpu_backend):
        """Test submit_put_task() with reinsertion."""
        key = create_test_key("test_key")
        memory_obj1 = create_test_memory_obj(shape=(2, 16, 8, 128))
        memory_obj2 = create_test_memory_obj(shape=(2, 32, 8, 128))

        # First insertion
        local_cpu_backend.submit_put_task(key, memory_obj1)
        assert local_cpu_backend.hot_cache[key] == memory_obj1

        # Reinsertion
        local_cpu_backend.submit_put_task(key, memory_obj2)
        assert local_cpu_backend.hot_cache[key] != memory_obj2
        assert memory_obj1.get_ref_count() == 2
        assert memory_obj2.get_ref_count() == 1

        local_cpu_backend.memory_allocator.close()

    def test_batched_submit_put_task(self, local_cpu_backend):
        """Test batched_submit_put_task()."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        futures = local_cpu_backend.batched_submit_put_task(keys, memory_objs)

        # LocalCPUBackend returns None for batched_submit_put_task
        assert futures is None

        # Check that all keys were inserted
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            assert key in local_cpu_backend.hot_cache
            assert local_cpu_backend.hot_cache[key] == memory_obj

        local_cpu_backend.memory_allocator.close()

    def test_batched_submit_put_task_disabled(self, local_cpu_backend_disabled):
        """Test batched_submit_put_task() when local_cpu is disabled."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        futures = local_cpu_backend_disabled.batched_submit_put_task(keys, memory_objs)

        # Should return None when local_cpu is disabled
        assert futures is None

        local_cpu_backend_disabled.memory_allocator.close()

    def test_submit_prefetch_task(self, local_cpu_backend):
        """Test submit_prefetch_task()."""
        key = create_test_key("test_key")
        ret = local_cpu_backend.submit_prefetch_task(key)

        # LocalCPUBackend always returns None for submit_prefetch_task
        assert ret is False

        local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_cpu_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key("nonexistent")
        result = local_cpu_backend.get_blocking(key)

        assert result is None

        local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_exists(self, local_cpu_backend):
        """Test get_blocking() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        result = local_cpu_backend.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result == memory_obj
        assert (
            result.get_ref_count() == 3
        )  # 1 from creation + 1 from submit_put_task + 1 from get_blocking

        local_cpu_backend.memory_allocator.close()

    def test_get_non_blocking_key_not_exists(self, local_cpu_backend):
        """Test get_non_blocking() when key doesn't exist."""
        key = create_test_key("nonexistent")
        future = local_cpu_backend.get_non_blocking(key)

        assert future is None

        local_cpu_backend.memory_allocator.close()

    def test_get_non_blocking_key_exists(self, local_cpu_backend):
        """Test get_non_blocking() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        future = local_cpu_backend.get_non_blocking(key)

        assert future is not None
        result = future.result()
        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result == memory_obj

        local_cpu_backend.memory_allocator.close()

    def test_pin_unpin(self, local_cpu_backend):
        """Test pin() and unpin() operations."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test pin
        assert local_cpu_backend.pin(key)
        assert memory_obj.is_pinned

        # Test unpin
        assert local_cpu_backend.unpin(key)
        assert not memory_obj.is_pinned

        # Test pin/unpin non-existent key
        non_existent_key = create_test_key("non_existent")
        assert not local_cpu_backend.pin(non_existent_key)
        assert not local_cpu_backend.unpin(non_existent_key)

        local_cpu_backend.memory_allocator.close()

    def test_remove(self, local_cpu_backend):
        """Test remove()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_cpu_backend.submit_put_task(key, memory_obj)
        assert key in local_cpu_backend.hot_cache

        # Remove the key
        result = local_cpu_backend.remove(key)

        assert result is True
        assert key not in local_cpu_backend.hot_cache
        assert memory_obj.get_ref_count() == 1  # Should be decremented

        local_cpu_backend.memory_allocator.close()

    def test_remove_non_existent(self, local_cpu_backend):
        """Test remove() with non-existent key."""
        key = create_test_key("nonexistent")
        result = local_cpu_backend.remove(key)

        assert result is False

        local_cpu_backend.memory_allocator.close()

    def test_remove_with_worker(self, memory_allocator):
        """Test remove() with LMCacheWorker."""
        config = create_test_config()
        lmcache_worker = MockLMCacheWorker()
        backend = LocalCPUBackend(
            config=config,
            memory_allocator=memory_allocator,
            lmcache_worker=lmcache_worker,
        )

        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        backend.submit_put_task(key, memory_obj)

        # Remove the key
        backend.remove(key)

        # Check that evict message was sent
        assert len(lmcache_worker.messages) == 2  # 1 admit + 1 evict
        # First Party
        from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg

        assert any(isinstance(msg, KVAdmitMsg) for msg in lmcache_worker.messages)
        assert any(isinstance(msg, KVEvictMsg) for msg in lmcache_worker.messages)

        memory_allocator.close()

    def test_allocate(self, local_cpu_backend):
        """Test allocate()."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16

        memory_obj = local_cpu_backend.allocate(shape, dtype)

        assert memory_obj is not None
        assert isinstance(memory_obj, MemoryObj)
        assert memory_obj.metadata.shape == shape
        assert memory_obj.metadata.dtype == dtype

        local_cpu_backend.memory_allocator.close()

    def test_allocate_with_format(self, local_cpu_backend):
        """Test allocate() with specific format."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD

        memory_obj = local_cpu_backend.allocate(shape, dtype, fmt)

        assert memory_obj is not None
        assert memory_obj.metadata.fmt == fmt

        local_cpu_backend.memory_allocator.close()

    def test_allocate_with_layerwise_config(self, memory_allocator):
        """Test allocate() with layerwise configuration."""
        config = create_test_config(use_layerwise=True, enable_blending=True)
        backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16

        memory_obj = backend.allocate(shape, dtype)

        assert memory_obj is not None
        # Should use KV_2TD format when layerwise=True and enable_blending=True
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD

        memory_allocator.close()

    def test_batched_allocate(self, local_cpu_backend):
        """Test batched_allocate()."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        batch_size = 3

        memory_objs = local_cpu_backend.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is not None
        assert len(memory_objs) == batch_size
        for memory_obj in memory_objs:
            assert isinstance(memory_obj, MemoryObj)
            assert memory_obj.metadata.shape == shape
            assert memory_obj.metadata.dtype == dtype

        local_cpu_backend.memory_allocator.close()

    def test_get_keys(self, local_cpu_backend):
        """Test get_keys()."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Insert keys
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            local_cpu_backend.submit_put_task(key, memory_obj)

        # Get keys
        retrieved_keys = local_cpu_backend.get_keys()

        assert len(retrieved_keys) == 3
        assert all(key in retrieved_keys for key in keys)

        local_cpu_backend.memory_allocator.close()

    def test_get_keys_empty(self, local_cpu_backend):
        """Test get_keys() when cache is empty."""
        keys = local_cpu_backend.get_keys()

        assert len(keys) == 0

        local_cpu_backend.memory_allocator.close()

    def test_concurrent_access(self, local_cpu_backend):
        """Test concurrent access to the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test concurrent contains() calls
        def check_contains():
            for _ in range(20):
                assert local_cpu_backend.contains(key)

        threads = [threading.Thread(target=check_contains) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        local_cpu_backend.memory_allocator.close()

    def test_thread_safety(self, local_cpu_backend):
        """Test thread safety of the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)

        # Test concurrent operations
        def concurrent_operations():
            for _ in range(10):
                # Test contains
                local_cpu_backend.contains(key)
                # Test pin/unpin
                local_cpu_backend.pin(key)
                local_cpu_backend.unpin(key)
                # Test get_blocking
                result = local_cpu_backend.get_blocking(key)
                assert result is not None

        threads = [threading.Thread(target=concurrent_operations) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # The backend should still be in a consistent state
        assert local_cpu_backend.contains(key)

        local_cpu_backend.memory_allocator.close()

    def test_ref_count_management(self, local_cpu_backend):
        """Test reference count management."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        initial_ref_count = memory_obj.get_ref_count()

        # Insert key
        local_cpu_backend.submit_put_task(key, memory_obj)
        assert memory_obj.get_ref_count() == initial_ref_count + 1

        # Get blocking
        local_cpu_backend.get_blocking(key)
        assert memory_obj.get_ref_count() == initial_ref_count + 2

        # Remove key
        local_cpu_backend.remove(key)
        assert memory_obj.get_ref_count() == initial_ref_count + 1
        local_cpu_backend.memory_allocator.close()
