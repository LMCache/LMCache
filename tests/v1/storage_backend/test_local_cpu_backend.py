# SPDX-License-Identifier: Apache-2.0
# Standard
import logging
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import BatchedKVOperationMsg, OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from tests.v1.utils import create_test_memory_obj
import lmcache.v1.storage_backend.local_cpu_backend as local_cpu_backend_module


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
        self._lock = threading.Lock()

    def put_msg(self, msg):
        with self._lock:
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
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def create_test_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = create_test_config()

    # Initialize PinMonitor before creating backend
    PinMonitor.GetOrCreate(config)

    backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

    yield backend

    # Cleanup: destroy PinMonitor after test
    PinMonitor.DestroyInstance()


@pytest.fixture
def local_cpu_backend_disabled(memory_allocator):
    """Create a LocalCPUBackend with local_cpu disabled."""
    config = create_test_config(local_cpu=False)

    # Initialize PinMonitor before creating backend
    PinMonitor.GetOrCreate(config)

    backend = LocalCPUBackend(config=config, memory_allocator=memory_allocator)

    yield backend

    # Cleanup: destroy PinMonitor after test
    PinMonitor.DestroyInstance()


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
        lmcache_worker = MockLMCacheWorker()

        backend = LocalCPUBackend(
            config=config,
            memory_allocator=memory_allocator,
            lmcache_worker=lmcache_worker,
        )

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
        memory_obj1 = create_test_memory_obj(shape=torch.Size([2, 16, 8, 128]))
        memory_obj2 = create_test_memory_obj(shape=torch.Size([2, 32, 8, 128]))

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

    def test_remove_with_worker(self, memory_allocator, lmcache_engine_metadata):
        """Test remove() with LMCacheWorker."""
        config = create_test_config()
        lmcache_worker = MockLMCacheWorker()

        backend = LocalCPUBackend(
            config=config,
            metadata=lmcache_engine_metadata,
            memory_allocator=memory_allocator,
            lmcache_worker=lmcache_worker,
        )

        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        backend.submit_put_task(key, memory_obj)

        # Remove the key
        backend.remove(key)

        # Manually flush to ensure messages are sent for testing
        if backend.batched_msg_sender is not None:
            backend.batched_msg_sender.flush()

        # Check that we have batched messages
        batched_msgs = [
            msg
            for msg in lmcache_worker.messages
            if isinstance(msg, BatchedKVOperationMsg)
        ]
        assert len(batched_msgs) >= 1, "Should have at least one batched message"

        # Collect all operations from all batches
        all_admit_ops = []
        all_evict_ops = []
        for msg in batched_msgs:
            for op in msg.operations:
                if op.op_type == OpType.ADMIT:
                    all_admit_ops.append(op)
                elif op.op_type == OpType.EVICT:
                    all_evict_ops.append(op)

        # Verify we have exactly one ADMIT and one EVICT operation
        assert len(all_admit_ops) == 1, "Should have exactly one ADMIT operation"
        assert len(all_evict_ops) == 1, "Should have exactly one EVICT operation"

        # Verify the operations are for the correct key
        assert all_admit_ops[0].key == key.chunk_hash
        assert all_evict_ops[0].key == key.chunk_hash

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


class TestCPUEvictionProportionalSpace:
    """Verify that allocate() evicts proportionally to the requested size (#1939).

    Before the fix, the eviction loop removed 1 entry per outer-loop iteration,
    causing O(required_size / avg_entry_size) retries (each producing a warning).
    After the fix, the inner eviction loop accumulates freed_bytes >= target_bytes
    before retrying the allocation, reducing the outer loop to O(1) retries.
    """

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_cpu_eviction_frees_proportional_space(self, memory_allocator):
        """Fill the hot_cache, then request an allocation that requires eviction.

        Assert that multiple entries are evicted in a single allocate() call
        (not one-at-a-time across many retries).
        """
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        try:
            backend = LocalCPUBackend(
                config=config, memory_allocator=memory_allocator
            )

            # Shape: [2, 16, 8, 128] = 65536 bytes (bfloat16) = 64 KiB
            entry_shape = torch.Size([2, 16, 8, 128])
            entry_dtype = torch.bfloat16
            num_entries = 20

            # Fill the hot_cache with evictable entries.
            # submit_put_task bumps ref_count from 1->2; we call
            # ref_count_down on the original to leave only the cache's
            # reference (ref_count=1), making the entry evictable.
            for i in range(num_entries):
                key = create_test_key(f"fill_{i}")
                mem_obj = create_test_memory_obj(
                    shape=entry_shape, dtype=entry_dtype
                )
                backend.submit_put_task(key, mem_obj)
                mem_obj.ref_count_down()  # caller releases its reference

            assert len(backend.hot_cache) == num_entries
            # Verify entries are evictable
            for v in backend.hot_cache.values():
                assert v.can_evict, (
                    f"Entry should be evictable (ref_count={v.get_ref_count()})"
                )

            # Patch allocator: first call fails (simulates full pool),
            # second call succeeds (after eviction freed space).
            real_allocate = memory_allocator.allocate
            call_count = [0]

            def patched_allocate(shapes, dtypes, fmt):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return None  # Pre-loop call fails
                return real_allocate(shapes, dtypes, fmt)

            backend.memory_allocator.allocate = patched_allocate

            # Request an allocation to trigger the eviction loop.
            #   alloc_bytes = 65536, target_bytes = max(131072, 32 MiB) = 32 MiB
            # Each entry contributes 64 KiB via get_size() (phy_size=0 for AdHoc).
            # The loop will exhaust all 20 entries (20*64K = 1.25 MiB < 32 MiB).
            # Key: all evictions happen in ONE outer-loop iteration.
            result = backend.allocate(
                entry_shape, entry_dtype, eviction=True, busy_loop=False
            )

            assert result is not None, (
                "allocate() should succeed after proportional eviction"
            )
            # Exactly 2 allocator calls: 1 initial (None) + 1 retry (success)
            assert call_count[0] == 2, (
                f"Expected 2 allocator calls (1 initial + 1 after eviction), "
                f"got {call_count[0]}. The eviction loop may not be freeing "
                f"enough in one pass."
            )
            # All entries evicted (target > total available)
            assert len(backend.hot_cache) == 0, (
                f"Expected all entries evicted, "
                f"got {len(backend.hot_cache)} remaining"
            )

            backend.memory_allocator.allocate = real_allocate
            backend.memory_allocator.close()
        finally:
            PinMonitor.DestroyInstance()

    def test_cpu_eviction_stops_when_target_reached(self, memory_allocator):
        """Verify that eviction stops once enough bytes are freed,
        without evicting the entire cache unnecessarily.

        Use entries with known phy_size > 0 to control when the target is hit.
        """
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        try:
            backend = LocalCPUBackend(
                config=config, memory_allocator=memory_allocator
            )

            # target_bytes = max(alloc_bytes * 2, 32 MiB) = 32 MiB
            # With phy_size = 8 MiB per entry, need ceil(32/8) = 4 evictions.
            entry_shape = torch.Size([2, 16, 8, 128])
            entry_dtype = torch.bfloat16
            large_phy_size = 8 * 1024 * 1024  # 8 MiB
            num_entries = 10

            for i in range(num_entries):
                key = create_test_key(f"large_{i}")
                mem_obj = create_test_memory_obj(
                    shape=entry_shape, dtype=entry_dtype
                )
                # Patch phy_size to simulate large entries
                mem_obj.metadata.phy_size = large_phy_size
                backend.submit_put_task(key, mem_obj)
                mem_obj.ref_count_down()  # caller releases its reference

            assert len(backend.hot_cache) == num_entries

            # Patch allocator: first call fails, second succeeds
            real_allocate = memory_allocator.allocate
            call_count = [0]

            def patched_allocate(shapes, dtypes, fmt):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return None
                return real_allocate(shapes, dtypes, fmt)

            backend.memory_allocator.allocate = patched_allocate

            result = backend.allocate(
                entry_shape, entry_dtype, eviction=True, busy_loop=False
            )

            assert result is not None
            assert call_count[0] == 2, (
                f"Expected 2 allocator calls, got {call_count[0]}"
            )

            # With 8 MiB entries and 32 MiB target, exactly 4 should be
            # evicted: freed_bytes reaches 32 MiB after 4 evictions.
            expected_remaining = num_entries - 4
            assert len(backend.hot_cache) == expected_remaining, (
                f"Expected {expected_remaining} entries remaining, "
                f"got {len(backend.hot_cache)}. Eviction should stop "
                f"once target_bytes is reached."
            )

            backend.memory_allocator.allocate = real_allocate
            backend.memory_allocator.close()
        finally:
            PinMonitor.DestroyInstance()


class TestLocalCPUBackendAllocatorAlignment:
    def test_rust_odirect_auto_alignment_for_mixed_allocator(self, monkeypatch):
        config = create_test_config(local_cpu=True)
        config.max_local_cpu_size = 0.01
        config.extra_config = {
            "rust_raw_block.device_path": "/tmp/dev.bin",
            "rust_raw_block.use_odirect": True,
            "rust_raw_block.block_align": 4096,
        }
        metadata = create_test_metadata()

        captured: dict[str, object] = {}

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured["size"] = size
                captured["kwargs"] = kwargs
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            kwargs = captured["kwargs"]
            assert isinstance(kwargs, dict)
            assert kwargs.get("align_bytes") == 4096
        finally:
            backend.memory_allocator.close()

    def test_explicit_alignment_override_for_mixed_allocator(self, monkeypatch):
        config = create_test_config(local_cpu=True)
        config.max_local_cpu_size = 0.01
        config.extra_config = {
            "local_cpu.pinned_align_bytes": 4096,
            "rust_raw_block.device_path": "/tmp/dev.bin",
            "rust_raw_block.use_odirect": False,
        }
        metadata = create_test_metadata()

        captured: dict[str, object] = {}

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured["size"] = size
                captured["kwargs"] = kwargs
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            kwargs = captured["kwargs"]
            assert isinstance(kwargs, dict)
            assert kwargs.get("align_bytes") == 4096
        finally:
            backend.memory_allocator.close()


class TestAllocateMaxAttemptsGuard:
    """Verify that allocate() and batched_allocate() give up after
    MAX_ALLOC_ATTEMPTS iterations instead of looping forever (#1939).
    """

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_allocate_gives_up_after_max_attempts(
        self, memory_allocator
    ):
        """When the allocator ALWAYS returns None, allocate() must:
        - return None (not hang forever)
        - complete in < 10 seconds
        - log the "CPU allocation failed after N attempts" error message
        """
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        try:
            backend = LocalCPUBackend(
                config=config, memory_allocator=memory_allocator
            )

            # Save and patch allocator to always return None.
            # The memory_allocator fixture is session-scoped, so we MUST
            # restore the original after the test.
            real_allocate = memory_allocator.allocate
            backend.memory_allocator.allocate = lambda *a, **kw: None

            shape = torch.Size([2, 16, 8, 128])
            dtype = torch.bfloat16

            # LMCache loggers set propagate=False, so caplog cannot
            # capture via the root logger.  Attach a temporary handler
            # directly to the module logger.
            cpu_logger = logging.getLogger(
                "lmcache.v1.storage_backend.local_cpu_backend"
            )
            captured_records: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = lambda record: captured_records.append(record)
            handler.setLevel(logging.ERROR)
            cpu_logger.addHandler(handler)
            try:
                t0 = time.monotonic()
                result = backend.allocate(
                    shape, dtype, eviction=True, busy_loop=True
                )
                elapsed = time.monotonic() - t0
            finally:
                cpu_logger.removeHandler(handler)
                # Restore the real allocate on the wrapper
                backend.memory_allocator.allocate = real_allocate

            assert result is None, (
                "allocate() must return None when allocation is impossible"
            )
            assert elapsed < 10.0, (
                f"allocate() took {elapsed:.1f}s; expected < 10s "
                f"(MAX_ALLOC_ATTEMPTS guard should cap it)"
            )
            # Check that the error message was logged
            assert any(
                "CPU allocation failed after" in record.getMessage()
                and record.levelno >= logging.ERROR
                for record in captured_records
            ), (
                "Expected ERROR log with 'CPU allocation failed after' "
                f"but got: {[r.getMessage() for r in captured_records]}"
            )
        finally:
            PinMonitor.DestroyInstance()

    def test_allocate_succeeds_within_retry_limit(self, memory_allocator):
        """When the allocator fails the first 5 times then succeeds,
        allocate() should return a valid MemoryObj and not hit the
        MAX_ALLOC_ATTEMPTS ceiling.
        """
        config = create_test_config()
        PinMonitor.GetOrCreate(config)
        try:
            backend = LocalCPUBackend(
                config=config, memory_allocator=memory_allocator
            )

            shape = torch.Size([2, 16, 8, 128])
            dtype = torch.bfloat16

            # Save real allocate before patching (session-scoped fixture).
            real_allocate = memory_allocator.allocate
            call_count = [0]
            fail_count = 5

            def patched_allocate(shapes, dtypes, f):
                call_count[0] += 1
                if call_count[0] <= fail_count:
                    return None
                return real_allocate(shapes, dtypes, f)

            backend.memory_allocator.allocate = patched_allocate
            try:
                result = backend.allocate(
                    shape, dtype, eviction=True, busy_loop=True
                )
            finally:
                # Restore the real allocate on the wrapper
                backend.memory_allocator.allocate = real_allocate

            assert result is not None, (
                "allocate() should succeed when the allocator "
                "recovers within the retry limit"
            )
            assert isinstance(result, MemoryObj)

            # The initial call (before the loop) counts as call 1.
            # The loop retries until success. Total calls should be
            # fail_count + 1 (the successful one), and the loop
            # iteration count should be well below MAX_ALLOC_ATTEMPTS.
            max_attempts = getattr(
                local_cpu_backend_module, "MAX_ALLOC_ATTEMPTS", 50
            )
            assert call_count[0] < max_attempts, (
                f"allocate() used {call_count[0]} allocator calls, "
                f"expected fewer than MAX_ALLOC_ATTEMPTS ({max_attempts})"
            )
        finally:
            PinMonitor.DestroyInstance()

    @pytest.mark.no_shared_allocator
    def test_batched_allocate_gives_up_after_max_attempts(self):
        """Same as allocate test but for batched_allocate().

        When the allocator ALWAYS returns None, batched_allocate() must:
        - return None (not hang forever)
        - complete in < 10 seconds
        - log the "CPU allocation failed after N attempts" error message

        Uses a real MixedMemoryAllocator (not the session-scoped wrapper)
        because batched_allocate's eviction path asserts
        isinstance(memory_allocator, MixedMemoryAllocator).
        """
        from lmcache.v1.memory_management import MixedMemoryAllocator

        config = create_test_config()
        # Small allocator -- just enough to construct the backend.
        alloc = MixedMemoryAllocator(64 * 1024 * 1024)  # 64 MiB
        PinMonitor.GetOrCreate(config)
        try:
            backend = LocalCPUBackend(
                config=config, memory_allocator=alloc
            )

            # Patch batched_allocate to always return None.
            real_batched = alloc.batched_allocate
            alloc.batched_allocate = lambda *a, **kw: None

            shape = torch.Size([2, 16, 8, 128])
            dtype = torch.bfloat16
            batch_size = 4

            # Attach a temporary handler to capture ERROR logs
            cpu_logger = logging.getLogger(
                "lmcache.v1.storage_backend.local_cpu_backend"
            )
            captured_records: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = lambda record: captured_records.append(record)
            handler.setLevel(logging.ERROR)
            cpu_logger.addHandler(handler)
            try:
                t0 = time.monotonic()
                result = backend.batched_allocate(
                    shape, dtype, batch_size,
                    eviction=True, busy_loop=True,
                )
                elapsed = time.monotonic() - t0
            finally:
                cpu_logger.removeHandler(handler)
                alloc.batched_allocate = real_batched

            assert result is None, (
                "batched_allocate() must return None when allocation "
                "is impossible"
            )
            assert elapsed < 10.0, (
                f"batched_allocate() took {elapsed:.1f}s; expected < 10s "
                f"(MAX_ALLOC_ATTEMPTS guard should cap it)"
            )
            assert any(
                "CPU allocation failed after" in record.getMessage()
                and record.levelno >= logging.ERROR
                for record in captured_records
            ), (
                "Expected ERROR log with 'CPU allocation failed after' "
                f"but got: {[r.getMessage() for r in captured_records]}"
            )
        finally:
            PinMonitor.DestroyInstance()
            alloc.close()


class TestP2PMLAFormatResolution:
    """Regression tests for the P2P + MLA format hardcode.

    Prior to the fix, `initialize_allocator` hardcoded
    `fmt=MemoryFormat.KV_2LTD` in the P2P branch regardless of
    `metadata.use_mla`. The P2PBackend itself uses `KV_MLA_FMT` for MLA
    models, causing `token_dim()` to diverge between the allocator and
    the backend and breaking unfull-chunk shape adjustment.
    """

    def _make_p2p_config(self) -> LMCacheEngineConfig:
        """Build a minimal P2P-enabled config that bypasses strict
        validation. `_validate_config` requires controller URLs and port
        lists, but `initialize_allocator` only reads `enable_p2p`,
        `max_local_cpu_size`, and a few others — so we build a relaxed
        config via from_defaults and flip `enable_p2p` directly."""
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,  # 100 MiB — enough for a couple pages
            lmcache_instance_id="test_instance",
        )
        object.__setattr__(config, "enable_p2p", True)
        return config

    def _make_metadata(self, use_mla: bool) -> LMCacheMetadata:
        return LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
            use_mla=use_mla,
        )

    def _capture_fmt(self, monkeypatch):
        """Patch PagedCpuGpuMemoryAllocator.init_cpu_memory_allocator to
        capture the fmt arg without actually allocating."""
        captured = {}

        def fake_init(self, size, shapes, dtypes, fmt, numa_mapping=None):
            captured["fmt"] = fmt
            captured["shapes"] = shapes
            captured["dtypes"] = dtypes
            # stub the minimum state the backend reads post-init
            self.cpu_allocator = type("StubCpu", (), {
                "shapes": shapes, "dtypes": dtypes,
                "buffer_ptr": 0, "buffer_size": 0, "align_bytes": 1,
            })()

        from lmcache.v1.memory_management import PagedCpuGpuMemoryAllocator
        monkeypatch.setattr(
            PagedCpuGpuMemoryAllocator,
            "init_cpu_memory_allocator",
            fake_init,
        )
        return captured

    def test_p2p_with_mla_uses_kv_mla_fmt(self, monkeypatch):
        """MLA model → allocator must init with KV_MLA_FMT."""
        captured = self._capture_fmt(monkeypatch)
        config = self._make_p2p_config()
        metadata = self._make_metadata(use_mla=True)

        PinMonitor.GetOrCreate(config)
        try:
            LocalCPUBackend(config=config, metadata=metadata)
        finally:
            PinMonitor.DestroyInstance()

        assert captured.get("fmt") == MemoryFormat.KV_MLA_FMT, (
            f"P2P + MLA should init allocator with KV_MLA_FMT, "
            f"got {captured.get('fmt')}"
        )

    def test_p2p_without_mla_uses_kv_2ltd(self, monkeypatch):
        """Non-MLA model → allocator must init with KV_2LTD (backward
        compat for existing P2P deployments)."""
        captured = self._capture_fmt(monkeypatch)
        config = self._make_p2p_config()
        metadata = self._make_metadata(use_mla=False)

        PinMonitor.GetOrCreate(config)
        try:
            LocalCPUBackend(config=config, metadata=metadata)
        finally:
            PinMonitor.DestroyInstance()

        assert captured.get("fmt") == MemoryFormat.KV_2LTD, (
            f"P2P + non-MLA should init allocator with KV_2LTD, "
            f"got {captured.get('fmt')}"
        )
