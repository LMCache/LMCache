import pytest
import torch
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import MemoryObj
from lmcache.experimental.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.utils import CacheEngineKey

# test utilities
class MockMemoryAllocator:
    def __init__(self, max_allocations=None):
        self.ref_counts = {}
        self.max_allocations = max_allocations
        self.allocation_count = 0
        self.pin_allocator = type('MockPinAllocator', (), {'num_active_allocations': 0})()

    def ref_count_up(self, memory_obj):
        if memory_obj not in self.ref_counts:
            self.ref_counts[memory_obj] = 1
        else:
            self.ref_counts[memory_obj] += 1

    def ref_count_down(self, memory_obj):
        if memory_obj in self.ref_counts:
            self.ref_counts[memory_obj] -= 1
            if self.ref_counts[memory_obj] <= 0:
                del self.ref_counts[memory_obj]

    def get_ref_count(self, memory_obj):
        return self.ref_counts.get(memory_obj, 0)

    def allocate(self, shape, dtype):
        if self.max_allocations is not None and self.allocation_count >= self.max_allocations:
            return None
        self.allocation_count += 1
        memory_obj = create_memory_obj()
        self.ref_counts[memory_obj] = 1
        return memory_obj

    def keys(self):
        return list(self.ref_counts.keys())

def generate_random_key():
    return CacheEngineKey("vllm", "test_model", 1, 0, f"chunk_{torch.randint(0, 1000, (1,)).item()}")

def create_memory_obj():
    tensor = torch.randn(10, 10)
    memory_obj = MemoryObj(tensor)
    return memory_obj

def create_memory_allocator():
    return MockMemoryAllocator()

def test_local_cpu_backend_basic_operations():
    # setup with no lookup server
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator)

    # test contains on empty backend
    key = generate_random_key()
    assert not backend.contains(key)

    # test put and contains
    memory_obj = create_memory_obj()
    backend.put(key, memory_obj)
    assert backend.contains(key)

    # test get
    retrieved = backend.get(key)
    assert retrieved is not None
    assert retrieved.tensor is not None

    # test touch (lru ordering)
    old_key = key
    new_key = generate_random_key()
    new_obj = create_memory_obj()
    backend.put(new_key, new_obj)
    backend.touch(old_key)  # move to end

    # test remove
    assert backend.remove(old_key)
    assert not backend.contains(old_key)

    # test clear
    num_cleared = backend.clear()
    assert num_cleared == 1  # should be just new_key left
    assert not backend.contains(new_key)

def test_local_cpu_backend_ref_counting():
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator)

    key = generate_random_key()
    memory_obj = create_memory_obj()

    # initial ref count should be 1
    assert memory_allocator.get_ref_count(memory_obj) == 1

    # after put, ref count should be 2 (one for caller, one for hot cache)
    backend.put(key, memory_obj)
    assert memory_allocator.get_ref_count(memory_obj) == 2

    # after get, ref count should be 3 (added for the caller of get)
    retrieved = backend.get(key)
    assert memory_allocator.get_ref_count(memory_obj) == 3

    # after caller is done with retrieved object
    memory_allocator.ref_count_down(retrieved)
    assert memory_allocator.get_ref_count(memory_obj) == 2

    # after remove, ref count should be 1 (just the original)
    backend.remove(key)
    assert memory_allocator.get_ref_count(memory_obj) == 1

def test_local_cpu_backend_allocation_eviction():
    memory_allocator = MockMemoryAllocator(max_allocations=5)
    backend = LocalCPUBackend(memory_allocator)

    # fill the cache to capacity
    keys = []
    for i in range(5):
        key = generate_random_key()
        memory_obj = create_memory_obj()
        backend.put(key, memory_obj)
        keys.append(key)

    # try to allocate a new object - should trigger eviction
    shape = torch.Size([10, 10])
    dtype = torch.float32
    new_obj = backend.allocate(shape, dtype)

    # should have evicted the least recently used object
    assert new_obj is not None
    assert not backend.contains(keys[0])
    assert backend.contains(keys[1])

def test_local_cpu_backend_not_implemented_methods():
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator)
    key = generate_random_key()
    memory_obj = create_memory_obj()

    with pytest.raises(NotImplementedError):
        backend.exists_in_put_tasks(key)

    with pytest.raises(NotImplementedError):
        backend.submit_put_task(key, memory_obj)

    with pytest.raises(NotImplementedError):
        backend.submit_prefetch_task(key)

    with pytest.raises(NotImplementedError):
        backend.get_blocking(key)

def test_storage_manager_with_local_cpu_backend():
    config = LMCacheEngineConfig.from_defaults(local_cpu=True)
    metadata = LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        format="vllm",
        dtype=torch.float32,
        kv_shape=(32, 2, 256, 8, 128)
    )
    allocator = create_memory_allocator()

    # mock the CreateStorageBackends function
    original_create_backends = None
    try:
        import lmcache.experimental.storage_backend
        original_create_backends = lmcache.experimental.storage_backend.CreateStorageBackends

        # create a mock that returns an empty OrderedDict
        from collections import OrderedDict
        lmcache.experimental.storage_backend.CreateStorageBackends = lambda *args, **kwargs: OrderedDict()

        # create the StorageManager
        manager = StorageManager(config, metadata, allocator)

        # verify manager.hot_cache is LocalCPUBackend
        assert isinstance(manager.hot_cache, LocalCPUBackend)

        # test operations through manager
        key = generate_random_key()
        memory_obj = create_memory_obj()

        # verify hot cache is empty
        assert not manager.contains(key, ["Hot"])

        # put the object in the manager
        manager.hot_cache.put(key, memory_obj)

        # verify it's now in the hot cache
        assert manager.contains(key, ["Hot"])

        # get the object and verify it's the same
        retrieved = manager.get(key)
        assert retrieved is not None
        assert memory_allocator.get_ref_count(retrieved) > 1

        # clean up
        allocator.ref_count_down(retrieved)

        # remove the object
        assert manager.remove(key, ["Hot"]) == 1
        assert not manager.contains(key, ["Hot"])

    finally:
        # restore the original function
        if original_create_backends:
            lmcache.experimental.storage_backend.CreateStorageBackends = original_create_backends

def test_local_cpu_backend_thread_safety():
    """test concurrent operations on LocalCPUBackend to verify thread safety."""
    memory_allocator = MockMemoryAllocator(max_allocations=100)
    backend = LocalCPUBackend(memory_allocator)

    # add keys method to access the backend's keys
    def get_keys(backend):
        # this is not thread-safe, but it's just for testing
        return list(backend.hot_cache_.keys())

    # pre-populate with some items
    initial_keys = []
    for i in range(10):
        key = generate_random_key()
        memory_obj = create_memory_obj()
        backend.put(key, memory_obj)
        initial_keys.append(key)

    # function to perform random operations
    def worker(worker_id, iterations=50):
        operations = []
        for _ in range(iterations):
            op = random.choice(['get', 'put', 'remove', 'contains', 'touch'])

            if op in ['get', 'remove', 'contains', 'touch']:
                # use an existing key if available
                if initial_keys and random.random() < 0.7:
                    key = random.choice(initial_keys)
                else:
                    key = generate_random_key()

                if op == 'get':
                    result = backend.get(key)
                    if result is not None:
                        memory_allocator.ref_count_down(result)
                elif op == 'remove':
                    backend.remove(key)
                elif op == 'contains':
                    backend.contains(key)
                elif op == 'touch':
                    backend.touch(key)
            else:  # put
                key = generate_random_key()
                memory_obj = create_memory_obj()
                backend.put(key, memory_obj)

            # small sleep to increase chance of thread interleaving
            time.sleep(0.001)
            operations.append((op, key))
        return operations

    # run multiple threads concurrently
    num_threads = 5
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [future.result() for future in futures]

    # verify backend still works after concurrent access
    test_key = generate_random_key()
    test_obj = create_memory_obj()
    backend.put(test_key, test_obj)

    retrieved = backend.get(test_key)
    assert retrieved is not None
    assert memory_allocator.get_ref_count(retrieved) > 0
    memory_allocator.ref_count_down(retrieved)

    # verify we can clear the cache
    num_cleared = backend.clear()
    assert num_cleared >= 0

def test_local_cpu_backend_put_from_callback():
    """test the from_callback parameter in the put method."""
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator)

    # create a key and memory object
    key = generate_random_key()
    memory_obj = create_memory_obj()

    # initial ref count should be 1
    assert memory_allocator.get_ref_count(memory_obj) == 1

    # use from_callback=True, which should NOT increment the ref count
    backend.put(key, memory_obj, from_callback=True)
    assert memory_allocator.get_ref_count(memory_obj) == 1

    # verify the object was added to the cache
    assert backend.contains(key)

    # create another object for comparison with default behavior
    key2 = generate_random_key()
    memory_obj2 = create_memory_obj()

    # initial ref count should be 1
    assert memory_allocator.get_ref_count(memory_obj2) == 1

    # default behavior (from_callback=False) should increment the ref count
    backend.put(key2, memory_obj2)
    assert memory_allocator.get_ref_count(memory_obj2) == 2

    # verify getting both objects works
    retrieved1 = backend.get(key)
    retrieved2 = backend.get(key2)

    assert retrieved1 is not None
    assert retrieved2 is not None

    # clean up
    memory_allocator.ref_count_down(retrieved1)
    memory_allocator.ref_count_down(retrieved2)

    # test overwriting an existing key
    new_obj = create_memory_obj()
    backend.put(key, new_obj, from_callback=True)

    # should have removed the old object and not incremented new_obj's ref count
    assert memory_allocator.get_ref_count(new_obj) == 1
    assert memory_allocator.get_ref_count(memory_obj) == 0  # old object should be gone
