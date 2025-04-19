import pytest
import torch
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from collections import OrderedDict

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryObj,
                                                    MemoryObjMetadata,
                                                    MemoryFormat,
                                                    TensorMemoryObj)
from lmcache.experimental.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.utils import CacheEngineKey

# for the storage manager
from concurrent.futures import Future
import asyncio


class MemoryObjFactory:

    def __init__(self):
        self.counter = 0

    def create_memory_obj(self):
        self.counter += 1
        tensor = torch.ones(10, 10) * self.counter
        metadata = MemoryObjMetadata(
            shape=tensor.shape,
            dtype=tensor.dtype,
            address=self.counter,  # address is used for key generation
            phy_size=tensor.numel() * tensor.element_size(),
            ref_count=1,
            fmt=MemoryFormat.KV_BLOB)
        memory_obj = TensorMemoryObj(tensor, metadata)
        return memory_obj


# usually the key and memory object are generated together but for testing
# we generate the memory object first and then the key
def generate_key(memory_obj: MemoryObj):
    return CacheEngineKey("vllm", "test_model", 1, 0,
                          f"chunk_{memory_obj.metadata.address}")


# fragile mock memory allocator that doesn't actually grab or distribute memory
class MockMemoryAllocator:

    def __init__(self, max_allocations=None):
        self.ref_counts = {}
        self.max_allocations = max_allocations
        self.memory_obj_factory = MemoryObjFactory()
        self.pin_allocator = type('MockPinAllocator', (),
                                  {'num_active_allocations': 0})()

    def ref_count_up(self, memory_obj):
        assert memory_obj in self.ref_counts, \
            "can not ref_count_up on either a non-existent memory object" \
            "or one that has already been freed (ref_count_down'ed to 0)"
        self.ref_counts[memory_obj] += 1

    def ref_count_down(self, memory_obj):
        if memory_obj in self.ref_counts:
            self.ref_counts[memory_obj] -= 1
            if self.ref_counts[memory_obj] == 0:
                del self.ref_counts[memory_obj]
                self.pin_allocator.num_active_allocations -= 1

    def get_ref_count(self, memory_obj):
        return self.ref_counts.get(memory_obj, 0)

    # the sizes passed to allocate are not used
    def allocate(self, shape, dtype):
        if self.max_allocations is not None and \
            self.pin_allocator.num_active_allocations >= self.max_allocations:
            return None
        self.pin_allocator.num_active_allocations += 1
        memory_obj = self.memory_obj_factory.create_memory_obj()
        self.ref_counts[memory_obj] = 1
        return memory_obj


def test_local_cpu_backend_basic_operations():
    # setup with no lookup server
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator, real_allocator=False)
    memory_obj = memory_allocator.allocate(torch.Size([10, 10]), torch.float32)
    key = generate_key(memory_obj)

    # test contains on empty backend
    assert not backend.contains(key)

    # test put and contains
    backend.put(key, memory_obj)
    assert backend.contains(key)

    # test get
    retrieved = backend.get(key)
    assert retrieved is not None
    assert retrieved.tensor is not None

    # test touch (lru ordering)
    old_key = key
    new_obj = memory_allocator.allocate(torch.Size([10, 10]), torch.float32)
    new_key = generate_key(new_obj)
    backend.put(new_key, new_obj)
    backend.touch(old_key)  # move to end
    assert backend.get_keys() == [new_key, old_key]

    # test remove (first release our references)
    memory_allocator.ref_count_down(retrieved)
    memory_allocator.ref_count_down(memory_obj)
    assert backend.remove(old_key)
    assert not backend.contains(old_key)

    # test clear (first release our references)
    memory_allocator.ref_count_down(new_obj)
    num_cleared = backend.clear()
    assert num_cleared == 1  # should be just new_key left
    assert not backend.contains(new_key)


def test_local_cpu_backend_ref_counting():
    memory_allocator = MockMemoryAllocator()
    backend = LocalCPUBackend(memory_allocator, real_allocator=False)
    memory_obj = memory_allocator.allocate(torch.Size([10, 10]), torch.float32)
    key = generate_key(memory_obj)

    # initial ref count should be 1
    assert memory_allocator.get_ref_count(memory_obj) == 1

    # after put, ref count should be 2 (one for caller, one for hot cache)
    backend.put(key, memory_obj)
    assert backend.contains(key)
    assert memory_allocator.get_ref_count(memory_obj) == 2

    # after get, ref count should be 3 (added for the caller of get)
    retrieved = backend.get(key)
    assert memory_allocator.get_ref_count(memory_obj) == 3

    # after caller is done with retrieved object
    memory_allocator.ref_count_down(retrieved)
    assert memory_allocator.get_ref_count(memory_obj) == 2

    # after remove, ref count should still be 2 because the hot cache refuses to
    # evict objects with ref count > 1 (and we are sitll holding it)
    assert not backend.remove(key)
    assert memory_allocator.get_ref_count(memory_obj) == 2

    # let's release the original reference
    memory_allocator.ref_count_down(memory_obj)
    assert memory_allocator.get_ref_count(memory_obj) == 1

    # now the hot cache should be able to evict the object
    assert backend.remove(key)
    assert memory_allocator.get_ref_count(memory_obj) == 0


def test_local_cpu_backend_allocation_eviction():
    memory_allocator = MockMemoryAllocator(max_allocations=5)
    backend = LocalCPUBackend(memory_allocator, real_allocator=False)

    # fill the cache to capacity
    keys = []
    for i in range(5):
        memory_obj = memory_allocator.allocate(torch.Size([10, 10]),
                                               torch.float32)
        key = generate_key(memory_obj)
        backend.put(key, memory_obj)
        keys.append(key)
        # release the reference so that the memory object can be evicted later
        memory_allocator.ref_count_down(memory_obj)

    # double check that the next allocation will fail
    failed_memory_obj = memory_allocator.allocate(torch.Size([10, 10]),
                                                  torch.float32)
    assert failed_memory_obj is None

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
    backend = LocalCPUBackend(memory_allocator, real_allocator=False)
    memory_obj = memory_allocator.allocate(torch.Size([10, 10]), torch.float32)
    key = generate_key(memory_obj)

    with pytest.raises(NotImplementedError):
        backend.exists_in_put_tasks(key)

    with pytest.raises(NotImplementedError):
        backend.submit_put_task(key, memory_obj)

    with pytest.raises(NotImplementedError):
        backend.submit_prefetch_task(key)

    with pytest.raises(NotImplementedError):
        backend.get_blocking(key)


def test_storage_manager_no_local_cpu_backend():
    # Set remote_url to None to avoid creating a remote backend
    config = LMCacheEngineConfig.from_defaults(local_cpu=False, \
                                                local_disk=False, remote_url=None, \
                                                lookup_url=None, distributed_url=None)
    metadata = LMCacheEngineMetadata(model_name="test_model",
                                     world_size=1,
                                     worker_id=0,
                                     fmt="vllm",
                                     kv_dtype=torch.float32,
                                     kv_shape=(32, 2, 256, 8, 128))
    allocator = MockMemoryAllocator()

    manager = StorageManager(config, metadata, allocator)
    assert manager.hot_cache is None
    assert len(manager.storage_backends) == 0


def test_storage_manager_with_local_cpu_backend():
    # Set remote_url to None to avoid creating a remote backend
    config = LMCacheEngineConfig.from_defaults(local_cpu=True, max_local_cpu_size=5, \
                                                local_disk=False, remote_url=None, \
                                                lookup_url=None, distributed_url=None)
    metadata = LMCacheEngineMetadata(model_name="test_model",
                                     world_size=1,
                                     worker_id=0,
                                     fmt="vllm",
                                     kv_dtype=torch.float32,
                                     kv_shape=(32, 2, 256, 8, 128))
    allocator = MockMemoryAllocator()

    # mock the CreateStorageBackends function as an empty OrderedDict so that we
    # can test just the storage manager with a hot cache
    with patch("lmcache.experimental.storage_backend.CreateStorageBackends",
               return_value=OrderedDict()):
        # create the StorageManager
        manager = StorageManager(config, metadata, allocator)

        # verify manager.hot_cache is LocalCPUBackend
        assert isinstance(manager.hot_cache, LocalCPUBackend)

        # test operations through manager
        memory_obj = manager.allocate(torch.Size([10, 10]), torch.float32)
        key = generate_key(memory_obj)

        # verify hot cache is empty
        assert not manager.contains(key, ["Hot"])

        # put the object in the manager
        manager.put(key, memory_obj)

        # verify it's now in the hot cache
        assert manager.contains(key, ["Hot"])

        # the reason why the ref count is 1 and not 2 is because put calls ref_count_down
        # as a way to clean up for the caller (but hot cache still holds a ref)
        assert allocator.get_ref_count(memory_obj) == 1

        # get the object and verify it's the same (gives us a ref count)
        retrieved = manager.get(key)
        assert retrieved is not None
        assert allocator.get_ref_count(retrieved) == 2

        # clean up (so we can remove from hot cache)
        allocator.ref_count_down(retrieved)

        # remove the object (only remove location is hot cache)
        assert manager.remove(key, ["Hot"]) == 1
        assert not manager.contains(key, ["Hot"])


def test_storage_manager_with_local_cpu_backend_with_disk():
    # Set remote_url to None to avoid creating a remote backend
    config = LMCacheEngineConfig.from_defaults(local_cpu=True, max_local_cpu_size=5, \
                                                local_disk="/tmp/test_disk", max_local_disk_size=5, \
                                                remote_url=None, lookup_url=None, \
                                                distributed_url=None)
    metadata = LMCacheEngineMetadata(model_name="test_model",
                                     world_size=1,
                                     worker_id=0,
                                     fmt="vllm",
                                     kv_dtype=torch.float32,
                                     kv_shape=(32, 2, 256, 8, 128))
    allocator = MockMemoryAllocator()

    # don't mock CreateStorageBackends because we want to test the disk backend
    # with the hot cache
    manager = StorageManager(config, metadata, allocator)

    # verify manager.hot_cache is LocalCPUBackend
    assert isinstance(manager.hot_cache, LocalCPUBackend)
    assert len(manager.storage_backends) == 1

    # test operations through manager
    memory_obj = manager.allocate(torch.Size([10, 10]), torch.float32)
    key = generate_key(memory_obj)

    # verify hot cache is empty
    assert not manager.contains(key, ["Hot"])

    # put the object in the manager
    manager.put(key, memory_obj)

    # verify it's now in the hot cache
    assert manager.contains(key, ["Hot"])

    # spin loop until the ref count is 1
    # (the disk backend is done writing to disk)
    while allocator.get_ref_count(memory_obj) != 1:
        time.sleep(0.001)

    # remove the object only from hot cache (should still be in disk)
    assert manager.remove(key, ["Hot"]) == 1
    assert not manager.contains(key, ["Hot"])

    # nobody should be holding a ref to the memory object
    assert allocator.get_ref_count(memory_obj) == 0

    # prefetch from the disk
    manager.prefetch(key)

    # hacky: wait for 1 second for the prefetch to complete
    time.sleep(1)

    # verify it's now in the hot cache
    assert manager.contains(key, ["Hot"])

    # remove the object only from hot cache (should still be in disk)
    assert manager.remove(key, ["Hot"]) == 1
    assert not manager.contains(key, ["Hot"])

    # nobody should be holding a ref to the memory object
    assert allocator.get_ref_count(memory_obj) == 0

    # this time use blocking get instead of prefetch
    retrieved = manager.get(key)
    assert retrieved is not None

    # both the hot cache and the caller (us) should be holding a ref
    assert allocator.get_ref_count(retrieved) == 2
