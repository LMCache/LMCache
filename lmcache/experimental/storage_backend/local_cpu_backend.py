from concurrent.futures import Future
from typing import Optional
from collections import OrderedDict

from lmcache.experimental.lookup_server import LookupServerInterface

import threading

from lmcache.experimental.memory_management import MemoryObj, MemoryAllocatorInterface, MixedMemoryAllocator
from lmcache.utils import CacheEngineKey
from lmcache.experimental.storage_backend.abstract_backend import StorageBackendInterface


class LocalCPUBackend(StorageBackendInterface):
    """
    The local CPU backend is primarily used for hot cache, thinly wrapping
    an ordered dictionary and tightly coupled with the memory allocator.

    It can not use the LRUEvictor() helper because its size is variable
    depending on how much free space is left in the allocator.
    (max_local_cpu_size initializes the memory_allocator)

    it stores memory objects in cpu memory and is completely synchronous
    """
    def __init__(self,
                memory_allocator: MemoryAllocatorInterface,
                lookup_server: Optional[LookupServerInterface] = None,
                dst_device: str = "cpu"):
        self.dict: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator
        assert isinstance(self.memory_allocator, MixedMemoryAllocator), \
            "LocalCPUBackend must be used with a MixedMemoryAllocator"
        self.dst_device = dst_device
        self.cpu_lock = threading.Lock()

    def contains(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            return key in self.dict

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def _put(self, key: CacheEngineKey, obj: MemoryObj) -> None:
        """
        synchronously (immediately) write to cpu memory
        ref count stays up because the memory object stays in cpu memory
        """
        with self.cpu_lock:
            if key in self.dict:
                old_memory_obj = self.dict.pop(key)
                self.memory_allocator.ref_count_down(old_memory_obj)
            self.dict[key] = obj
            self.memory_allocator.ref_count_up(obj)

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        return a dummy future object (result is immediately available so
        non blocking)
        """
        self._put(key, obj)
        return Future()

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        None if not in cpu cache
        a future that is instantly available with the memory_obj if cpu backend
        already has it
        """
        with self.cpu_lock:
            if key not in self.dict:
                return None
            # ref count up for caller to avoid situation where the memory_obj
            # is freed from local cpu backend before the caller unlocks the
            # future
            self.memory_allocator.ref_count_up(self.dict[key])
            memory_obj = self.dict[key]

            future: Future[MemoryObj] = Future()
            future.set_result(memory_obj)
            return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.dict:
                return None
            memory_obj = self.dict[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            self.memory_allocator.ref_count_up(memory_obj)
            return memory_obj

    def clear(self) -> int:
        """
        counts the number of memory objects removed
        """
        num_removed = 0
        with self.cpu_lock:
            if self.lookup_server is not None:
                self.lookup_server.batched_remove(list(self.dict.keys()))
            for memory_obj in self.dict.values():
                self.memory_allocator.ref_count_down(memory_obj)
                num_removed += 1
            self.dict.clear()
        return num_removed

    def close(self) -> None:
        self.clear()
