from collections import OrderedDict
from concurrent.futures import Future
import threading
from typing import List, Optional, Tuple

import torch

from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LocalCPUBackend(StorageBackendInterface):
    """
    The local CPU backend is primarily used for hot cache, thinly wrapping
    an ordered dictionary and tightly coupled with the memory allocator.

    It can not use the LRUEvictor() helper because its size is variable
    depending on how much free space is left in the allocator.

    R/W from RAM is synchronous and so does not need an event loop and
    does not use futures.
    """
    def __init__(self,
        memory_allocator: MemoryAllocatorInterface,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        # rely completely on ordered dict to manage LRU
        self.hot_cache_: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator
        # multiple threads can access the hot cache (protects self.hot_cache_)
        self.hot_cache_lock = threading.Lock()

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        do not check asynchronous futures for cpu backend
        """
        raise NotImplementedError

    def submit_put_task(self, key: CacheEngineKey,
                        memory_obj: MemoryObj) -> Optional[Future]:
        """
        do not run the asynchronous put for cpu backend
        """
        raise NotImplementedError

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """
        the cpu backend does not need to prefetch into itself
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        please use regular get() to avoid possible confusion about async
        """
        raise NotImplementedError

    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check if the key is in the hot cache.
        """
        self.hot_cache_lock.acquire()
        contains_key = key in self.hot_cache_
        self.hot_cache_lock.release()
        return contains_key

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Get a memory object from the cpu backend.

        The caller is responsible for ref_count_down() when they're
        done with the memory object.
        """
        self.hot_cache_lock.acquire()
        if key not in self.hot_cache_:
            self.hot_cache_lock.release()
            return None
        memory_obj = self.hot_cache_[key]
        # ref count up for the caller
        self.memory_allocator.ref_count_up(memory_obj)
        self.hot_cache_.move_to_end(key)
        self.hot_cache_lock.release()
        return memory_obj

    def pop(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Pop a memory object from the cpu backend.

        The caller is responsible for ref_count_down() when they're
        done with the memory object.
        """
        self.hot_cache_lock.acquire()
        if key not in self.hot_cache_:
            self.hot_cache_lock.release()
            return None
        memory_obj = self.hot_cache_.pop(key)
        # we ref up here for the caller but we also ref down
        # because the hot cache is no longer referencing the object
        # these two operations cancel so we do nothing
        self.hot_cache_lock.release()
        return memory_obj

    def touch(self, key: CacheEngineKey) -> None:
        """
        Touch a key in the hot cache (maximize recency)
        """
        self.hot_cache_lock.acquire()
        if key in self.hot_cache_:
            self.hot_cache_.move_to_end(key)
        self.hot_cache_lock.release()

    def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """
        Put a key, memory object pair in the hot cache.

        If the key is already in the hot cache, we need to evict the old
        memory object and replace it with the new one.
        """
        # During overwrite, we need to free the old memory object
        # to avoid memory leak.
        # NOTE(Jiayi): overwrite should not happen, at least for
        # prefix caching
        self.hot_cache_lock.acquire()
        if key in self.hot_cache_:
            old_memory_obj = self.hot_cache_.pop(key)
            self.memory_allocator.ref_count_down(old_memory_obj)
        self.hot_cache_[key] = memory_obj
        self.memory_allocator.ref_count_up(memory_obj)
        self.hot_cache_lock.release()

    def allocate(self, shape: torch.Size,
                    dtype: torch.dtype) -> Optional[MemoryObj]:
        """
        allocate a memory object in the cpu backend by evicting LRU
        from hot cache

        takes in the shape and dtype of the memory object to be allocated

        returns:
        - None if we could not make space for the memory object in hot cache
        - the allocated memory object otherwise
        """
        evict_keys = []

        self.hot_cache_lock.acquire()
        for evict_key in self.hot_cache_:
            # If the ref_count > 1, we cannot evict it as the hot cache
            # might be used as buffers by other storage backends
            if self.memory_allocator.get_ref_count(
                    self.hot_cache_[evict_key]) > 1:
                continue
            evict_keys.append(evict_key)
            self.memory_allocator.ref_count_down(self.hot_cache_[evict_key])
            memory_obj = self.memory_allocator.allocate(shape, dtype)
            logger.debug("Evicting 1 chunk from hot cache")
            if memory_obj is not None:
                break
            # TODO(Jiayi): move this before the loop
            # In this way, we don't need to do eviction for big objects
            # TODO(Jiayi): the following code is hacky, please refactor
            if self.memory_allocator.pin_allocator.num_active_allocations == 0:
                break
        for evict_key in evict_keys:
            self.hot_cache_.pop(evict_key)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)
        self.hot_cache_lock.release()
        return memory_obj

    def close(self) -> None:
        pass


