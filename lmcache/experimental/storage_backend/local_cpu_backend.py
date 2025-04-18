from collections import OrderedDict
from concurrent.futures import Future
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
    It is synchronous and so does not need a loop and does not use futures.
    """
    def __init__(self,
        memory_allocator: MemoryAllocatorInterface,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.hot_cache_: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator

    def contains(self, key: CacheEngineKey) -> bool:
        return key in self.hot_cache_

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
        return None

    def make_space_for(self, shape: torch.Size, dtype: torch.dtype) -> Optional[MemoryObj]:
        """
        make space for and allocate a memory object in the cpu backend
        """
        evict_keys = []

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
        return memory_obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        pass

    def put_blocking(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        pass

    def close(self) -> None:
        pass


