import asyncio
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import List, Optional, Tuple

import torch

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LocalCPUBackend(StorageBackendInterface):
    """
    The local CPU backend is primarily used for hot cache.
    It can not use LRU eviction because its size is variable depending on how
    much free space is left in the allocator.
    It is synchronous and so does not need a loop and does not use futures.
    """
    def __init__(self,
        memory_allocator: MemoryAllocatorInterface,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.dict: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            return key in self.dict

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        # no asynchronous put tasks for cpu backend
        return False

    def submit_put_task(self, key: CacheEngineKey,
                        memory_obj: MemoryObj) -> Optional[Future]:
        # since cpu operations are synchronous, implement as a direct put
        self.put_blocking(key, memory_obj)
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        # the cpu backend does not need to prefetch into itself
        return None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        pass

    def put_blocking(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        pass

    def close(self) -> None:
        pass


