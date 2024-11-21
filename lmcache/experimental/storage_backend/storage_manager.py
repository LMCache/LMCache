from typing import Optional

from sortedcontainers import SortedDict

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.utils import CacheEngineKey


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 allocator: MemoryAllocatorInterface):
        self.hot_cache = SortedDict()

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        """Non-blocking function to put the memory object into the storages.
        """
        self.hot_cache[key] = memory_obj

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking function to get the memory object from the storages.
        """
        return self.hot_cache.get(key, None)

    def prefetch(self, key: CacheEngineKey) -> None:
        """Launch a prefetch request in the storage backend. Non-blocking
        """
        pass

    def contains(self, key: CacheEngineKey) -> bool:
        """Check whether the key exists in the storage backend.
        """
        return key in self.hot_cache
