import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Optional

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj,
                                                    MixedMemoryAllocator)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.cache_controller.message import (KVAdmitMsg,
                                                           KVEvictMsg)
from lmcache.utils import CacheEngineKey
from lmcache.observability import LMCStatsMonitor
if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker


class LocalCPUBackend(StorageBackendInterface):
    """
    The local cpu backend size is variable depending on how much free space is
    left in the allocator so we cannot use LRUEvictor().
    (max_local_cpu_size > 0 initializes the memory_allocator)
    """

    def __init__(self,
                 config: LMCacheEngineConfig,
                 memory_allocator: MemoryAllocatorInterface,
                 lookup_server: Optional[LookupServerInterface] = None,
                 lmcache_worker: Optional["LMCacheWorker"] = None,
                 dst_device: str = "cpu"):
        assert config.local_cpu is not None
        self.dict: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator
        assert isinstance(self.memory_allocator, MixedMemoryAllocator), \
            "LocalCPUBackend must be used with a MixedMemoryAllocator"
        self.dst_device = dst_device
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

    def contains(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            return key in self.dict

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def insert_key(self, key: CacheEngineKey, obj: MemoryObj) -> None:
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

            self.usage += obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            # push kv admit msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash,
                               "cpu"))

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        return a dummy future object (result is immediately available so
        non blocking)
        """
        self.insert_key(key, obj)
        return Future()

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        this is not a meaningful operation for local cpu backend
        """
        pass

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

    def remove(self, key: CacheEngineKey) -> None:
        with self.cpu_lock:
            if key in self.dict:
                memory_obj = self.dict.pop(key)
                self.memory_allocator.ref_count_down(memory_obj)

                self.usage -= memory_obj.get_size()
                self.stats_monitor.update_local_cache_usage(self.usage)

                if self.lmcache_worker is not None:
                    self.lmcache_worker.put_msg(
                        KVEvictMsg(self.instance_id, key.worker_id,
                                   key.chunk_hash, "cpu"))

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
