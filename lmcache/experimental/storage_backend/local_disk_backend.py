import abc
import threading
from typing import Callable
from concurrent.futures import Future
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor

from lmcache.experimental.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey, DiskCacheMetadata


class LocalDiskBackend(StorageWorkerInterface):
    def __init__(self):
        # put worker can have multiple workers
        # prefetch should have fewer or just one as latency is
        # more important than throughput
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        
        self.put_executor = ProcessPoolExecutor(max_workers=4)
        self.prefetch_executor = ProcessPoolExecutor(max_workers=1)
        
        self.storage_lock = threading.Lock()
        
        self.dst_device = "cuda"
        
    
    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Future:
        """An async function to put the MemoryObj into the storage backend.
        It should free the memory object after finish putting the object.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored.
        """
        future = self.proc_pool_executor.submit(self.save_disk, path,
                                                kv_obj.data)

    def submit_get_task(
        self,
        key: CacheEngineKey,
        callback: Callable[[
            MemoryObj,
        ], None],
    ) -> None:
        """An async function to get the MemoryObj from the storage backend.
        Will call the callback with the MemoryObj when finished.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param Callable[MemoryObj, None] callback: The callback function to 
            be called with the MemoryObj.
        """
        raise NotImplementedError
    
    def get_blocking(
        self,
        key: CacheEngineKey,
    ):
        pass
