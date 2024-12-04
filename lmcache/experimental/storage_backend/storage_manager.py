import asyncio
import threading
from typing import Optional

from sortedcontainers import SortedDict
from concurrent.futures import Future

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    BufferMemoryObj,
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
        self.memory_allocator = allocator
        self.hot_cache = SortedDict()
        self.use_hot = True
        
        # TODO(Jiayi): initialize devices based on config
        self.storage_backends: List[StorageWorkerInterface] = [] 
        self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}
        self.put_tasks: Dict[str, Dict[CacheEngineKey, 
                            Tuple[Future, MemoryObj]]] = {}
        for storage_backend in storage_backends:
            storage_name = str(storage_backend)
            self.put_tasks[storage_name] = {}
        
        self.manager_lock = threading.Lock()
    
    def put_callback(self, future, storage_type, key):
        """
        Update metadata and free resources after put.
        """
        self.manager_lock.acquire()
        memory_obj = self.put_tasks[storage_type][key]
        size = memory_obj.get_size()
        if not self.use_hot:
            self.memory_allocator.free(memory_obj)
        self.put_tasks[storage_type].pop(key)
        self.storage_backends[storage_type].insert_key(key, size)
        self.manager_lock.release()
        
    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by 
        storage manager) or has been stored (handled by storage backend).
        """
        self.manager_lock.acquire()
        if self.use_hot:
            self.hot_cache[key] = memory_obj
        
        if key in self.put_tasks:
            if not self.use_hot:
                self.memory_allocator.free(memory_obj)
            self.manager_lock.release()
            return
        
        for storage_backend in self.storage_backends:
            put_task = storage_backend.submit_put_task(key, memory_obj)
            # NOTE(Jiayi): Callback is executed in worker thread in
            # ThreadPoolExecutor and in main process in ProcessPoolExecutor
            lambda_callback = lambda f: self.put_callback(f, storage_type, key)
            put_task.add_done_callback(lambda_callback)
            storage_type = str(storage_backend)
            
            self.manager_lock.acquire()
            self.put_tasks[storage_type][key] = put_task
            self.manager_lock.release()
            
    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking function to get the memory object from the storages.
        """
        # Search in prefetch task
        self.manager_lock.acquire()
        prefetch_task = self.prefetch_tasks.get(key, None)
        self.manager_lock.release()
        
        # Wait until prefetch task finishes
        # Here, it is assumed all prefetch tasks load the memoryobj to
        # hot cache (pinned cpu buffer)
        if prefetch_task is not None:
            prefetch_task.result()
        
        # Search in hot_cache
        self.manager_lock.acquire()
        memory_obj = self.hot_cache.get(key, None)
        if memory_obj is not None:
            self.manager_lock.release()
            return memory_obj
        
        # Search all backends for blocking get
        for storage_backend in self.storage_backends:
            # Avoid read-write contention
            if key in self.put_tasks[]:
                continue
            tensor_gpu = storage_backend.get_blocking(key)
            if tensor_gpu is not None:
                # NOTE(Jiayi): bypass the allocator for now
                self.manager_lock.release()
                return BufferMemoryObj(tensor_gpu)
        
        self.manager_lock.release()
        return None
        
    def prefetch_callback(self, future, key):
        """
        Update metadata after prefetch.
        """
        self.manager_lock.acquire()
        prefetch_task = self.prefetch_tasks.pop(key)
        kv_chunk = prefetch_task.result()
        kv_shape = kv_chunk.shape
        kv_dtype = kv_chunk.kv_dtype
        memory_obj = self.memory_allocator.allocate(kv_shape, kv_dtype)
        self.manager_lock.release()
        
        prefetch_stream = torch.cuda.Stream()
        with torch.cuda.stream(put_stream):
            memory_obj.tensor.copy_(kv_chunk, non_blocking=True)
        prefetch_stream.synchronize()
        
        self.manager_lock.acquire()
        self.hot_cahce[key] = memory_obj
        self.manager_lock.release()
        
    
    def prefetch(self, key: CacheEngineKey) -> None:
        """Launch a prefetch request in the storage backend. Non-blocking
        """
        
        # Call contains for each backend. Find the nearest cache
        self.manager_lock.acquire()
        if key in self.hot_cache:
            self.manager_lock.release()
            return
        if key in self.prefetch_tasks:
            self.manager_lock.release()
            return
        for storage_backend in self.storage_backends:
            task = staorage_backend.submit_prefetch_task(key)
            if task is None:
                continue
            lambda_callback = lambda f: self.prefetch_callback(
                f, key)
            prefetch_task.add_done_callback(self.prefetch_callback)
            self.prefetch_tasks[key] = task
            break
        self.manager_lock.release()

    def contains(self, key: CacheEngineKey) -> bool:
        """Check whether the key exists in the storage backend.
        """
        if key in self.hot_cache:
            return True
        
        for storage_backend in storage_backends:
            if storage_backend.contains(key):
                return True 
        
        return False