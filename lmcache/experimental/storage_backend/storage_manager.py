import threading
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import torch
from sortedcontainers import SortedDict

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (BufferMemoryObj,
                                                    MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj)
from lmcache.experimental.storage_backend import CreateStorageBackends
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


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
        self.use_hot = config.local_cpu

        #TODO: remove hardcode
        dst_device = "cuda"
        self.storage_backends: Dict[str, StorageBackendInterface] =\
            CreateStorageBackends(config, metadata, dst_device)
        self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}
        self.put_tasks: Dict[str, Dict[CacheEngineKey, Tuple[Future,
                                                             MemoryObj]]] = {}

        for backend_name in self.storage_backends.keys():
            self.put_tasks[backend_name] = {}

        self.manager_lock = threading.Lock()

    def put_callback(self, future, backend_name, key):
        """
        Update metadata and free resources after put.
        """
        self.manager_lock.acquire()
        future, memory_obj = self.put_tasks[backend_name][key]

        # raises exception if put failed
        try:
            future.result()
        except Exception as e:
            logger.error(
                f"Exception captured from future in put_callback: {e}")
            raise e
        self.put_tasks[backend_name].pop(key)

        # TODO: Might need to modify free such that it's `ref_count-1`
        # because there might be multiple references (backends)
        # using the same memory_obj
        # It won't error now for we only have disk backend
        if not self.use_hot:
            self.memory_allocator.free(memory_obj)

        self.storage_backends[backend_name].insert_key(key, memory_obj)
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

        for backend_name in self.storage_backends:
            if key in self.put_tasks[backend_name]:
                if not self.use_hot:
                    self.memory_allocator.free(memory_obj)
                self.manager_lock.release()
                return
        self.manager_lock.release()

        for backend_name, backend in self.storage_backends.items():
            put_task = backend.submit_put_task(key, memory_obj)

            # For debugging purpose
            #backend.put_blocking(key, memory_obj)

            lambda_callback = lambda f, backend_name=backend_name: \
                   self.put_callback(f, backend_name, key)

            self.manager_lock.acquire()
            self.put_tasks[backend_name][key] = (put_task, memory_obj)
            self.manager_lock.release()
            put_task.add_done_callback(lambda_callback)

    def get(self, key: CacheEngineKey) -> Optional[BufferMemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """
        # Search in prefetch task
        self.manager_lock.acquire()
        prefetch_task = self.prefetch_tasks.get(key, None)
        self.manager_lock.release()

        # Wait until prefetch task finishes
        # Here, it is assumed all prefetch tasks load the memoryobj to
        # hot cache (pinned cpu buffer)
        if prefetch_task is not None:
            assert self.use_hot is True,\
                "CPU cache must be enabled for prefetching"
            logger.debug("Waiting for prefetching result. "
                         "Optimally, this should not happen.")
            # Calling result() twice (already once in callback) will have
            # no effect
            # Tune the timeout for better performance
            prefetch_task.result(timeout=1)

        # Search in hot_cache
        self.manager_lock.acquire()
        memory_obj = self.hot_cache.get(key, None)
        if memory_obj is not None:
            self.manager_lock.release()
            return memory_obj

        # Search all backends for blocking get
        for backend_name, backend in self.storage_backends.items():
            # Avoid read-write contention
            if key in self.put_tasks[backend_name]:
                continue
            buffer_memory_obj = backend.get_blocking(key)
            if buffer_memory_obj is not None:
                # NOTE(Jiayi): bypass the allocator for now
                self.manager_lock.release()
                return buffer_memory_obj

        self.manager_lock.release()
        return None

    def prefetch_callback(self, future, key):
        """
        Update metadata after prefetch.
        """
        self.manager_lock.acquire()
        prefetch_task = self.prefetch_tasks.pop(key)
        self.manager_lock.release()
        try:
            buffer_memory_obj = prefetch_task.result()
        except Exception as e:
            logger.error(
                f"Exception captured from future in prefetch_callback: {e}")
            raise e
        kv_chunk = buffer_memory_obj.tensor
        kv_shape = kv_chunk.shape
        kv_dtype = kv_chunk.dtype
        memory_obj = self.memory_allocator.allocate(kv_shape, kv_dtype)
        if memory_obj is None:
            logger.warning("Memory allocation failed in prefetch_callback")
            return

        assert memory_obj.tensor is not None, "Encounter invalid tensor"

        # TODO(Jiayi): this part should be done in another process if
        # the cpu->pinned cpu copy is blocking.
        prefetch_stream = torch.cuda.Stream()
        with torch.cuda.stream(prefetch_stream):
            memory_obj.tensor.copy_(kv_chunk, non_blocking=True)
        prefetch_stream.synchronize()
        # TODO(Jiayi): please remove this hardcode
        memory_obj.metadata.fmt = MemoryFormat.KV_BLOB

        self.manager_lock.acquire()
        self.hot_cache[key] = memory_obj
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
        self.manager_lock.release()

        for backend in self.storage_backends.values():
            prefetch_task = backend.submit_prefetch_task(key)
            if prefetch_task is None:
                continue
            lambda_callback = lambda f: \
                self.prefetch_callback(f, key)

            self.manager_lock.acquire()
            self.prefetch_tasks[key] = prefetch_task
            prefetch_task.add_done_callback(lambda_callback)
            self.manager_lock.release()
            break

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
    ) -> bool:
        """
        Check whether the key exists in the storage backend.
        
        :param CacheEngineKey key: The key to check.
        
        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["Hot", "LocalDiskBackend"] for now.
        If None, search in all backends.
        
        return: True if the key exists in the specified storage backends.
        """
        with self.manager_lock:
            if search_range is None or "Hot" in search_range:
                if key in self.hot_cache:
                    return True

            for backend_name, backend in self.storage_backends.items():
                if search_range is not None and \
                    backend_name not in search_range:
                    continue
                if backend.contains(key):
                    return True

            return False
