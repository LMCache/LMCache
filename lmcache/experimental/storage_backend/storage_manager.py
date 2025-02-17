import asyncio
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj,
                                                    MixedMemoryAllocator)
from lmcache.experimental.storage_backend import CreateStorageBackends
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

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
        self.hot_cache: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.use_hot = config.local_cpu

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        #TODO: remove hardcode
        dst_device = "cuda"
        self.storage_backends: OrderedDict[str, StorageBackendInterface] =\
            CreateStorageBackends(
                config, metadata, self.loop, allocator, dst_device)
        self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}
        self.put_tasks: Dict[str, Dict[CacheEngineKey, Tuple[Future,
                                                             MemoryObj]]] = {}

        for backend_name in self.storage_backends.keys():
            self.put_tasks[backend_name] = {}

        self.manager_lock = threading.Lock()

        self.stream = torch.cuda.Stream()

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        eviction=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        self.manager_lock.acquire()
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if not eviction or memory_obj is not None:
            self.manager_lock.release()
            return memory_obj

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)
        evict_keys = []

        for evict_key in self.hot_cache:

            # If the ref_count > 1, we cannot evict it as the hot cache
            # might be used as buffers by other storage backends
            if self.memory_allocator.get_ref_count(
                    self.hot_cache[evict_key]) > 1:
                continue
            evict_keys.append(evict_key)
            self.memory_allocator.ref_count_down(self.hot_cache[evict_key])
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
            self.hot_cache.pop(evict_key)

        self.manager_lock.release()
        return memory_obj

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
            # During overwrite, we need to free the old memory object
            # to avoid memory leak.
            # NOTE(Jiayi): overwrite should not happen, at least for
            # prefix caching
            if key in self.hot_cache:
                old_memory_obj = self.hot_cache.pop(key)
                self.memory_allocator.ref_count_down(old_memory_obj)

            self.hot_cache[key] = memory_obj
            self.memory_allocator.ref_count_up(memory_obj)

        # TODO(Jiayi): currently, the entire put task will be cancelled
        # if one of the backend is already storing this cache.
        # This might not be ideal.
        for storage_backend in self.storage_backends.values():
            if storage_backend.exists_in_put_tasks(key):
                self.memory_allocator.ref_count_down(memory_obj)
                self.manager_lock.release()
                return
        self.manager_lock.release()

        #ever_put = False
        for backend_name, backend in self.storage_backends.items():
            put_task = backend.submit_put_task(key, memory_obj)

            if put_task is None:
                continue

        self.manager_lock.acquire()
        self.memory_allocator.ref_count_down(memory_obj)
        self.manager_lock.release()

    @_lmcache_nvtx_annotate
    def _update_hot_cache(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if memory_obj is None or not self.use_hot:
            return

        if memory_obj.tensor is not None and memory_obj.tensor.is_cuda:
            self.manager_lock.acquire()
            if key in self.hot_cache:
                self.manager_lock.release()
                return
            self.manager_lock.release()

            # Allocate a cpu memory object
            cpu_memory_obj = self.memory_allocator.allocate(
                memory_obj.get_shape(),
                memory_obj.get_dtype(),
                fmt=memory_obj.get_memory_format())

            if cpu_memory_obj is None:
                logger.warning(
                    "Memory allocation failed in cachegen deserializer")
                return None

            # Copy the tensor to the cpu memory object
            assert cpu_memory_obj.tensor is not None
            self.stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.stream):
                cpu_memory_obj.tensor.copy_(memory_obj.tensor,
                                            non_blocking=True)
            memory_obj.tensor.record_stream(self.stream)

            # Update the hot cache
            self.manager_lock.acquire()
            self.hot_cache[key] = cpu_memory_obj
            self.memory_allocator.ref_count_up(cpu_memory_obj)
            self.manager_lock.release()
            logger.debug("Updated hot cache!")
            return
        else:
            self.manager_lock.acquire()
            if self.use_hot and key not in self.hot_cache:
                self.hot_cache[key] = memory_obj
                self.memory_allocator.ref_count_up(memory_obj)
            self.manager_lock.release()

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
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
            self.memory_allocator.ref_count_up(memory_obj)
            self.hot_cache.move_to_end(key)
            self.manager_lock.release()
            return memory_obj

        self.manager_lock.release()

        # Search all backends for blocking get
        for backend_name, backend in self.storage_backends.items():
            # Avoid read-write contention
            #if key in self.put_tasks[backend_name]:
            #    continue

            # NOTE(Jiayi): bypass the allocator for now
            memory_obj = backend.get_blocking(key)
            if memory_obj is not None:
                self._update_hot_cache(key, memory_obj)
                return memory_obj

        return None

    # TODO(Jiayi): we need to consider eviction in prefetch
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

        # NOTE: no need to ref_count_up here because
        # the memory_obj's ref_count is already 1
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

    def close(self):

        # using threadsafe method here as stop modifies
        # the internal state of the loop (in another thread)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
        #logger.info("Storage manager closed.")

    def __del__(self):
        self.close()
