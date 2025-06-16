# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
)
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        allocator: MemoryAllocatorInterface,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        dst_device = "cuda"
        # FIXME (Jiayi): The allocator is a dummy allocator in nixl for now.
        # The real allocator is initialized inside the NixlBackend.
        self.storage_backends: OrderedDict[str, StorageBackendInterface] = (
            CreateStorageBackends(
                config,
                metadata,
                self.loop,
                allocator,
                dst_device,
                lmcache_worker,
                lookup_server,
            )
        )

        if config.enable_nixl:
            self.allocator_backend = self.storage_backends["NixlBackend"]
        else:
            self.allocator_backend = self.storage_backends["LocalCPUBackend"]

        self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}

        self.manager_lock = threading.Lock()

        self.lookup_server = lookup_server

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.stream = torch.cuda.Stream()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.allocate(shape, dtype, fmt, eviction=eviction)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
    ) -> Optional[MemoryObj]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.batched_allocate(
            shape, dtype, batch_size, fmt, eviction=eviction
        )

    # FIXME: Should be deprecated
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

        # TODO(Jiayi): currently, the entire put task will be cancelled
        # if one of the backend is already storing this cache.
        # This might not be ideal. We need a caching policy to
        # configure caching policies (e.g., write-through,
        # write-back, etc.)
        for storage_backend in self.storage_backends.values():
            if storage_backend.exists_in_put_tasks(key):
                memory_obj.ref_count_down()
                return

        for backend_name, backend in self.storage_backends.items():
            backend.submit_put_task(key, memory_obj)

        memory_obj.ref_count_down()

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        # FIXME(Jiayi): fix docstring
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """

        # TODO(Jiayi): currently, the cache is stored to a certain
        # backend if this backend does not have this cache.
        # There's no way to configure a global caching policy
        # among different storage backends.
        for backend in self.storage_backends.values():
            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            backend.batched_submit_put_task(keys, memory_objs)

        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

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
            logger.debug(
                "Waiting for prefetching result. Optimally, this should not happen."
            )
            # Calling result() twice (already once in callback) will have
            # no effect
            # Tune the timeout for better performance
            prefetch_task.result(timeout=1)

        # Search all backends for blocking get
        for backend_name, backend in self.storage_backends.items():
            # NOTE(Jiayi): bypass the allocator for now
            memory_obj = backend.get_blocking(key)
            if memory_obj is not None:
                if backend_name not in ["LocalCPUBackend", "NixlBackend"]:
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    local_cpu_backend.write_back(key, memory_obj)
                return memory_obj

        return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.storage_backends.items():
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task is not None:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
    ) -> Generator[List[Future], None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a list of futures for each layer.
        """
        for keys_multi_chunk in keys:
            # Store all chunks for one layer
            tasks = []
            for key in keys_multi_chunk:
                task = self.get_non_blocking(key)
                assert task is not None
                tasks.append(task)
            yield tasks

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
            logger.error(f"Exception captured from future in prefetch_callback: {e}")
            raise e
        kv_chunk = buffer_memory_obj.tensor
        kv_shape = kv_chunk.shape
        kv_dtype = kv_chunk.dtype
        memory_obj = self.allocator_backend.allocate(kv_shape, kv_dtype)
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

        # NOTE: no need to ref_count_up here because
        # the memory_obj's ref_count is already 1
        self.manager_lock.acquire()
        self.storage_backends["LocalCPUBackend"].submit_put_task(key, memory_obj)
        self.manager_lock.release()

    def prefetch(self, key: CacheEngineKey) -> None:
        """Launch a prefetch request in the storage backend. Non-blocking"""

        if self.storage_backends["LocalCPUBackend"].contains(key):
            return
        self.manager_lock.acquire()
        if key in self.prefetch_tasks:
            self.manager_lock.release()
            return
        self.manager_lock.release()

        for backend in self.storage_backends.values():
            prefetch_task = backend.submit_prefetch_task(key)
            if prefetch_task is None:
                continue
            lambda_callback = lambda f: self.prefetch_callback(f, key)

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
        pin: bool = False,
    ) -> bool:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.storage_backends.items():
            if search_range is not None and backend_name not in search_range:
                continue

            if backend.contains(key, pin):
                return True

        return False

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or "LocalCPUBackend" in locations:
                assert hasattr(backend, "remove")
                num_removed += backend.remove(key)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                for key in keys:
                    backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared caches in the specified
        storage backends.
        """

        num_cleared = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared += backend.clear()
                else:
                    logger.warning(
                        f"Storage backend {backend_name} does not support "
                        "clear operation. Skipping."
                    )

        return num_cleared

    def close(self):
        for backend in self.storage_backends.values():
            backend.close()

        # using threadsafe method here as stop modifies
        # the internal state of the loop (in another thread)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()

        logger.info("Storage manager closed.")
