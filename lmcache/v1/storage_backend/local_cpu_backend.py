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
from typing import TYPE_CHECKING, List, Optional
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalCPUBackend(StorageBackendInterface):
    """
    The local cpu backend size is variable depending on how much free space is
    left in the allocator so we cannot use LRUEvictor().
    (max_local_cpu_size > 0 initializes the memory_allocator)
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        memory_allocator: MemoryAllocatorInterface,
        lookup_server: Optional[LookupServerInterface] = None,
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        self.hot_cache: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.use_hot = config.local_cpu
        self.lookup_server = lookup_server
        self.memory_allocator = memory_allocator
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        self.stream = torch.cuda.Stream()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """
        Synchronously put the MemoryObj into the local cpu backend.
        """

        with self.cpu_lock:
            if key in self.hot_cache:
                old_memory_obj = self.hot_cache.pop(key)
                old_memory_obj.ref_count_down()
            self.hot_cache[key] = memory_obj
            memory_obj.ref_count_up()

            self.usage += memory_obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            # TODO(Jiayi): optimize this with batching?
            # push kv admit msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )
        return None

    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> Optional[List[Future]]:
        """
        Synchronously put the MemoryObjs into the local cpu backend.
        """
        if not self.use_hot:
            return None

        # TODO(Jiayi): optimize this with batching
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

        return None

    # NOTE (Jiayi): prefetch might be deprecated in the future.
    # Should be replaced by `move`.
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        return None

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            memory_obj.ref_count_up()
            self.hot_cache.move_to_end(key)
            return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Return the dummy future object.
        """
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            memory_obj.ref_count_up()
            self.hot_cache.move_to_end(key)
            f: Future = Future()
            f.set_result(memory_obj)
            return f

    def pin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, free_obj=True) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache.pop(key)
            if free_obj:
                memory_obj.ref_count_down()

            self.usage -= memory_obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )
            # NOTE (Jiayi): This `return True` might not accurately reflect
            # whether the key is removed from the actual memory because
            # other backends might still (temporarily) hold the memory object.
            return True

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
        if memory_obj is not None or not eviction:
            return memory_obj

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        evict_keys = []
        with self.cpu_lock:
            for evict_key in self.hot_cache:
                old_mem_obj = self.hot_cache[evict_key]
                # If the ref_count > 1, we cannot evict it as the cpu memory
                # might be used as buffers by other storage backends
                if old_mem_obj.get_ref_count() > 1:
                    continue
                evict_keys.append(evict_key)

                old_mem_obj.ref_count_down()
                memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
                logger.debug("Evicting 1 chunk from cpu memory")
                if memory_obj is not None:
                    break
        for evict_key in evict_keys:
            self.remove(evict_key)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_objs = self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt
        )
        if memory_objs is not None or not eviction:
            return memory_objs

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        # NOTE: Tune this number for performance.
        # Setting it to small will cause more eviction overhead.
        # Setting it to large might result in lower cache hit
        # because more caches are evicted.
        # blocks_to_free = batch_size

        evict_keys = []
        old_mem_objs = []
        with self.cpu_lock:
            for evict_key in self.hot_cache:
                if evict_key in evict_keys:
                    continue
                old_mem_obj = self.hot_cache[evict_key]
                # If the ref_count > 1, we cannot evict it as the cpu memory
                # might be used as buffers by other storage backends
                if old_mem_obj.get_ref_count() > 1 and not old_mem_obj.is_pinned:
                    continue
                # HACK: We assume batch_size=num_layers here.
                # We also assume if the one layer's ref_count > 1 or pinned,
                # then the other layers are also ref_count > 1 or
                # pinned in the cpu memory.
                evict_key_all_layer = evict_key.split_layers(batch_size)
                evict_keys.extend(evict_key_all_layer)
                for key in evict_key_all_layer:
                    old_mem_objs.append(self.hot_cache[key])

                # if len(old_mem_objs) < blocks_to_free:
                #    continue

                self.memory_allocator.batched_free(old_mem_objs)
                memory_objs = self.memory_allocator.batched_allocate(
                    shape, dtype, batch_size, fmt
                )

                logger.debug(f"Evicting {len(old_mem_objs)} chunks from cpu memory")

                if memory_objs is not None:
                    break
                old_mem_objs = []
        for evict_key in evict_keys:
            self.remove(evict_key, free_obj=False)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)
        return memory_objs

    def write_back(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if memory_obj is None or not self.use_hot:
            return

        if memory_obj.tensor is not None and memory_obj.tensor.is_cuda:
            self.cpu_lock.acquire()
            if key in self.hot_cache:
                self.cpu_lock.release()
                return
            self.cpu_lock.release()

            # Allocate a cpu memory object
            cpu_memory_obj = self.memory_allocator.allocate(
                memory_obj.get_shape(),
                memory_obj.get_dtype(),
                fmt=memory_obj.get_memory_format(),
            )

            if cpu_memory_obj is None:
                logger.warning("Memory allocation failed in cachegen deserializer")
                return None

            # Copy the tensor to the cpu memory object
            assert cpu_memory_obj.tensor is not None
            self.stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.stream):
                cpu_memory_obj.tensor.copy_(memory_obj.tensor, non_blocking=True)
            memory_obj.tensor.record_stream(self.stream)

            # Update the hot cache
            self.cpu_lock.acquire()
            self.hot_cache[key] = cpu_memory_obj
            cpu_memory_obj.ref_count_up()
            self.cpu_lock.release()

            # Push kv msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "cpu")
                )

            logger.debug("Updated hot cache!")
        else:
            self.cpu_lock.acquire()
            if self.use_hot and key not in self.hot_cache:
                self.hot_cache[key] = memory_obj
                memory_obj.ref_count_up()
                self.cpu_lock.release()

                # Push kv msg
                if self.lmcache_worker is not None:
                    self.lmcache_worker.put_msg(
                        KVAdmitMsg(
                            self.instance_id,
                            key.worker_id,
                            key.chunk_hash,
                            "cpu",
                        )
                    )
            else:
                self.cpu_lock.release()

    def get_keys(self) -> List[CacheEngineKey]:
        """
        array ordering of keys from LRU to MRU
        """
        with self.cpu_lock:
            return list(self.hot_cache.keys())

    def clear(self) -> int:
        """
        counts the number of memory objects removed
        """
        if not self.use_hot:
            return 0
        clear_keys = []
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if memory_obj.get_ref_count() > 1:
                    continue
                clear_keys.append(key)
                memory_obj.ref_count_down()

        for key in clear_keys:
            self.remove(key)

        if self.lookup_server is not None:
            self.lookup_server.batched_remove(clear_keys)

        return len(clear_keys)

    def close(self) -> None:
        self.clear()
