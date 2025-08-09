# SPDX-License-Identifier: Apache-2.0
# Standard
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
    NixlCPUMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalCPUBackend(StorageBackendInterface):
    """
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
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

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

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cpu_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

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
                return None

            memory_obj.ref_count_up()
            self.hot_cache[key] = memory_obj

            self.cache_policy.update_on_put(key)

            self.usage += memory_obj.get_size()
            self.stats_monitor.update_local_cache_usage(self.usage)

            # TODO(Jiayi): optimize this with batching?
            # push kv admit msg
            if self.lmcache_worker is not None:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(
                        self.instance_id, key.worker_id, key.chunk_hash, str(self)
                    )
                )
        return None

    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
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
    ) -> bool:
        return False

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

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.cpu_lock.acquire()
        if key not in self.hot_cache:
            if force:
                self.cpu_lock.release()
            return False

        memory_obj = self.hot_cache.pop(key)
        memory_obj.ref_count_down()

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.cpu_lock.release()

        self.usage -= memory_obj.get_size()
        self.stats_monitor.update_local_cache_usage(self.usage)

        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
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

        assert isinstance(self.memory_allocator, MixedMemoryAllocator) or isinstance(
            self.memory_allocator, NixlCPUMemoryAllocator
        )

        with self.cpu_lock:
            while True:
                # TODO(Jiayi): optimize `num_cacndidates` with estimation.
                # Accurate estimation is hard due to fragmentation.
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.hot_cache, num_candidates=1
                )

                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found in local cpu backend. "
                        "Local cpu memory is under pressure."
                    )
                    break

                self.batched_remove(evict_keys, force=False)

                # TODO(Jiayi): Move this inside `batched_remove`
                if self.lookup_server is not None:
                    self.lookup_server.batched_remove(evict_keys)

                memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
                logger.debug(f"Evicting {len(evict_keys)} chunk from cpu memory")
                if memory_obj is not None:
                    break

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

        assert isinstance(self.memory_allocator, MixedMemoryAllocator) or isinstance(
            self.memory_allocator, NixlCPUMemoryAllocator
        )

        with self.cpu_lock:
            while True:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.hot_cache, num_candidates=1
                )

                # HACK: We assume batch_size=num_layers here.
                # FIXME: We also assume if the one layer's ref_count > 1 or pinned,
                # then the other layers are also ref_count > 1 or
                # pinned in the cpu memory. This might not be true.

                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found in local cpu backend. "
                        "Local cpu memory is under pressure."
                    )
                    break

                for evict_key in evict_keys:
                    evict_key_all_layer = evict_key.split_layers(batch_size)

                    # TODO(Jiayi): batched allocate is not supported through
                    # `batched_remove`. Therefore, features like usage tracking
                    # is not supported.
                    old_mem_objs = []
                    for key in evict_key_all_layer:
                        old_mem_objs.append(self.hot_cache[key])
                        self.cache_policy.update_on_force_evict(key)

                    self.memory_allocator.batched_free(old_mem_objs)
                    self.hot_cache.pop(evict_key, None)

                    if self.lookup_server is not None:
                        self.lookup_server.batched_remove(evict_key_all_layer)

                    logger.debug(f"Evicting {len(old_mem_objs)} chunks from cpu memory")

                memory_objs = self.memory_allocator.batched_allocate(
                    shape, dtype, batch_size, fmt
                )

                if memory_objs:
                    break
        return memory_objs

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
        num_cleared_tokens = 0
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()

        # TODO(Jiayi): might not be accurate if we don't calculate
        # `num_cleared_token` and remove the keys in an atomic way.
        self.batched_remove(clear_keys)

        if self.lookup_server is not None:
            self.lookup_server.batched_remove(clear_keys)

        return num_cleared_tokens

    def close(self) -> None:
        self.clear()
