# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, List, Optional, Sequence
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.system_detection import NUMADetector, SystemMemoryDetector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalCPUBackend(AllocatorBackendInterface):
    """
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.use_hot = config.local_cpu
        # NOTE: we keep the memory allocator argument for temporary
        # test compatibility
        # TODO: fix the tests to get rid the memory allocator
        assert metadata is not None or memory_allocator is not None
        self.memory_allocator = (
            self.initialize_allocator(config, metadata)  # type: ignore
            if memory_allocator is None
            else memory_allocator
        )
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

        # Store config and metadata for chunk budget calculation
        self.config = config
        self.metadata = metadata

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []
        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.local_cpu_hot_cache_count.set_function(
                lambda: len(self.hot_cache)
            )
            prometheus_logger.local_cpu_keys_in_request_count.set_function(
                lambda: len(self.keys_in_request)
            )

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
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """
        Synchronously put the MemoryObjs into the local cpu backend.
        """
        if not self.use_hot:
            return

        # TODO(Jiayi): optimize this with batching
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

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

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs = []
        with self.cpu_lock:
            for key in keys:
                mem_obj = self.hot_cache[key]
                mem_obj.ref_count_up()
                mem_objs.append(mem_obj)
        return mem_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        # NOTE(Jiayi): Only prefix chunks are counted.
        num_hit_chunks = 0
        with self.cpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_chunks
                if pin:
                    self.hot_cache[key].pin()
                    # vllm lookup sets pin to True
                    self.keys_in_request.append(key)
                num_hit_chunks += 1
        return num_hit_chunks

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

        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )
        # NOTE (Jiayi): This `return True` might not accurately reflect
        # whether the key is removed from the actual memory because
        # other backends might still (temporarily) hold the memory object.
        return True

    def _calculate_effective_cpu_size(
        self,
        configured_cpu_size: float,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> float:
        """
        Calculate the effective CPU memory size based on system available memory
        and reserve memory configuration.

        Args:
            configured_cpu_size: The configured CPU memory size in GB
            config: The LMCache engine configuration
            metadata: Optional metadata for first rank handling

        Returns:
            The effective CPU memory size in GB
        """

        save_only_first_rank = (
            metadata is not None
            and config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if not save_only_first_rank:
            # Do not adjust cpu_size if save_only_first_rank is False for now
            return configured_cpu_size

        # Get the system available memory and calculate effective cpu_size
        system_available_memory_gb = SystemMemoryDetector.get_available_memory_gb()
        # Get reserve memory size from config
        reserve_cpu_size = config.reserve_local_cpu_size

        # TODO(baoloongmao): For disable save_only_first_rank case,
        #  we need to avoid multi-rank race condition in future.
        #  But for enable save_only_first_rank case,
        #  we can handle reserve memory simply since non-first ranks
        #  do not allocate memory.
        # Effective memory: min(configured_size, available_memory - reserve_size)
        if system_available_memory_gb > 0:
            max_usable_memory = max(0, system_available_memory_gb - reserve_cpu_size)
            effective_cpu_size = min(configured_cpu_size, max_usable_memory)
            logger.info(
                f"Adjusted CPU memory size from {configured_cpu_size:.2f} GB "
                f"to {effective_cpu_size:.2f} GB "
                f"(system available: {system_available_memory_gb:.2f} GB, "
                f"reserve: {reserve_cpu_size:.2f} GB)"
            )
            assert effective_cpu_size > 0
            return effective_cpu_size
        else:
            logger.warning(
                "Could not determine system available memory, using configured cpu_size"
            )
            return configured_cpu_size

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> MemoryAllocatorInterface:
        cpu_size = config.max_local_cpu_size

        if metadata is not None:
            # save_only_first_rank only works when use mla
            save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )

            if save_only_first_rank and metadata.is_first_rank():
                # Only the first rank will save the cache,
                # so we need to set it larger than other ranks
                cpu_size = config.get_extra_config_value(
                    "first_rank_max_local_cpu_size", cpu_size
                )

        # Detect the numa mapping
        numa_mapping = NUMADetector.get_numa_mapping(config)
        logger.info(f"NUMA mapping {numa_mapping}")

        # Calculate effective CPU memory size
        cpu_size = self._calculate_effective_cpu_size(cpu_size, config, metadata)

        if config.enable_p2p:
            assert metadata is not None
            meta_shape = torch.Size(metadata.kv_shape)
            # TODO(Jiayi): remove this hardcode
            new_shape = torch.Size(
                [
                    meta_shape[1],
                    meta_shape[0],
                    meta_shape[2],
                    meta_shape[3] * meta_shape[4],
                ]
            )
            paged_mem_allocator = PagedCpuGpuMemoryAllocator()
            paged_mem_allocator.init_cpu_memory_allocator(
                int(cpu_size * 1024**3),
                shape=new_shape,
                dtype=metadata.kv_dtype,
                fmt=MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
                numa_mapping=numa_mapping,
            )
            return paged_mem_allocator
        else:
            return MixedMemoryAllocator(
                int(cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        logger.debug(
            f"Allocating memory in local cpu backend with busy loop: {busy_loop}"
        )
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

        evict_keys_count = 0
        num_attempts = 0
        while True:
            # whether or not this request needs to wait or other requests
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )
                    if evict_keys:
                        # we can continue trying to evict from the hot_cache
                        # and don't need to wait for other requests yet
                        wait_other_requests = False
                        logger.debug(
                            f"Evicting {len(evict_keys)} chunks from cpu memory"
                        )
                        # remove
                        self.batched_remove(evict_keys, force=False)
                        evict_keys_count += len(evict_keys)
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if memory_obj is not None:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend allocate()"
            )

        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False
        """
        logger.debug(
            f"Batched allocating memory in local cpu backend"
            f" with busy loop: {busy_loop}"
        )
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

        evict_keys_count = 0
        num_attempts = 0
        while True:
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )

                    # HACK: We assume batch_size=num_layers here.
                    # FIXME: We also assume if the one layer's ref_count > 1 or pinned,
                    # then the other layers are also ref_count > 1 or
                    # pinned in the cpu memory. This might not be true.
                    if evict_keys:
                        evict_keys_count += len(evict_keys)
                        wait_other_requests = False
                        for evict_key in evict_keys:
                            evict_key_all_layer = evict_key.split_layers(batch_size)

                            # TODO(Jiayi): batched allocate is not supported through
                            # `batched_remove`. Therefore, features like usage tracking
                            # is not supported.
                            old_mem_objs = []
                            for key in evict_key_all_layer:
                                old_mem_objs.append(self.hot_cache[key])
                                self.cache_policy.update_on_force_evict(key)
                                self.hot_cache.pop(key, None)

                            self.memory_allocator.batched_free(old_mem_objs)

                            logger.debug(
                                f"Evicting {len(old_mem_objs)} chunks from cpu memory"
                            )
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_objs = self.memory_allocator.batched_allocate(
                shape, dtype, batch_size, fmt
            )
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend batched_allocate()"
            )
        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_objs

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the maximum number of chunks that can be allocated concurrently
        without causing memory deadlocks in the async loading system.

        Returns:
            int: The estimated chunk budget for concurrent allocations
        """
        logger.info("Attempting to calculate chunk budget for async loading")
        assert self.metadata is not None, (
            "metadata required for chunk budget calculation"
        )

        total_memory = int(self.config.max_local_cpu_size * 1024**3)
        chunk_tokens = self.config.chunk_size
        # already accounted for parallelism
        kv_shape = (
            self.metadata.kv_shape
        )  # [num_layers, kv_size, chunk_size, num_heads, head_size]
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]  # 1 for MLA, 2 for regular
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize

        if self.layerwise:
            # layerwise: [chunk_tokens, kv_size, hidden_dim]
            chunk_bytes = chunk_tokens * kv_size * hidden_dim * dtype_size
        else:
            # full: [kv_size, num_layers, chunk_tokens, hidden_dim]
            chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        logger.info(
            f"Stats received: num_layers={num_layers}, kv_size={kv_size}, "
            f"chunk_tokens={chunk_tokens}, head_dim={head_size}, "
            f"dtype_size={dtype_size}, "
            f"hidden_dim={hidden_dim}"
        )
        logger.info(f"Calculated bytes per chunk per rank: {chunk_bytes}")
        # add alignment overhead
        # (MixedMemoryAllocator uses TensorMemoryAllocator with 4KB alignment)
        assert hasattr(self.memory_allocator, "align_bytes")
        alignment = self.memory_allocator.align_bytes
        aligned_chunk_bytes = ((chunk_bytes + alignment - 1) // alignment) * alignment

        # calculate budget with safety margin
        max_chunks = total_memory // aligned_chunk_bytes

        chunk_budget = int(max_chunks)
        return chunk_budget

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
                if not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()

        # TODO(Jiayi): might not be accurate if we don't calculate
        # `num_cleared_token` and remove the keys in an atomic way.
        self.batched_remove(clear_keys)

        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        self.memory_allocator.close()
        self.clear()
