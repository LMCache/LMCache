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
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalGPUBackend(AllocatorBackendInterface):
    """
    A GPU-only storage backend that keeps KV cache in pre-allocated GPU memory.
    When enabled, LMCache will store and retrieve KV cache directly from GPU
    memory without forwarding to other backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("GPUBackend requires CUDA but CUDA is not available")
        super().__init__(dst_device)

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.use_hot = config.local_gpu
        assert metadata is not None or memory_allocator is not None
        self.memory_allocator = (
            self.initialize_allocator(config, metadata)  # type: ignore
            if memory_allocator is None
            else memory_allocator
        )
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.gpu_lock = threading.Lock()

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

        self.config = config
        self.metadata = metadata

        self.keys_in_request: List[CacheEngineKey] = []

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.gpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        with self.gpu_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        with self.gpu_lock:
            if key in self.hot_cache:
                return None

            memory_obj.ref_count_up()
            self.hot_cache[key] = memory_obj
            self.cache_policy.update_on_put(key)

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
        if not self.use_hot:
            return
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.gpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            memory_obj.ref_count_up()
            return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs = []
        with self.gpu_lock:
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
        num_hit_chunks = 0
        with self.gpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_chunks
                if pin:
                    self.hot_cache[key].pin()
                    self.keys_in_request.append(key)
                num_hit_chunks += 1
        return num_hit_chunks

    def pin(self, key: CacheEngineKey) -> bool:
        with self.gpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.gpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.gpu_lock.acquire()
        if key not in self.hot_cache:
            if force:
                self.gpu_lock.release()
            return False

        memory_obj = self.hot_cache.pop(key)
        memory_obj.ref_count_down()

        if force:
            self.gpu_lock.release()
        return True

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> MemoryAllocatorInterface:
        gpu_size = config.max_local_gpu_size
        if gpu_size <= 0:
            raise RuntimeError("max_local_gpu_size must be > 0 when local_gpu is True")
        return GPUMemoryAllocator(int(gpu_size * 1024**3), device=self.dst_device)

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        logger.debug(
            f"Allocating memory in local gpu backend with busy loop: {busy_loop}"
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
            wait_other_requests = True
            if self.use_hot:
                num_candidates = 1
                with self.gpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )
                    if evict_keys:
                        wait_other_requests = False
                        logger.debug(
                            f"Evicting {len(evict_keys)} chunks from gpu memory"
                        )
                        self.batched_remove(evict_keys, force=False)
                        evict_keys_count += len(evict_keys)
                    else:
                        logger.warning(
                            "No eviction candidates found in local gpu backend."
                        )

            if wait_other_requests and not busy_loop:
                break

            if wait_other_requests:
                time_to_wait = 0.1
                logger.warning(
                    "Local gpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                time.sleep(time_to_wait)

            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt)
            if memory_obj is not None:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local gpu backend allocate()"
            )
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
    ) -> Optional[list[MemoryObj]]:
        logger.debug(
            "Allocating batched memory in local gpu backend "
            f"with busy loop: {busy_loop}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_objs = self.memory_allocator.batched_allocate(shape, dtype, batch_size, fmt)
        if memory_objs is not None or not eviction:
            return memory_objs

        evict_keys_count = 0
        num_attempts = 0
        while True:
            wait_other_requests = True
            if self.use_hot:
                num_candidates = 1
                with self.gpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )
                    if evict_keys:
                        wait_other_requests = False
                        logger.debug(
                            f"Evicting {len(evict_keys)} chunks from gpu memory"
                        )
                        self.batched_remove(evict_keys, force=False)
                        evict_keys_count += len(evict_keys)
                    else:
                        logger.warning(
                            "No eviction candidates found in local gpu backend."
                        )

            if wait_other_requests and not busy_loop:
                break

            if wait_other_requests:
                time_to_wait = 0.1
                logger.warning(
                    "Local gpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                time.sleep(time_to_wait)

            memory_objs = self.memory_allocator.batched_allocate(
                shape, dtype, batch_size, fmt
            )
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local gpu backend batched_allocate()"
            )
        return memory_objs

    def calculate_chunk_budget(self) -> int:
        logger.debug("Attempting to calculate chunk budget for async loading")
        assert self.metadata is not None, (
            "metadata required for chunk budget calculation"
        )

        total_memory = int(self.config.max_local_gpu_size * 1024**3)
        chunk_tokens = self.config.chunk_size
        kv_shape = self.metadata.kv_shape
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize

        if self.layerwise:
            chunk_bytes = chunk_tokens * kv_size * hidden_dim * dtype_size
        else:
            chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        logger.debug(
            f"Stats received: num_layers={num_layers}, kv_size={kv_size}, "
            f"chunk_tokens={chunk_tokens}, head_dim={head_size}, "
            f"dtype_size={dtype_size}, hidden_dim={hidden_dim}"
        )
        logger.debug(f"Calculated bytes per chunk per rank: {chunk_bytes}")

        alignment = getattr(self.memory_allocator, "align_bytes", None)
        if alignment is None:
            alignment = getattr(
                getattr(self.memory_allocator, "allocator", None), "align_bytes", None
            )
        assert alignment is not None
        aligned_chunk_bytes = ((chunk_bytes + alignment - 1) // alignment) * alignment
        max_chunks = total_memory // aligned_chunk_bytes

        return max_chunks

    def get_keys(self) -> List[CacheEngineKey]:
        with self.gpu_lock:
            return list(self.hot_cache.keys())

    def clear(self) -> int:
        if not self.use_hot:
            return 0
        clear_keys = []
        num_cleared_tokens = 0
        with self.gpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()

        self.batched_remove(clear_keys)
        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        self.memory_allocator.close()
        self.clear()
