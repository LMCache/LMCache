import asyncio
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj,
                                                    MixedMemoryAllocator, BytesBufferMemoryObj)
from lmcache.experimental.storage_backend import CreateStorageBackends
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

from dataclasses import dataclass

from lmcache.experimental.storage_backend.naive_serde.kivi_serde import (
    KIVIDeserializer, KIVISerializer)

logger = init_logger(__name__)

@dataclass
class KVDecision:
    device: str
    compression_method: str
    compression_rate: float

# TODO(Shaoting): add freqency estimator
class KVCacheManager:
    def __init__(self, hot_cache: OrderedDict[CacheEngineKey, MemoryObj]):
        # NOTE(Shaoting): policy related variables define here
        self.method = "ours"
        self.rate = 0
        self.cpu_size = 5368709120 * 6 # 30 GB

        self.hot_cache = hot_cache

    def inform_new(self, key: CacheEngineKey):
        size = key.metadata.length
        # TODO(Shaoting): add other manager logics
        if self.method == "baseline_KIVI":
            size_kv_cpu = sum(key.metadata.length for key in self.hot_cache.keys())
            if size_kv_cpu + size > self.cpu_size:
                return KVDecision("cpu", "kivi", 0), {} #hahaha
            else:
                return KVDecision("cpu", "kivi", self.rate), {}
            
        elif self.method == "ours":

            # TODO(Shaoting): save unit quality drop to speed up decisions. Also need to update the storage when retrieval.
            size_kv_cpu = sum(key.metadata.length for key in self.hot_cache.keys())
            size_kv_cpu += size

            final_drop_list = {}
            new_kv_rate = key.metadata.rate

            # If cpu is full
            while size_kv_cpu > self.cpu_size: 

                drop_list = {}
                min_quality_drop = 0
                
                # First calculate unit quality drop of the new cache
                if new_kv_rate != 0:
                    for i in range(len(key.metadata.context_id)):
                        for ii, (rate, score) in enumerate(key.metadata.score_table[i]):
                            # NOTE(Shaoting): ">=" is used here to handle "one chunk multiple method" scenario
                            if new_kv_rate >= rate:
                                next_rate, next_score = key.metadata.score_table[i][ii + 1]
                                min_quality_drop += (score - next_score) / (key.metadata.length / key.metadata.rate * (key.metadata.rate - next_rate)) * (10**9)
                                # If is the first score_table
                                if i == 0:
                                    chosen_rate = next_rate
                                break
                    # -1 represents the new cache
                    drop_list[-1] = chosen_rate

                # Then calculate the unit quality drop of each cache in the hot cache
                for hot_cache_key in self.hot_cache.keys():

                    unit_quality_drop = 0
                    
                    if hot_cache_key in final_drop_list:
                        current_rate = final_drop_list[hot_cache_key]
                    else:
                        current_rate = hot_cache_key.metadata.rate
                    if current_rate == 0:
                        continue

                    for i in range(len(hot_cache_key.metadata.context_id)):
                        for ii, (rate, score) in enumerate(hot_cache_key.metadata.score_table[i]):
                            # NOTE(Shaoting): ">=" is used here to handle "one chunk multiple method" scenario
                            if current_rate >= rate:
                                next_rate, next_score = hot_cache_key.metadata.score_table[i][ii + 1]
                                unit_quality_drop += (score - next_score) / (hot_cache_key.metadata.length / hot_cache_key.metadata.rate * (hot_cache_key.metadata.rate - next_rate)) * (10**9)
                                # If is the first score_table
                                if i == 0:
                                    chosen_rate = next_rate
                                break

                    # Update the drop list
                    if unit_quality_drop < min_quality_drop:
                        min_quality_drop = unit_quality_drop
                        drop_list = {}
                        drop_list[hot_cache_key] = chosen_rate
                    elif unit_quality_drop == min_quality_drop:
                        drop_list[hot_cache_key] = chosen_rate

                # Handle the drop_list
                for idx in drop_list:
                    if idx == -1:
                        size_kv_cpu = size_kv_cpu - key.metadata.length / key.metadata.rate * (1 - drop_list[idx])
                        new_kv_rate = drop_list[idx]
                    else:
                        size_kv_cpu = size_kv_cpu - idx.metadata.length / idx.metadata.rate * (1 - drop_list[idx])
                        final_drop_list[idx] = drop_list[idx]

                # TODO(Shaoting): check disk
            return KVDecision("cpu", key.metadata.method[0], new_kv_rate), final_drop_list
        else:
            return KVDecision("cpu", "kivi", 0.6), {}

# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 allocator: MemoryAllocatorInterface):
        self.memory_allocator = allocator
        self.kivi_de = KIVIDeserializer(self.memory_allocator)
        self.hot_cache: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()
        self.kivi_cache = OrderedDict()
        self.use_hot = config.local_cpu
        self.kivi_ser = KIVISerializer(self.memory_allocator)

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

        self.manager = KVCacheManager(self.hot_cache)

        self.update_queue: OrderedDict[CacheEngineKey, MemoryObj] = OrderedDict()

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
    
    def put_in_queue(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Put the memory object into the queue.
        """
        self.update_queue[key] = memory_obj

    def update(self) -> None:  
        """
        Update the hot cache and storage backends.
        """
        for key, memory_obj in self.update_queue.items():
            self.put(key, memory_obj)
        self.update_queue.clear()

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
        size_in_bytes = memory_obj.get_size()
        key.metadata.length = size_in_bytes

        self.manager_lock.acquire()

        current_kv_decision, update_decision = self.manager.inform_new(key)
        
        # TODO(Shaoting): update_decision should be in a separate thread: thread 1, decode; thread 2, update and store
        for update_key in update_decision:
            update_memory_obj = self.hot_cache.pop(update_key)
            update_rate = update_decision[update_key]

            if update_rate == 0:
                # Move to disk
                # self.manager_lock.release()
                # for backend_name, backend in self.storage_backends.items():
                #     put_task = backend.submit_put_task(update_key, update_memory_obj)

                #     if put_task is None:
                #         continue
                # self.manager_lock.acquire() #hahaha
                self.memory_allocator.ref_count_down(update_memory_obj)
                continue

            # KIVI mapping defined here
            if update_rate == 0.6:
                BITS = 8
            elif update_rate == 0.3:
                BITS = 4
            elif update_rate == 0.2:
                BITS = 2

            # Need to deserialize first
            if update_key.metadata.rate != 1:
                # KIVI mapping defined here
                if update_key.metadata.rate == 0.6:
                    DE_BITS = 8
                elif update_key.metadata.rate == 0.3:
                    DE_BITS = 4
                elif update_key.metadata.rate == 0.2:
                    DE_BITS = 2
                update_memory_kv_cache = self.kivi_de.deserialize(update_memory_obj, DE_BITS, self.kivi_cache[update_key][0], self.kivi_cache[update_key][1], self.kivi_cache[update_key][2], self.kivi_cache[update_key][3], self.kivi_cache[update_key][4])
                self.memory_allocator.ref_count_down(update_memory_obj)
                update_memory_obj = update_memory_kv_cache
        
            compressed_update_memory_obj, metadata, entry_offsets, split_metadata, quant_metadata, quant_entry_offsets = self.kivi_ser.serialize(update_memory_obj, BITS)
            if type(update_memory_obj) != Tensor:
                self.memory_allocator.ref_count_down(update_memory_obj)

            # Move memory obj from tmp buffer to real location
            self.manager_lock.release()
            compressed_blank_memory_obj = self.allocate(
                compressed_update_memory_obj.get_shape(),
                compressed_update_memory_obj.get_dtype())
            self.manager_lock.acquire()
            # NOTE(Shaoting): Extra memory copy here
            compressed_blank_memory_obj.tensor.copy_(compressed_update_memory_obj.tensor)
            self.memory_allocator.ref_count_down(compressed_update_memory_obj)
            compressed_update_memory_obj = compressed_blank_memory_obj 

            # Update key
            update_key.metadata.rate = update_decision[update_key]
            update_key.metadata.length = compressed_update_memory_obj.get_physical_size()
            
            self.hot_cache[update_key] = compressed_update_memory_obj
            self.kivi_cache[update_key] = (metadata, entry_offsets, split_metadata, quant_metadata, quant_entry_offsets)

        # TODO(Shaoting): compress memory_obj with cachegen and streamingllm
        if current_kv_decision.compression_method == "cachegen":
            pass
        elif current_kv_decision.compression_method == "kivi" and current_kv_decision.compression_rate != 1 and current_kv_decision.compression_rate != 0:

            # KIVI mapping defined here
            if current_kv_decision.compression_rate == 0.6:
                BITS = 8
            elif current_kv_decision.compression_rate == 0.3:
                BITS = 4
            elif current_kv_decision.compression_rate == 0.2:
                BITS = 2
        
            # NOTE(Shaoting): KV Cache that's less than 256 tokens will have no compression
            # Update memory obj
            compressed_memory_obj, metadata, entry_offsets, split_metadata, quant_metadata, quant_entry_offsets = self.kivi_ser.serialize(memory_obj, BITS)
            self.memory_allocator.ref_count_down(memory_obj)
            memory_obj = compressed_memory_obj

            # Update key
            key.metadata.rate = current_kv_decision.compression_rate
            key.metadata.length = memory_obj.get_physical_size()
            self.kivi_cache[key] = (metadata, entry_offsets, split_metadata, quant_metadata, quant_entry_offsets)
        elif current_kv_decision.compression_method == "streamingllm":
            pass

        if self.use_hot and current_kv_decision.device == "cpu" and current_kv_decision.compression_rate != 0:
            # During overwrite, we need to free the old memory object
            # to avoid memory leak.
            # NOTE(Jiayi): overwrite should not happen, at least for
            # prefix caching
            if key in self.hot_cache:
                old_memory_obj = self.hot_cache.pop(key)
                self.memory_allocator.ref_count_down(old_memory_obj)

            # Move memory obj from tmp buffer to real location
            self.manager_lock.release()
            blank_memory_obj = self.allocate(
                memory_obj.get_shape(),
                memory_obj.get_dtype())
            self.manager_lock.acquire()
            # NOTE(Shaoting): Extra memory copy here
            blank_memory_obj.tensor.copy_(memory_obj.tensor)
            self.memory_allocator.ref_count_down(memory_obj)
            memory_obj = blank_memory_obj 

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

        # TODO(Shaoting): add third tier storage (remote ssd)
        # if current_kv_decision.device == "disk" or (current_kv_decision.device == "cpu" and current_kv_decision.compression_rate == 0): #hahaha
        if current_kv_decision.device == "disk":
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
    
                # Move memory obj from tmp buffer to real location
                self.manager_lock.release()
                blank_memory_obj = self.allocate(
                    memory_obj.get_shape(),
                    memory_obj.get_dtype())
                self.manager_lock.acquire()
                blank_memory_obj.raw_data.copy_(memory_obj.raw_data, non_blocking=True)
                blank_memory_obj.valid = memory_obj.valid
                self.memory_allocator.ref_count_down(memory_obj)
                memory_obj = blank_memory_obj

                self.hot_cache[key] = memory_obj
                self.memory_allocator.ref_count_up(memory_obj)
            self.manager_lock.release()

    def get(self, key: CacheEngineKey) -> Optional[Union[MemoryObj, Tensor]]:
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

        # Customed get function
        memory_obj = None
        for old_key, value in self.hot_cache.items():
            if old_key == key:
                memory_obj = value
                break

        if memory_obj is not None:
            self.memory_allocator.ref_count_up(memory_obj)

            # NOTE(Shaoting): didn't think about partial prefill. This may cause calculate unit quality drop to be wrong. Maybe not.
            # Update key
            if key.metadata.context_id[0] not in old_key.metadata.context_id:
                old_key.metadata.context_id.append(key.metadata.context_id[0])
                old_key.metadata.method.append(key.metadata.method[0])
                old_key.metadata.score_table.append(key.metadata.score_table[0])
                self.hot_cache[old_key] = self.hot_cache.pop(key)

            self.manager_lock.release()

            # De-compress memory_obj
            if old_key.metadata.method[0] == "kivi" and old_key.metadata.rate != 1:  

                # KIVI mapping defined here
                if old_key.metadata.rate == 0.6:
                    BITS = 8
                elif old_key.metadata.rate == 0.3:
                    BITS = 4
                elif old_key.metadata.rate == 0.2:
                    BITS = 2

                memory_obj = self.kivi_de.deserialize(memory_obj, BITS, self.kivi_cache[old_key][0], self.kivi_cache[old_key][1], self.kivi_cache[old_key][2], self.kivi_cache[old_key][3], self.kivi_cache[old_key][4]) 

            return memory_obj

        self.manager_lock.release()

        # Search all backends for blocking get
        for backend_name, backend in self.storage_backends.items():
            # Avoid read-write contention
            #if key in self.put_tasks[backend_name]:
            #    continue

            # NOTE(Jiayi): bypass the allocator for now
            memory_obj, new_key = backend.get_blocking(key)
            if memory_obj is not None:

                # De-compress memory_obj
                if new_key.metadata.method[0] == "kivi" and new_key.metadata.rate != 1:  

                    # KIVI mapping defined here
                    if new_key.metadata.rate == 0.6:
                        BITS = 8
                    elif new_key.metadata.rate == 0.3:
                        BITS = 4
                    elif new_key.metadata.rate == 0.2:
                        BITS = 2

                    memory_obj = self.kivi_de.deserialize(memory_obj, BITS)            

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
