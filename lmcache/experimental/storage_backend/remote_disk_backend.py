import asyncio
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import List, Optional, Tuple, Union
import time
import aiofiles
import torch

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryAllocatorInterface, MemoryObj, MemoryFormat, BytesBufferMemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.logging import init_logger
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata,
                           _lmcache_nvtx_annotate)

logger = init_logger(__name__)


class RemoteDiskBackend(StorageBackendInterface):

    def __init__(self,
                 config: LMCacheEngineConfig,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface,
                 dst_device: str = "cuda"):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.dst_device = dst_device

        self.disk_lock = threading.Lock()
        assert config.remote_disk is not None
        self.path: str = config.remote_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created remote disk cache directory: {self.path}")

        # Initialize the evictor
        self.evictor = LRUEvictor(max_cache_size=config.max_remote_disk_size)

        self.loop = loop
        self.put_tasks: List[CacheEngineKey] = []

        self.memory_allocator = memory_allocator
        self.policy = config.policy

    def __str__(self):
        return self.__class__.__name__

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return self.path + key.to_string().replace("/", "-") + ".pt"

    def contains(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.dict

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.put_tasks

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        
        logger.info(f"Manually removing {key} from disk cache.")
        logger.info(f"Then disk cache size: {self.evictor.current_cache_size} bytes")

        path = self.dict[key].path
        self.disk_lock.acquire()
        self.dict.pop(key)
        self.disk_lock.release()
        os.remove(path)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)

            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:

        # Update cache recency
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, memory_obj.get_physical_size())
        if put_status == PutStatus.ILLEGAL:
            return None
        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        self.memory_allocator.ref_count_up(memory_obj)

        self.disk_lock.acquire()
        self.put_tasks.append(key)
        self.disk_lock.release()

        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj), self.loop)
        return future

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.evictor.update_on_hit(key, self.dict)

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        self.disk_lock.release()
        logger.info(f"Prefetching {key} from disk.")

        assert dtype is not None
        assert shape is not None
        future = asyncio.run_coroutine_threadsafe(
            self.async_load_bytes_from_disk(path, dtype, shape), self.loop)
        return future

    def get_blocking(
        self,
        key: CacheEngineKey, emerge_id
    ) -> Tuple[Union[MemoryObj, str], CacheEngineKey]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        found = False
        for old_key in self.dict.keys():
            if old_key == key:
                found = True
                break
        if not found:
            self.disk_lock.release()
            return None, key
        
        # Update key
        if key.metadata.context_id[0] not in old_key.metadata.context_id: 
            old_key.metadata.context_id.append(key.metadata.context_id[0])
            old_key.metadata.method.append(key.metadata.method[0])
            old_key.metadata.score_table.append(key.metadata.score_table[0])
            old_key.metadata.disk_score_table.append(key.metadata.disk_score_table[0])
        # Record request pattern
        if emerge_id:
            old_key.metadata.emerge_id.append(emerge_id)
            self.dict[old_key] = self.dict.pop(key)

        path = self.dict[old_key].path
        dtype = self.dict[old_key].dtype
        shape = self.dict[old_key].shape
        assert dtype is not None
        assert shape is not None
        memory_obj = self.load_bytes_from_disk(path, dtype=dtype, shape=shape)

        self.disk_lock.release()
        return memory_obj, old_key

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        path = self._key_to_path(key)
        if kv_chunk is not None:
            byte_array = memory_obj.byte_array

            async with aiofiles.open(path, 'wb') as f:
                await f.write(byte_array)
        else:
            async with aiofiles.open(path, 'wb') as f:
                await f.write(memory_obj.raw_data)

        self.insert_key(key, memory_obj)
        self.memory_allocator.ref_count_down(memory_obj)

        self.disk_lock.acquire()
        self.put_tasks.remove(key)
        self.disk_lock.release()

    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_BLOB2)
        if memory_obj is None:
            raise RuntimeError(
                "Failed to allocate memory for the async remote-retrieved KV cache.\n"
                "The KV cache will not be stored."
            )
        buffer = memory_obj.byte_array
        async with aiofiles.open(path, 'rb') as f:
            await f.readinto(buffer)
        return memory_obj

    # TODO(Jiayi): use memory allocator to redeuce cpu buffer allocation
    # TODO(Jiayi): the pinned cpu memory_obj should directly be passed into
    # gpu connector; this gpu buffer could be avoided
    def load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """
        if dtype == torch.int8:

            # file_size = os.path.getsize(path)
            # buffer = bytearray(file_size)
            # with open(path, 'rb') as f:
            #     f.readinto(buffer)
            # memory_obj = BytesBufferMemoryObj(buffer)
            # return memory_obj

            return path
        else:
            memory_obj = self.memory_allocator.allocate(shape, dtype, MemoryFormat.KV_BLOB2)
            if memory_obj is None:
                raise RuntimeError(
                    "Failed to allocate memory for the sync remote-retrieved KV cache.\n"
                    "The KV cache will not be stored."
                )
            
            ##### 3
            TARGET_RATE = 0.5 * 1024**3  # 0.5 GiB/s

            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            f  = os.fdopen(fd, 'rb', buffering=0)

            while True:
                t0 = time.perf_counter()
                n = f.readinto(memory_obj.byte_array)
                if n <= 0:
                    break
                elapsed = time.perf_counter() - t0
                expected = n / TARGET_RATE
                to_sleep = expected - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
            f.close()

            return memory_obj

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def load_disk(
        self,
        path: str,
        backend: str = "bytes",
        dtype: Optional[torch.dtype] = None,
        shape: Optional[torch.Size] = None,
    ) -> Optional[MemoryObj]:
        """
        Load KV from disk.
        """
        if backend == "bytes":
            assert dtype is not None
            assert shape is not None
            memory_obj = self.load_bytes_from_disk(path, dtype, shape)
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return memory_obj

    def close(self) -> None:
        pass
