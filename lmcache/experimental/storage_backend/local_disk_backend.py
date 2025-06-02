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


class LocalDiskBackend(StorageBackendInterface):

    def __init__(self,
                 config: LMCacheEngineConfig,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface,
                 dst_device: str = "cuda"):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.dst_device = dst_device

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        # Initialize the evictor
        self.evictor = LRUEvictor(max_cache_size=config.max_local_disk_size)

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
        # assert memory_obj.tensor is not None

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

        #kv_chunk = memory_obj.tensor
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
            self.dict[old_key] = self.dict.pop(key)
        # Record request pattern
            old_key.metadata.emerge_id.append(emerge_id)

        # # Update cache recency
        # self.evictor.update_on_hit(key, self.dict)

        path = self.dict[old_key].path
        dtype = self.dict[old_key].dtype
        shape = self.dict[old_key].shape
        assert dtype is not None
        assert shape is not None
        memory_obj = self.load_bytes_from_disk(path, dtype=dtype, shape=shape)

        # # For baseline, moved to CPU
        # if self.policy == "baseline_KIVI":
        #     self.remove(old_key)
        #     self.evictor.current_cache_size -= memory_obj.get_physical_size()

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
            logger.debug("Memory allocation failed during async disk load.")
            return None
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
                logger.debug("Memory allocation failed during async disk load.")
                return None
            
            ##### 1
            # buffer = memory_obj.byte_array
            # start = time.perf_counter()
            # with open(path, 'rb') as f:
            #     f.readinto(buffer)
            # duration = time.perf_counter() - start
            # logger.info(
            #     "Async disk load: read %d bytes into buffer in %.3f seconds",
            #     len(buffer), duration
            # )

            ##### 2
            # # Linux 下绕过页缓存打开标志
            # O_DIRECT = os.O_DIRECT
            # # 根据你的需求也可以加 O_SYNC、O_DSYNC
            # flags = os.O_RDONLY | O_DIRECT

            # # fd 打开时就带上 O_DIRECT
            # fd = os.open(path, flags)

            # # 用无缓冲的 FileIO
            # f = os.fdopen(fd, 'rb', buffering=0)

            # # 注意：O_DIRECT 要求 buffer 和读长度都与磁盘块大小（通常 4096 字节）对齐
            # # memory_obj.byte_array 必须满足对齐，否则会报 Invalid argument。
            # start = time.perf_counter()
            # f.readinto(memory_obj.byte_array)
            # duration = time.perf_counter() - start

            # logger.info(
            #     "Direct I/O: read %d bytes from %s in %.3f s (≈ %.2f GiB/s)",
            #     len(memory_obj.byte_array), path, duration,
            #     len(memory_obj.byte_array) / duration / (1024**3)
            # )

            # f.close()

            ##### 3
            TARGET_RATE = 1000 * 1024**2  # 1 GiB/s

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
