import asyncio
import ctypes
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Optional

import aiofiles
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (BufferMemoryObj,
                                                    BufferMemoryObjMetadata,
                                                    MemoryFormat, MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata,
                           _lmcache_nvtx_annotate)

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
@torch.inference_mode()
async def async_save_bytes_to_disk(
    path: str,
    kv_chunk: torch.Tensor,
) -> None:
    """
    Convert KV to bytes and async store bytes to disk.
    """
    num_bytes = kv_chunk.numel() * kv_chunk.element_size()
    ptr = kv_chunk.data_ptr()
    ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    byte_array = (ctypes.c_ubyte * num_bytes).from_address(
        ctypes.addressof(ubyte_ptr.contents))
    async with aiofiles.open(path, 'wb') as f:
        await f.write(byte_array)


@_lmcache_nvtx_annotate
@torch.inference_mode()
def save_disk(
    path: str,
    kv_chunk: torch.Tensor,
    backend: str = "safetensors",
) -> None:
    """
    Save KV to disk.
    """
    if backend == "safetensors":
        save_file({"kv_chunk": kv_chunk}, path)
    elif backend == "torch":
        torch.save({"kv_chunk": kv_chunk}, path)
    else:
        raise ValueError(f"Invalid backend: {backend}")


async def async_load_bytes_from_disk(
    path: str,
    dst_device: str,
    dtype: torch.dtype,
    shape: torch.Size,
) -> BufferMemoryObj:
    """
    Async Load bytearray from disk.
    """
    async with aiofiles.open(path, 'rb') as f:
        bytes_data = await f.read()
    kv_chunk = torch.frombuffer(bytes_data,
                                dtype=dtype).reshape(shape).to(dst_device)
    return BufferMemoryObj(kv_chunk,
                           BufferMemoryObjMetadata(MemoryFormat.KV_BLOB))


def load_bytes_from_disk(
    path: str,
    dst_device: str,
    dtype: torch.dtype,
    shape: torch.Size,
) -> torch.Tensor:
    """
    Load bytearray from disk.
    """
    with open(path, 'rb') as f:
        bytes_data = f.read()
    kv_chunk = torch.frombuffer(bytes_data, dtype=dtype).view(shape)
    kv_chunk = kv_chunk.to(dst_device)
    return kv_chunk


@_lmcache_nvtx_annotate
@torch.inference_mode()
def load_disk(
    path: str,
    dst_device: str,
    backend: str = "bytes",
    dtype: Optional[torch.dtype] = None,
    shape: Optional[torch.Size] = None,
) -> BufferMemoryObj:
    """
    Load KV from disk.
    """
    if backend == "safetensors":
        with safe_open(path, framework="pt",
                       device=dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
    elif backend == "torch":
        data_dict = torch.load(path)
        kv_chunk = data_dict["kv_chunk"]
        kv_chunk = kv_chunk.to(dst_device)
    elif backend == "bytes":
        assert dtype is not None
        assert shape is not None
        kv_chunk = load_bytes_from_disk(path, dst_device, dtype, shape)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return BufferMemoryObj(kv_chunk,
                           BufferMemoryObjMetadata(MemoryFormat.KV_BLOB))


class LocalDiskBackend(StorageBackendInterface):

    def __init__(self, config: LMCacheEngineConfig, dst_device: str = "cuda"):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.dst_device = dst_device

        self.use_ipc = False

        if self.use_ipc:
            # TODO: Tune the number of workers for better performance
            self.proc_pool_executor = ProcessPoolExecutor(max_workers=4)
        else:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever)
            self.thread.start()

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        # TODO(Jiayi): Size and evictor should be configured

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.dict

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return self.path + key.to_string().replace("/", "-") + ".pt"

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        with self.disk_lock:
            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Future:
        assert memory_obj.tensor is not None
        path = self._key_to_path(key)
        if self.use_ipc:
            # TODO(Jiayi): Please get rid of this `clone()`
            # with shared memory. Directly passing a view of the tensor
            # will result in wrong result.
            assert memory_obj.tensor is not None
            kv_chunk = memory_obj.tensor.clone()
            path = self._key_to_path(key)
            future = self.proc_pool_executor.submit(save_disk, path, kv_chunk)
        else:
            kv_chunk = memory_obj.tensor
            future = asyncio.run_coroutine_threadsafe(
                async_save_bytes_to_disk(path, kv_chunk), self.loop)
        return future

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None
        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        self.disk_lock.release()
        logger.info(f"Prefetching {key} from disk.")
        if self.use_ipc:
            future = self.proc_pool_executor.submit(load_disk, path, "cpu")
        else:
            assert dtype is not None
            assert shape is not None
            future = asyncio.run_coroutine_threadsafe(
                async_load_bytes_from_disk(path, "cpu", dtype, shape),
                self.loop)
        return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[BufferMemoryObj]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None
        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        buffer_memory_obj = load_disk(path,
                                      self.dst_device,
                                      dtype=dtype,
                                      shape=shape)
        self.disk_lock.release()
        return buffer_memory_obj

    def close(self):
        if self.use_ipc:
            self.proc_pool_executor.shutdown()
        else:
            # using threadsafe method here as stop modifies
            # the internal state of the loop (in another thread)
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread.is_alive():
                self.thread.join()
        logger.info("Local disk backend closed.")

    def __del__(self):
        self.close()
