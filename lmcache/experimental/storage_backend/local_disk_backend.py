import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Optional

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
def save_disk(
    path: str,
    kv_chunk: torch.Tensor,
    fmt_str: str,
    backend: str = "safetensors",
) -> None:
    """
    Save KV to disk.
    """
    if backend == "safetensors":
        save_file({"kv_chunk": kv_chunk}, path, {"fmt": fmt_str})
    elif backend == "torch":
        torch.save({"kv_chunk": kv_chunk, "fmt": fmt_str}, path)


@_lmcache_nvtx_annotate
@torch.inference_mode()
def load_disk(
    path: str,
    dst_device: str,
    backend: str = "safetensors",
) -> BufferMemoryObj:
    """
    Load KV from disk.
    """
    if backend == "safetensors":
        with safe_open(path, framework="pt",
                       device=dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
            metadata = f.metadata()
            fmt_str = metadata["fmt"]
    elif backend == "torch":
        data_dict = torch.load(path)
        kv_chunk = data_dict["kv_chunk"]
        fmt_str = data_dict["fmt"]
    return BufferMemoryObj(kv_chunk,
                           BufferMemoryObjMetadata(MemoryFormat(int(fmt_str))))


class LocalDiskBackend(StorageBackendInterface):

    def __init__(self, config: LMCacheEngineConfig, dst_device: str = "cuda"):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.dst_device = dst_device

        # TODO: Tune the number of workers for better performance
        self.proc_pool_executor = ProcessPoolExecutor(max_workers=4)

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        # TODO(Jiayi): Size and evictor should be configured

        self.is_shutdown = False

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

    def insert_key(self, key: CacheEngineKey, size: int):
        path = self._key_to_path(key)
        with self.disk_lock:
            self.dict[key] = DiskCacheMetadata(path, size)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Future:
        # TODO(Jiayi): Please get rid of this `clone()`
        # with shared memory. Directly passing a view of the tensor
        # will result in wrong result.
        assert memory_obj.tensor is not None
        kv_chunk = memory_obj.tensor.clone()
        fmt_str = str(memory_obj.metadata.fmt.value)
        path = self._key_to_path(key)
        future = self.proc_pool_executor.submit(save_disk, path, kv_chunk,
                                                fmt_str)
        return future

    def put_blocking(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Blocking put function.
        This is for debugging and testing purposes.
        """
        path = self._key_to_path(key)
        kv_chunk = memory_obj.tensor
        fmt_str = str(memory_obj.metadata.fmt.value)
        save_disk(path, kv_chunk, fmt_str)
        self.insert_key(key, memory_obj.get_size())

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:

        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None
        path = self.dict[key].path
        self.disk_lock.release()
        logger.info(f"Prefetching {key} from disk.")
        future = self.proc_pool_executor.submit(load_disk, path, "cpu")
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
        buffer_memory_obj = load_disk(path, self.dst_device)
        self.disk_lock.release()
        return buffer_memory_obj

    def close(self):
        # TODO: Might be contention here
        if not self.is_shutdown:
            self.proc_pool_executor.shutdown()
        logger.info("Local disk backend closed.")

    def __del__(self):
        self.close()
