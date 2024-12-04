import threading
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import MemoryObj
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata,
                           _lmcache_nvtx_annotate)


@_lmcache_nvtx_annotate
@torch.inference_mode()
def save_disk(
    path: str,
    kv_chunk: torch.Tensor,
) -> None:
    save_file({"kv_chunk": kv_chunk.contiguous()}, path)


@_lmcache_nvtx_annotate
@torch.inference_mode()
def load_disk(
    path: str,
    dst_device: str,
) -> torch.Tensor:
    with safe_open(path, framework="pt",
                   device=dst_device) as f:  # type: ignore
        kv_chunk = f.get_tensor("kv_chunk")
    return kv_chunk


class LocalDiskBackend(StorageBackendInterface):

    def __init__(self, config: LMCacheEngineConfig, dst_device: str = "cuda"):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.device = dst_device
        self.put_executor = ProcessPoolExecutor(max_workers=4)
        self.prefetch_executor = ProcessPoolExecutor(max_workers=1)

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk

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

    def insert_key(self, key: CacheEngineKey, size: int):
        path = self._key_to_path(key)
        with self.disk_lock:
            self.dict[key] = DiskCacheMetadata(path, size)

    def submit_put_task(self, key: CacheEngineKey,
                        memory_obj: MemoryObj) -> Future:

        path = self._key_to_path(key)
        future = self.put_executor.submit(save_disk, path, memory_obj.tensor)
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
        self.disk_lock.release()

        future = self.prefetch_executor.submit(load_disk, path, "cpu")
        return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None
        path = self.dict[key].path
        kv_chunk = load_disk(path, self.dst_device)
        self.disk_lock.release()
        return kv_chunk
