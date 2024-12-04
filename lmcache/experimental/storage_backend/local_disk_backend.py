import abc
import threading
from typing import Callable
from concurrent.futures import Future
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor

from lmcache.experimental.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey, DiskCacheMetadata


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
    dst_device:str,
) -> torch.Tensor:
    with safe_open(path, framework="pt",
                   device=dst_device) as f:
        kv_chunk = f.get_tensor("kv_chunk")
    return kv_chunk

class LocalDiskBackend(StorageWorkerInterface):
    def __init__(self):
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        
        self.put_executor = ProcessPoolExecutor(max_workers=4)
        self.prefetch_executor = ProcessPoolExecutor(max_workers=1)
        
        #
        self.dst_device = "cuda"
        self.disk_lock = threading.Lock()
    
    def contains(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.dict
    
    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return self.path + key.to_string().replace("/", "-") + ".pt"
    
    def insert_key(
        self, 
        key: CacheEngineKey,
        size: int):
        path = self._key_to_path(path)
        self.disk_lock:
            self.dict[key] = DiskCacheMetadata(path, size)
    
    
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        """An async function to put the MemoryObj into the storage backend.
        It should free the memory object after finish putting the object.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored.
        """
        
        path = self._key_to_path(path)
        future = self.put_executor.submit(save_disk, path,
                                                memory_obj.tensor)
        return future

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Future:
        """An async function to get the MemoryObj from the storage backend.
        Will call the callback with the MemoryObj when finished.

        :param CacheEngineKey key: The key of the MemoryObj.
        """
        if key not in self.dict:
            return None
        
        path = self.dict[key].path
        future = self.prefetch_executor.submit(load_disk, path, "cpu")
        return future
    
    def get_blocking(
        self,
        key: CacheEngineKey,
    ):
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None
        path = self.dict[key].path
        kv_chunk = load_disk(path, self.dst_device)
        self.disk_lock.release()
        return kv_chunk
        
