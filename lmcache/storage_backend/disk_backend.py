import os
import queue
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.config import LMCacheEngineConfig
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCKeyManagerInterface,LMCKeyManagerValue,LMCBackendInterface
from lmcache.storage_backend.evictor import DummyEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata, KVCache,LMCKeyManagerKey,LMCKeyManagerValue,
                           _lmcache_nvtx_annotate)
import xmlrpc.client

logger = init_logger(__name__)


class LocalBackendEndSignal:
    pass



# TODO(Jiayi): need to optimize disk saving/loading
# current impl. with "safetensors" might not be efficient
# but it is better than "torch.save/load"

# TODO(Jiayi): need to support prefetch for disk


class LMCDiskBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """
    def __init__(self, config: LMCacheEngineConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__()
        
        assert config.disk_url is not None, ("Need to specify local path if when "
                                       "using LMCLocalDiskBackend")
        self.proxy = xmlrpc.client.ServerProxy(config.disk_url)
        Info = self.proxy.Info()

        self.remote_fmt = Info["fmt"]
        self.remote_dtype = Info["dtype"]
        self.remote_chunk_size = Info["chunk_size"]
        self.remote_serde = Info["serde"]

        self.chunk_size = config.chunk_size

        # TODO(Jiayi): the following async put code is repeated in all backends
        # Please consider use a parent class that can be inherited by all
        # (local) backends
        # This should be also be helpful for more flexible hierarchical backends
        # For async put
        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # TODO (Jiayi): please remove this hard code
        self.dst_device = "cuda"

        self.evictor = DummyEvictor()

    def _key_transform(self, key: CacheEngineKey) -> LMCKeyManagerKey:
        return LMCKeyManagerKey(key.model_name, key.world_size, key.worker_id, key.chunk_hash).to_string()
    
    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return self.proxy.contains(self._key_transform(key))=="YES"

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        put_stream = torch.cuda.Stream()
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            with torch.cuda.stream(put_stream):
                self.put_blocking(key, value)

    def put_blocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
    ) -> None:
        self.update_lock.acquire()
        print(self._key_transform(key), self.evictor.get_size(kv_chunk),0)
        path=self.proxy.put(self._key_transform(key), self.evictor.get_size(kv_chunk),0)

        print(path)

        if path == "":
            return
        logger.debug(f"Saving cache to {path}")

        save_file({"kv_chunk": kv_chunk}, path)
        self.update_lock.release()
        self.proxy.put(self._key_transform(key), 0,1)

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            self.put_queue.put((key, kv_chunk))

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[KVCache]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        print(self._key_transform(key))
        value=self.proxy.get(self._key_transform(key))
        print(value)
        # breakpoint()
        while value['status'] == 1:
            time.sleep(0.1)
            value=self.proxy.get(self._key_transform(key))
        
        if value['status'] == 0:
            return None

        self.update_lock.acquire()
        with safe_open(value['path'], framework="pt",
                       device=self.dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
        self.update_lock.release()
        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local disk backend")

    def __del__(self):
        self.close()

