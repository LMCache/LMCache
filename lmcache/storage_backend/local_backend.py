from typing import Tuple, Optional, Iterator
from safetensors import safe_open
from safetensors.torch import save_file
import re
import io
import torch
import redis
import os
import threading
import queue
import time

from lmcache.utils import CacheEngineKey, KVCache
from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LocalBackendEndSignal:
    pass

class LMCLocalBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu memory.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        self.chunk_size = config.chunk_size 
        self.config = config
        self.dict = {}
        self.device = config.local_device
        
        self.put_queue = queue.Queue()
        self.put_thread = threading.Thread(
                target=self.put_worker, args=()
            ) 
        self.put_thread.start()
        self.update_lock = threading.Lock()
        self.use_pin_memory = True
        logger.info(f"Using pinned cpu memory: {self.use_pin_memory}")
        #self.async_put_flag = False
        #self.put_events = {}
        

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
        return key in self.dict
    
    @_lmcache_nvtx_annotate
    def put_worker(
            self,
    ):
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            #with torch.cuda.stream(self.put_stream):
            self.put_nonblocking(key, value)

    def put_nonblocking(
        self,
        key,
        kv_chunk
    ):
        # TODO(Jiayi): torch.cuda.synchronize() needs to be removed
        # to enable actual async put
        # torch.cuda.synchronize() may disturb inference engine in inference engine
        if self.use_pin_memory:
            kv_chunk_local = kv_chunk.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            kv_chunk_local = kv_chunk.to(self.device)
        self.update_lock.acquire()
        self.dict[key] = kv_chunk_local
        self.update_lock.release()
    
    def put_blocking(
        self,
        key,
        kv_chunk
    ):
        if self.use_pin_memory:
            self.dict[key] = kv_chunk.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            self.dict[key] = kv_chunk.to(self.device)
    
    def put(
            self, 
            key: CacheEngineKey,
            kv_chunk: KVCache,
            blocking: bool = True,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            #self.put_blocking(key, kv_chunk)
            self.put_queue.put((key, kv_chunk))


    @_lmcache_nvtx_annotate
    def get(
            self,
            key: CacheEngineKey,
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        #if self.async_put_flag:
        #    while (not self.put_events[key].is_set()):
        #        time.sleep(0.001)
        return self.dict.get(key, None)

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local disk backend")
    
    def __del__(self):
        self.close()

# TODO(Jiayi): need to optimize disk saving/loading
# current impl. with "safetensors" might not be efficient
# but it is better than "torch.save/load"

#TODO(Jiayi): need to support prefetch for disk

class LMCLocalDiskBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        self.chunk_size = config.chunk_size 
        self.config = config
        self.path = config.local_device
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.cache_metadata = {}
        
        # TODO(Jiayi): the following async put code is repeated in all backends
        # Please consider use a parent class that can be inherited by all (local) backends
        # This should be also be helpful for more flexible heirarchical backends
        # For async put
        self.put_queue = queue.Queue()
        self.put_thread = threading.Thread(
                target=self.put_worker, args=()
            ) 
        self.put_thread.start()
        self.update_lock = threading.Lock()


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
        return key in self.cache_metadata.keys()

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        """
        Covert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.to_string().replace("/","-") + ".pt"
    
    @_lmcache_nvtx_annotate
    def put_worker(
            self,
    ):
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
        kv_chunk: KVCache,
    ) -> None:
        logger.info(f"Saving cache to {self._key_to_path(key)}")
        # The following order matters of `save_file` and `update dictionary` matters
        save_file({'kv_chunk': kv_chunk}, self._key_to_path(key))
        self.update_lock.acquire()
        self.cache_metadata[key] = {'device': str(kv_chunk.device)}
        self.update_lock.release()
        
    
    def put(
            self, 
            key: CacheEngineKey,
            kv_chunk: KVCache,
            blocking: bool = True,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

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
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        if key not in self.cache_metadata.keys():
            return None
        
        with safe_open(
            self._key_to_path(key), 
            framework="pt", 
            device=self.cache_metadata[key]['device']) as f:
            return f.get_tensor('kv_chunk')    
        #return torch.load(self._key_to_path(key))
    
    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local disk backend")
    
    def __del__(self):
        self.close()