from typing import Tuple, Optional, Iterator
from safetensors import safe_open
from safetensors.torch import save_file
import re
import io
import torch
import redis
import os

from lmcache.utils import CacheEngineKey, KVCache
from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

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
        if not blocking:
            logger.warning("Non-blocking is not implemented for local backend")
        self.dict[key] = kv_chunk


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
        return self.dict.get(key, None)


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
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        self.cache_metadata[key] = {'device': str(kv_chunk.device)}
        logger.info(f"Saving cache to {self._key_to_path(key)}")
        #torch.save(kv_chunk, self._key_to_path(key))
        save_file({'kv_chunk': kv_chunk.contiguous()}, self._key_to_path(key))


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