from typing import Tuple, Optional, Iterator, List
from safetensors import safe_open
from safetensors.torch import save_file
import re
import io
import torch
import redis
import os
import pickle

from lmcache.server.server_storage_backend.abstract_backend import LMSBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LMSLocalBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu memory.
    """
    def __init__(
            self, 
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        self.dict = {}
    
    def list_keys(
            self
        ) -> List[str]:
        
        return list(self.dict.keys())
    
    def contains(
            self, 
            key: str,
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
            key: str,
            kv_chunk_bytes: bytes,
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
        self.dict[key] = kv_chunk_bytes


    @_lmcache_nvtx_annotate
    def get(
            self,
            key: str,
        ) -> Optional[bytes]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        return self.dict.get(key, None)


# TODO(Jiayi): need to optimize disk loading
# current impl. with "naive open read/write" might not be efficient (better than torch.load)
class LMSLocalDiskBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """
    def __init__(
            self, 
            path: str,
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.filenames = set()

    def list_keys(
            self
        ) -> List[str]:
        
        return list(self.filenames)
    
    def contains(
            self, 
            key: str,
        ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.filenames

    def _key_to_path(
        self,
        key: str,
    ) -> str:
        """
        Covert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.replace("/","-") + ".bin"
        
    
    def put(
            self, 
            key: str,
            kv_chunk_bytes: bytes,
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
        self.filenames.add(key)
        logger.info(f"Saving cache to {self._key_to_path(key)}")
        #torch.save(kv_chunk_bytes, self._key_to_path(key))
        with open(self._key_to_path(key), "wb") as binary_file:
            binary_file.write(kv_chunk_bytes)


    @_lmcache_nvtx_annotate
    def get(
            self,
            key: str,
        ) -> Optional[bytes]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        if key not in self.filenames:
            return None
        
        with open(self._key_to_path(key), "rb") as binary_file:
            return binary_file.read()
        
        #return torch.load(self._key_to_path(key))
