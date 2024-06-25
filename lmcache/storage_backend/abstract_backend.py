import abc
import torch
from lmcache.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey
from lmcache.logging import init_logger
from typing import Tuple, Optional, Iterator

logger = init_logger(__name__)

class LMCBackendInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def put(
            self,
            key: CacheEngineKey,
            kv_chunk: torch.Tensor,
            blocking = True,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, in the format of CacheEngineKey
            kv_chunk: the kv cache of the token chunk, in the format of a big tensor
            blocking: whether to block the call before the operation is completed

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
            self,
            key: CacheEngineKey,
        ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
            self,
            key: CacheEngineKey,
        ) -> Optional[torch.Tensor]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output: 
            the kv cache of the token chunk, in the format of a big tensor
            None if the key is not found
        """
        raise NotImplementedError
    

    def batched_put(
            self,
            keys_and_chunks: Iterator[Tuple[CacheEngineKey, torch.Tensor]],
            blocking = True,
        ) -> int:
        """
        Store the multiple keys and KV cache chunks into the cache engine in a batched manner.

        Input:
            keys: the iterator of keys of the token chunks, in the format of CacheEngineKey
            kv_chunks: the iterator of kv cache of the token chunks, in the format of a big tensor
            blocking: whether to block the call before the operation is completed

        Returns:
            the number of chunks are stored
        """
        logger.info("Using default batched implementation of the put() method")
        nchunks = 0
        for key, kv_chunk in keys_and_chunks:
            self.put(key, kv_chunk, blocking = blocking)
            nchunks += 1
        return nchunks
    
    def batched_get(
            self,
            keys: Iterator[CacheEngineKey],
        ) -> Iterator[Optional[torch.Tensor]]:
        """
        Retrive the kv cache chunks by the given keys in a batched manner

        Input:
            keys: the iterator of keys of the token chunks, including prefix hash and format

        Output:
            the iterator of kv cache of the token chunks, in the format of a big tensor
            None if the key is not found
        """
        logger.info("Using default batched implementation of the get() method")
        for key in keys:
            if self.contains(key): # Jiayi: This seems to be redundant?
                yield self.get(key)
            else:
                yield None

    def close(self):
        """
        Do the cleanup things
        Children classes should override this method if necessary
        """
        pass
