import abc
from lmcache.types import KVCache
from lmcache.config import LMCacheEngineConfig
from typing import Tuple, Optional

class LMCBackendInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def put(
            self,
            key: Tuple[str, str],
            kv_chunk: KVCache,
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
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
            self,
            key: Tuple[str, str],
        ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        raise NotImplementedError
