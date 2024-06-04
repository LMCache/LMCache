import abc
import torch
from lmcache.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey
from typing import Tuple, Optional

class LMCBackendInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def put(
            self,
            key: CacheEngineKey,
            kv_chunk: torch.Tensor,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, in the format of CacheEngineKey
            kv_chunk: the kv cache of the token chunk, in the format of a big tensor

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
