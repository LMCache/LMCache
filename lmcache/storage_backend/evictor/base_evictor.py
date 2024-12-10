import abc
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Union

import torch

from lmcache.logging import init_logger
from lmcache.storage_backend.mem_pool import KVObj
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _get_size

logger = init_logger(__name__)


class PutStatus(Enum):
    LEGAL = 1
    ILLEGAL = 2


class BaseEvictor(metaclass=abc.ABCMeta):
    """
    Interface for cache evictor
    """

    @abc.abstractmethod
    def update_on_get(self, key: Union[CacheEngineKey, str],
                      cache_dict: OrderedDict) -> None:
        """
        Update cache_dict when a cache is used is used

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_on_put(
            self, cache_dict: OrderedDict, cache_size: int
    ) -> Tuple[List[Union[CacheEngineKey, str]], PutStatus]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected
        
        Return:
            return a key to be evicted
        """
        raise NotImplementedError

    # TODO (Jiayi): KV object should have a better abstraction
    # e.g., a kv_obj class wize size field
    def get_size(
            self, kv_obj: Union[torch.Tensor, bytes, DiskCacheMetadata,
                                KVObj]) -> int:
        """
        Get the size of the kv cache
        
        Input:
            kv_obj: kv cache

        Return:
            the size of the cache (in bytes)
        """
        return _get_size(kv_obj)


class DummyEvictor(BaseEvictor):

    def update_on_get(self, key: Union[CacheEngineKey, str],
                      cache_dict: OrderedDict) -> None:
        # Dummy implementation does nothing
        pass

    def update_on_put(self, cache_dict: OrderedDict, cache_size: int):
        # Dummy implementation does not evict anything
        return [], PutStatus.LEGAL
