import abc
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class PutStatus(Enum):
    LEGAL = 1
    ILLEGAL = 2


class BaseEvictor(metaclass=abc.ABCMeta):
    """
    Interface for cache evictor
    """

    @abc.abstractmethod
    def update_on_hit(self, key: CacheEngineKey,
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
            self, cache_dict: OrderedDict,
            cache_size: int) -> Tuple[List[CacheEngineKey], PutStatus]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected
        
        Return:
            return a list of keys to be evicted and a PutStatus
            to indicate whether the put is allowed
        """
        raise NotImplementedError
