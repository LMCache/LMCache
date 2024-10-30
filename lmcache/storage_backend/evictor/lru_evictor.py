from collections import OrderedDict
from typing import Union

import torch

from lmcache.logging import init_logger
from lmcache.storage_backend.evictor.base_evictor import BaseEvictor, PutStatus
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class LRUEvictor(BaseEvictor):
    """
    LRU cache evictor
    """

    def __init__(self, max_cache_size: float = 10.0):
        # TODO(Jiayi): need to be configured
        # the storage size limit (in GB)
        self.MAX_CACHE_SIZE = max_cache_size

        # TODO(Jiayi): need a way to avoid fragmentation
        # current storage size (in GB)
        self.current_cache_size = 0.0

    def update_on_get(self, key: Union[CacheEngineKey, str],
                      cache_dict: OrderedDict) -> None:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            key: a CacheEngineKey or a str
            cache_dict: a dict consists of current cache
        """
        cache_dict.move_to_end(key)

    # FIXME(Jiayi): comment out return type to bypass type checks
    # Need to align CacheEngineKey & str
    def update_on_put(self, cache_dict: OrderedDict, kv_obj: Union[
        torch.Tensor,
        bytes]):  #-> Tuple[List[Union[CacheEngineKey, str]], PutStatus]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected
        
        Return:
            evict_keys: a list of keys to be evicted
        """
        evict_keys = []
        cache_size = self.get_size(kv_obj)
        iter_cache_dict = iter(cache_dict)

        if cache_size > self.MAX_CACHE_SIZE:
            logger.info("Put failed due to limited cache storage")
            return [], PutStatus.ILLEGAL

        # evict cache until there's enough space
        while cache_size + self.current_cache_size > \
            self.MAX_CACHE_SIZE:
            evict_key = next(iter_cache_dict)
            evict_cache_size = self.get_size(cache_dict[evict_key])
            self.current_cache_size -= evict_cache_size
            evict_keys.append(evict_key)

        # update cache size
        self.current_cache_size += cache_size
        logger.debug(f"Evicting {len(evict_keys)} chunks")
        return evict_keys, PutStatus.LEGAL
