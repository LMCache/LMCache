import torch
from typing import OrderedDict, Optional, Union
from collections import OrderedDict

from lmcache.utils import CacheEngineKey
from lmcache.storage_backend.evictor.base_evictor import BaseEvictor

class LRUEvictor(BaseEvictor):
    """
    LRU cache evictor
    """
    
    def __init__(self, max_cache_size: float = 10):
        # the storage size limit (in GB)
        self.MAX_CACHE_SIZE = max_cache_size
        
        # TODO (Jiayi): need a way to avoid fragmentation
        # current storage size (in GB)
        self.current_cache_size = 0
            
    
    def update(self, key: CacheEngineKey, cache_dict: OrderedDict) -> None:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        cache_dict.move_to_end(key)
    
    def get_evict_key(
        self, 
        cache_dict: OrderedDict, 
        kv_obj: Union[torch.Tensor, bytes]) -> Optional[str]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            kv_obj: the new kv cache to be injected
        
        Return:
            evict_key: a key to be evicted
        """
        evict_key = None
        cache_size = self.get_size(kv_obj)
        if cache_size + self.current_cache_size > \
            self.MAX_CACHE_SIZE:
            evict_key = next(iter(cache_dict))
        return evict_key
        
            
        
        
    
    
        
    