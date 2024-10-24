import abc
import torch
from typing import OrderedDict, Optional, Union
from collections import OrderedDict
from enum import Enum

from lmcache.utils import CacheEngineKey
from lmcache.logging import init_logger

logger = init_logger(__name__)

class PutStatus(Enum):
    LEGAL = 1
    ILLEGAL = 2

class BaseEvictor(metaclass=abc.ABCMeta):
    """
    Interface for cache evictor
    """
    
    @abc.abstractmethod
    def update_on_get(self, key: Union[CacheEngineKey, str], cache_dict: OrderedDict) -> None:
        """
        Update cache_dict when a cache is used is used

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_on_put(
        self, 
        cache_dict: OrderedDict, 
        kv_obj: Union[torch.Tensor, bytes]) -> Tuple[List[Union[CacheEngineKey, str]], PutStatus]:
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
    def get_size(self, kv_obj: Union[torch.Tensor, bytes]) -> float:
        """
        Get the size of the kv cache
        
        Input:
            kv_obj: kv cache

        Return:
            the size of the cache (in GB)
        """
        
        # Get size of one element in bytes
        if isinstance(kv_obj, torch.Tensor):
            num_elements = kv_obj.numel()
            element_size = kv_obj.element_size()
            size_in_bytes = num_elements * element_size
        elif isinstance(kv_obj, bytes):
            size_in_bytes = len(kv_obj)
        else:
            raise Exception("Encountered unknown kv data type!")
        
        # Convert to gigabytes (GB)
        size_in_gb = size_in_bytes / (1024 ** 3)
        return size_in_gb