# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Union
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.storage_backend.mem_pool import KVObj
from lmcache.utils import CacheEngineKey, DiskCacheMetadata

logger = init_logger(__name__)


class PutStatus(Enum):
    LEGAL = 1
    ILLEGAL = 2


class BaseEvictor(metaclass=abc.ABCMeta):
    """
    Interface for cache evictor
    """

    @abc.abstractmethod
    def update_on_get(
        self, key: Union[CacheEngineKey, str], cache_dict: OrderedDict
    ) -> None:
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
            cache_size: the size of the cache to be injected

        Return:
            evict_keys: a list of keys to be evicted
            status:
                PutStatus.LEGAL if the cache is legal,
                PutStatus.ILLEGAL if the cache is illegal
        """
        raise NotImplementedError

    # TODO (Jiayi): KV object should have a better abstraction
    # e.g., a kv_obj class wize size field
    def get_size(self, kv_obj: Union[torch.Tensor, bytes, KVObj]) -> int:
        """
        Get the size of the kv cache

        Input:
            kv_obj: kv cache

        Return:
            the size of the cache (in bytes)
        """

        if isinstance(kv_obj, torch.Tensor):
            num_elements = kv_obj.numel()
            element_size = kv_obj.element_size()
            size_in_bytes = num_elements * element_size
        elif isinstance(kv_obj, bytearray):
            size_in_bytes = len(kv_obj)
        elif isinstance(kv_obj, KVObj):
            size_in_bytes = kv_obj.size
        elif isinstance(kv_obj, DiskCacheMetadata):
            size_in_bytes = kv_obj.size
        else:
            raise Exception(f"Encountered unknown kv data type {type(kv_obj)}!")

        return size_in_bytes


class DummyEvictor(BaseEvictor):
    def update_on_get(
        self, key: Union[CacheEngineKey, str], cache_dict: OrderedDict
    ) -> None:
        # Dummy implementation does nothing
        pass

    def update_on_put(self, cache_dict: OrderedDict, cache_size: int):
        # Dummy implementation does not evict anything
        return [], PutStatus.LEGAL
