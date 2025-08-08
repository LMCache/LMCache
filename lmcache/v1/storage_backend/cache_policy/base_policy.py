# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
import abc

# First Party
from lmcache.utils import CacheEngineKey



class BaseCachePolicy(metaclass=abc.ABCMeta):
    """
    Interface for cache policy.
    """

    @abc.abstractmethod
    def init_mutable_mapping(self) -> MutableMapping[CacheEngineKey, Any]:
        """
        Initialize a mutable mapping for cache storage.

        Return:
            A mutable mapping that can be used to store cache entries.
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_hit(
        self, 
        key: CacheEngineKey, 
        cache_dict: MutableMapping[CacheEngineKey, Any],
    ) -> None:
        """
        Update cache_dict when a cache is used

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def get_evict_candidates(
        self, 
        cache_dict: MutableMapping[CacheEngineKey, Any], 
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: number of candidates to be evicted

        Return:
            return a list of CacheEngineKeys
        """
        raise NotImplementedError
