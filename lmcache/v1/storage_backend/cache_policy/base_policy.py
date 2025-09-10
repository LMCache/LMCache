# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar
import abc

# First Party
from lmcache.utils import CacheEngineKey

TCache = TypeVar("TCache", bound=MutableMapping[CacheEngineKey, Any])


class BaseCachePolicy(Generic[TCache], metaclass=abc.ABCMeta):
    """
    Interface for cache policy.
    """

    @abc.abstractmethod
    def init_mutable_mapping(self) -> TCache:
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
        cache_dict: TCache,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is used

        Input:
            key: a CacheEngineKey
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_put(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is stored

        Input:
            key: a CacheEngineKey
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Update internal states when a cache is force evicted

        Input:
            key: a CacheEngineKey
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def get_evict_candidates(
        self,
        cache_dict: TCache,
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
