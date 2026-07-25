# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar
import abc

KeyType = TypeVar("KeyType")
MapType = TypeVar("MapType", bound=MutableMapping)


class BaseCachePolicy(Generic[KeyType, MapType], metaclass=abc.ABCMeta):
    """
    Interface for cache policy.
    """

    @abc.abstractmethod
    def init_mutable_mapping(self) -> MapType:
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
        key: KeyType,
        cache_dict: MapType,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is used

        Input:
            key: an object of KeyType
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is stored

        Input:
            key: an object of KeyType
        """
        raise NotImplementedError

    def update_on_put_with_metadata(
        self,
        key: KeyType,
        cache_obj: Any = None,
        **metadata: Any,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is stored with optional metadata.
        Default implementation falls back to update_on_put(key).

        Input:
            key: an object of KeyType
            cache_obj: optional cache object (e.g. MemoryObj)
            metadata: additional metadata key-value pairs
        """
        self.update_on_put(key)

    def update_cost_observation(
        self,
        key: KeyType,
        **metadata: Any,
    ) -> None:
        """
        Record a cost observation for key without changing recency or ordering.
        Default implementation is a no-op.

        Input:
            key: an object of KeyType
            metadata: additional cost observation metadata
        """
        pass

    def should_admit(
        self,
        key: KeyType,
        cache_dict: MapType,
    ) -> bool:
        """
        Decide whether a new key should be admitted into the cache, as an
        alternative to always evicting an existing entry to make room.
        Default implementation always admits (existing behavior for every
        policy that doesn't override this).

        Precondition: callers must only invoke this when the cache is
        already at or over capacity, mirroring the existing convention
        that `get_evict_candidates` is only called under capacity
        pressure. Calling it on a cache with free space can cause a
        policy that overrides this (e.g. an admission-controlled policy)
        to wrongly reject an admission that didn't need any eviction at
        all -- this method has no visibility into capacity itself, only
        into `cache_dict`'s current contents.

        Input:
            key: an object of KeyType for the candidate new entry
            cache_dict: a dict consists of current cache

        Return:
            True if the key should be admitted (evicting a candidate if
            necessary), False if the admission should be skipped entirely.
        """
        return True

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        """
        Update internal states when a cache is force evicted

        Input:
            key: an object of KeyType
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def get_evict_candidates(
        self,
        cache_dict: MapType,
        num_candidates: int = 1,
    ) -> list[KeyType]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: number of candidates to be evicted

        Return:
            return a list of keys to be evicted
        """
        raise NotImplementedError
