# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
from typing import Generic, TypeVar
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

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        """
        Update internal states when a cache is removed from cache_dict

        Backends must call this for every removal, whether it is a forced
        remove from the external or an internal eviction of a key returned
        by ``get_evict_candidates``. Calling it for a key the policy does
        not track is a no-op.

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
        Select the keys to evict when a new cache comes and the storage is
        full

        This only selects: it does not modify cache_dict or internal states,
        and the caller may evict fewer keys than returned. The caller must
        call ``update_on_force_evict`` for each key it actually removes.

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: maximum number of candidates to return

        Return:
            return a list of at most num_candidates evictable keys, in
            eviction order
        """
        raise NotImplementedError
