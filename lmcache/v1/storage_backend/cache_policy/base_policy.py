# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
from typing import Generic, Optional, TypeVar
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

    def disk_access_get(self, key: KeyType) -> int:
        """
        Return the access count used for optional SSD write gating.

        Backends that gate disk writes on frequency should update this count
        on cache hits (under their storage lock). Default: 0.

        Args:
            key: Cache key (same type as used in the backend dict).

        Returns:
            Current access count for ``key``.
        """
        return 0

    def disk_access_increment(self, key: KeyType) -> None:
        """
        Increment the access count for SSD frequency gating.

        Default: no-op (policies that do not implement gating).

        Args:
            key: Cache key.
        """
        pass

    def disk_access_reset(self, key: KeyType) -> None:
        """
        Reset the access count after a chunk is admitted to disk storage.

        Default: no-op.

        Args:
            key: Cache key.
        """
        pass

    def disk_access_pop(self, key: KeyType) -> None:
        """
        Drop access-count state when a chunk is removed from disk.

        Default: no-op.

        Args:
            key: Cache key.
        """
        pass

    def disk_gate_block_reason(
        self,
        key: KeyType,
        size_bytes: int,
        access_count: int,
    ) -> Optional[str]:
        """
        If an SSD write should be skipped, return why; otherwise ``None``.

        Implementations may compare ``size_bytes`` and ``access_count`` to
        configured thresholds. Return values are ``\"length\"`` or
        ``\"frequency\"`` for observability; backends map these to metrics.

        Args:
            key: Cache key.
            size_bytes: Serialized chunk size in bytes.
            access_count: Value from :meth:`disk_access_get` (taken under the
                same lock as dict access).

        Returns:
            ``\"length\"`` or ``\"frequency\"`` if gated, else ``None`` to admit.
        """
        return None
