# SPDX-License-Identifier: Apache-2.0
"""
Per-user LRU eviction policy.

Maintains a separate LRU OrderedDict for each ``user_id``, so eviction
can be scoped to a single user without disturbing other users' cache
data.
"""

# Standard
from collections import OrderedDict
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class UserLRUEvictionPolicy(EvictionPolicy):
    """Per-user LRU eviction policy.

    Each user's keys are tracked in their own ``OrderedDict`` (most
    recently used at the end).  ``get_eviction_actions`` can target a
    single user or operate globally across all users.

    Thread Safety:
        All operations are protected by a single lock.

    Attributes:
        _lock: Threading lock for thread-safe operations.
        _per_user_order: Per-user OrderedDicts maintaining LRU order.
        _destinations: Registered eviction destinations.
        _default_destination: Default destination for eviction actions.
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        """Initialize the per-user LRU eviction policy.

        Args:
            default_destination: Default destination for evicted objects.
                Defaults to DISCARD.
        """
        self._lock = threading.Lock()
        self._per_user_order: dict[str, OrderedDict[ObjectKey, None]] = {}
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        """Register an eviction destination.

        Args:
            destination: The eviction destination to register.
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Notify that new keys have been created.

        New keys are added as most recently used within their user's
        LRU list.

        Args:
            keys: The keys that have been created.
        """
        with self._lock:
            # Later keys in a request should be evicted first, so
            # iterate in reverse so earlier keys end up oldest.
            for key in reversed(keys):
                user_id = key.user_id
                if user_id not in self._per_user_order:
                    self._per_user_order[user_id] = OrderedDict()
                user_order = self._per_user_order[user_id]
                if key in user_order:
                    user_order.move_to_end(key)
                else:
                    user_order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Notify that keys have been accessed.

        Touched keys are moved to the most recently used position
        within their user's LRU list.

        Args:
            keys: The keys that have been accessed.
        """
        with self._lock:
            for key in reversed(keys):
                user_id = key.user_id
                user_order = self._per_user_order.get(user_id)
                if user_order and key in user_order:
                    user_order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Notify that keys have been deleted.

        Deleted keys are removed from their user's LRU list. If a
        user's list becomes empty it is cleaned up.

        Args:
            keys: The keys that have been deleted.
        """
        with self._lock:
            for key in keys:
                user_id = key.user_id
                user_order = self._per_user_order.get(user_id)
                if user_order:
                    user_order.pop(key, None)
                    if not user_order:
                        del self._per_user_order[user_id]

    def get_eviction_actions(
        self,
        expected_ratio: float,
        user_id: str | None = None,
    ) -> list[EvictionAction]:
        """Select victims, optionally scoped to a single user.

        Args:
            expected_ratio: Fraction of keys to evict (0.0 to 1.0).
            user_id: If set, evict only from this user's LRU list.
                If None, evict globally across all users.

        Returns:
            A list containing at most one ``EvictionAction`` with the
            keys to evict and their destination.
        """
        with self._lock:
            if user_id is not None:
                order = self._per_user_order.get(user_id)
                if not order:
                    return []
                pool = list(order.keys())
            else:
                pool = []
                for user_order in self._per_user_order.values():
                    pool.extend(user_order.keys())

            if not pool:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target = int(len(pool) * expected_ratio)
            if expected_ratio > 0 and target == 0 and len(pool) > 0:
                target = 1
            if target == 0:
                return []

            destination = self._default_destination
            if self._destinations:
                destination = self._destinations[0]

            return [
                EvictionAction(
                    keys=pool[:target],
                    destination=destination,
                )
            ]
