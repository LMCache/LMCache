# SPDX-License-Identifier: Apache-2.0
"""
LRU (Least Recently Used) eviction policy implementation
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class LRUEvictionPolicy(EvictionPolicy):
    """
    LRU (Least Recently Used) eviction policy.

    This policy tracks the order of key accesses and evicts the least recently
    used keys first when eviction is needed.

    Thread Safety:
        This class is thread-safe. All operations are protected by a global lock
        to ensure eventual consistency without corrupted states.

    Attributes:
        _lock: Threading lock for thread-safe operations
        _order: OrderedDict maintaining LRU order and object sizes
        _destinations: List of registered eviction destinations
        _default_destination: The default destination for eviction actions
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        """
        Initialize the LRU eviction policy.

        Args:
            default_destination: The default destination for evicted objects.
                Defaults to DISCARD.
        """
        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # OrderedDict to maintain LRU order - keys at the beginning are oldest
        self._order: OrderedDict[ObjectKey, int | None] = OrderedDict()
        self._total_size: int | None = None
        self._total_size_squared: int | None = None

        # List of registered eviction destinations
        self._destinations: list[EvictionDestination] = []

        # Default destination for eviction
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to register
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """
        Notify the eviction policy that new keys have been created.
        New keys are added as most recently used.

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        if not keys:
            return
        with self._lock:
            if self._total_size is None:
                for key in reversed(keys):
                    if key in self._order:
                        self._order.move_to_end(key)
                    else:
                        self._order[key] = None
                return
            for key in reversed(keys):
                self._store_sized_key(key, 1)

    def on_keys_created_with_sizes(
        self,
        keys: list[ObjectKey],
        sizes: list[int],
    ) -> None:
        """Track newly created keys and their eviction weights.

        Args:
            keys: Keys that have been created.
            sizes: Size in bytes for each key.

        Raises:
            ValueError: If ``keys`` and ``sizes`` have different lengths.
        """
        if len(keys) != len(sizes):
            raise ValueError("keys and sizes must have the same length")
        if not keys:
            return
        with self._lock:
            self._enable_size_tracking()
            # NOTE: for the request, the later keys should be evicted first.
            # For example, the request has (key1, key2, key3), if we first
            # evict key1, due to prefix match, key2 and key3 will not be hit.
            for key, size in zip(reversed(keys), reversed(sizes), strict=True):
                self._store_sized_key(key, size)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """
        Notify the eviction policy that keys have been accessed.
        Touched keys are moved to the most recently used position.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        if not keys:
            return
        with self._lock:
            # NOTE: for the request, the later keys should be evicted first.
            # The example is the same as `on_keys_created`.
            for key in reversed(keys):
                if key in self._order:
                    # Move to end (most recently used)
                    self._order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """
        Notify the eviction policy that keys have been deleted.
        Deleted keys are removed from tracking.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        if not keys:
            return
        with self._lock:
            if self._total_size is None:
                for key in keys:
                    if key in self._order:
                        del self._order[key]
                return

            removed_size = 0
            removed_size_squared = 0
            for key in keys:
                size = self._order.pop(key, None)
                if size is not None:
                    removed_size += size
                    removed_size_squared += size * size
            self._total_size -= removed_size
            assert self._total_size_squared is not None
            self._total_size_squared -= removed_size_squared

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """
        Get the eviction actions to evict objects from L1 cache.
        Returns keys in LRU order (least recently used first).

        Args:
            expected_ratio (float): A hint indicating approximately what fraction
                of tracked eviction weight should be evicted. L2 keys are weighted
                by bytes; callers without sizes use unit weight. Value should be in
                range [0.0, 1.0].
            key_eligible_filter: An optional callable that takes an ObjectKey
                and returns True if the key is eligible for eviction. When
                provided, keys for which the filter returns False will be
                skipped. This is useful for skipping locked keys that
                cannot be deleted.
            cache_salt: Ignored by LRU policy (not user-level).

        Returns:
            list[EvictionAction]: The eviction actions to perform. Each
                action contains the keys and one eviction destination.

        Notes:
            The eviction action may not be successfully executed, or it
            may be executed asynchronously. Therefore, the eviction policy
            should not assume that the objects are evicted immediately, but
            it should use `on_keys_deleted` to know when the objects are actually
            deleted.
        """
        with self._lock:
            if not self._order:
                return []

            # Clamp expected_ratio to valid range
            expected_ratio = max(0.0, min(1.0, expected_ratio))

            keys_to_evict: list[ObjectKey] = []
            if self._sizes_are_uniform():
                # Preserve the original count-based rounding exactly when
                # object sizes do not differ.
                target_count = int(len(self._order) * expected_ratio)
                if expected_ratio > 0 and target_count == 0:
                    target_count = 1
                if target_count == 0:
                    return []
                for key in self._order:
                    if key_eligible_filter is not None and not key_eligible_filter(key):
                        continue
                    keys_to_evict.append(key)
                    if len(keys_to_evict) >= target_count:
                        break
            else:
                assert self._total_size is not None
                target_size = int(self._total_size * expected_ratio)
                if expected_ratio > 0 and target_size == 0:
                    target_size = 1
                if target_size == 0:
                    return []
                selected_size = 0
                for key, size in self._order.items():
                    if key_eligible_filter is not None and not key_eligible_filter(key):
                        continue
                    assert size is not None
                    keys_to_evict.append(key)
                    selected_size += size
                    if selected_size >= target_size:
                        break

            if not keys_to_evict:
                return []

            # Determine the destination
            destination = self._default_destination
            if self._destinations:
                # Use the first registered destination if available
                destination = self._destinations[0]

            return [EvictionAction(keys=keys_to_evict, destination=destination)]

    # =========================================================================
    # Methods below are NOT part of the EvictionPolicy interface.
    # They are provided for testing and debugging purposes only.
    # =========================================================================

    def get_num_tracked_keys(self) -> int:
        """
        Get the number of keys currently being tracked.

        Note:
            This method is NOT part of the EvictionPolicy interface.
            It is provided for testing and debugging purposes only.

        Returns:
            int: The number of tracked keys
        """
        with self._lock:
            return len(self._order)

    def get_eviction_candidates(self, count: int) -> list[ObjectKey]:
        """
        Get a list of eviction candidates without creating eviction actions.
        Useful for querying what would be evicted.

        Note:
            This method is NOT part of the EvictionPolicy interface.
            It is provided for testing and debugging purposes only.

        Args:
            count: Maximum number of candidates to return

        Returns:
            list[ObjectKey]: List of keys that are candidates for eviction,
                in LRU order (least recently used first)
        """
        with self._lock:
            candidates: list[ObjectKey] = []

            for key in self._order:
                candidates.append(key)
                if len(candidates) >= count:
                    break

            return candidates

    def _enable_size_tracking(self) -> None:
        if self._total_size is not None:
            return
        self._total_size = len(self._order)
        self._total_size_squared = len(self._order)
        for key in self._order:
            self._order[key] = 1

    def _store_sized_key(self, key: ObjectKey, size: int) -> None:
        assert self._total_size is not None
        assert self._total_size_squared is not None
        if key in self._order:
            old_size = self._order[key]
            assert old_size is not None
            self._total_size -= old_size
            self._total_size_squared -= old_size * old_size
            self._order.move_to_end(key)

        self._order[key] = size
        self._total_size += size
        self._total_size_squared += size * size

    def _sizes_are_uniform(self) -> bool:
        """Check uniformity using n * sum(size^2) == sum(size)^2."""
        if self._total_size is None:
            return True
        assert self._total_size_squared is not None
        return (
            len(self._order) * self._total_size_squared
            == self._total_size * self._total_size
        )
