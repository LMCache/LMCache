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

# Upper bound on staged chunk positions. Positions are reported by the key
# resolver, which runs on retrieve as well as store, so a lookup that misses
# stages positions for keys that are never created and never reach
# `on_keys_removed`. The map is therefore capped rather than left to grow with
# every chunk ever looked up. Overflow is harmless: the resolver re-reports
# positions before each create and each touch, so a dropped entry costs at most
# one batch its position ordering, and only until that key is next accessed.
MAX_TRACKED_POSITIONS = 1 << 16


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
        _order: OrderedDict maintaining LRU order (most recent at end)
        _destinations: List of registered eviction destinations
        _default_destination: The default destination for eviction actions
        _key_positions: Staged chunk positions, bounded by MAX_TRACKED_POSITIONS
        _prefix_run: Keys of the prefix run currently being ordered
        _prefix_run_hwm: Highest position the run has reached
    """

    # Created on first use, so a policy that never receives positions behaves
    # exactly as before. Declared here for typing only -- an annotation without
    # an assignment does not create the attribute.
    _key_positions: OrderedDict[ObjectKey, int]
    _prefix_run: dict[ObjectKey, int]
    _prefix_run_hwm: int

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
        self._order: OrderedDict[ObjectKey, None] = OrderedDict()

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

    def note_key_positions(self, keys: list[ObjectKey], positions: list[int]) -> None:
        """Record each key's chunk position within its prefix.

        Reported by the key resolver, which is the only component that still knows
        the position. Entries for cached keys are dropped in
        :meth:`on_keys_removed`; entries for keys that are never created -- a
        lookup that missed -- are bounded by `MAX_TRACKED_POSITIONS` instead.
        """
        if not keys:
            return
        with self._lock:
            pm = getattr(self, "_key_positions", None)
            if pm is None:
                pm = self._key_positions = OrderedDict()
            for k, p in zip(keys, positions, strict=False):
                pm[k] = p
                pm.move_to_end(k)
            while len(pm) > MAX_TRACKED_POSITIONS:
                pm.popitem(last=False)

    def _prefix_positions(self, keys):
        """Positions for these keys, or None if any is unknown.

        Caller holds the lock.
        """
        pm = getattr(self, "_key_positions", None)
        if not pm:
            return None
        pos = [pm.get(k) for k in keys]
        return pos if all(p is not None for p in pos) else None

    def _prefix_ordered_insert(
        self, keys: list[ObjectKey], *, insert_missing: bool = True
    ) -> None:
        """Order a batch within its prefix run, or fall back to arrival order.

        `reversed(keys)` only orders within one call, but a prefix arrives across
        several: a serving engine stores one batch per
        `max_num_batched_tokens / chunk_size` chunks, and a cache hit touches the
        matched head as a separate call. Ordering those independently leaves the
        boundary between them least-recently-used, so eviction opens a hole in the
        middle of the prefix -- and a prefix is only usable contiguously from chunk 0.

        Batches are therefore folded into one run and ordered by position. A batch
        continues the run if it SHARES A KEY with it (a touch always does; an
        overlapping store does) or begins exactly at run_max + 1 (a clean extension).
        A different prefix restarting at position 0 does neither, so it correctly
        starts a new run and ordinary cross-request LRU is preserved.

        With `insert_missing=False` an unknown key is dropped instead of admitted,
        for callers that report access rather than storage.

        Caller holds the lock.
        """
        if not insert_missing:
            # A touch reports access, not admission: a key absent from `_order`
            # was evicted (or never stored) and must not be resurrected. Both
            # paths below admit unknown keys unconditionally, so filter here --
            # which also keeps those keys out of the run and its high-water mark.
            keys = [k for k in keys if k in self._order]
        if not keys:
            return
        pos = self._prefix_positions(keys)
        if pos is None:
            for key in reversed(keys):
                if key in self._order:
                    self._order.move_to_end(key)
                else:
                    self._order[key] = None
            return

        run = getattr(self, "_prefix_run", None) or {}
        # High-water mark of the run, tracked separately: eviction removes entries
        # from `run`, and if the extension check used max(run.values()) it would
        # regress every time the tail was trimmed -- so the next batch would look
        # like a new prefix, become most-recently-used, and push eviction back into
        # the middle. The mark only advances, and only resets with the run.
        run_hwm = getattr(self, "_prefix_run_hwm", -1)
        if run or run_hwm >= 0:
            shares_key = any(k in run for k in keys)
            extends = min(pos) <= run_hwm + 1
            if not (shares_key or extends):
                run, run_hwm = {}, -1
        for k, p in zip(keys, pos, strict=True):
            run[k] = p
            if k not in self._order:
                self._order[k] = None
        # Entries evicted out from under us are no longer orderable.
        for k in [k for k in run if k not in self._order]:
            del run[k]
        # Descending position => position 0 ends most-recently-used, so eviction
        # trims the TAIL and the retained set stays a contiguous prefix.
        for k in sorted(run, key=lambda k: -run[k]):
            self._order.move_to_end(k)
        self._prefix_run = run
        self._prefix_run_hwm = max(run_hwm, max(pos))

    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that new keys have been created.
        New keys are added as most recently used.

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        if not keys:
            return
        with self._lock:
            self._prefix_ordered_insert(keys)

    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been accessed.
        Touched keys are moved to the most recently used position.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        if not keys:
            return
        with self._lock:
            self._prefix_ordered_insert(keys, insert_missing=False)

    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been deleted.
        Deleted keys are removed from tracking.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        if not keys:
            return
        with self._lock:
            posmap = getattr(self, "_key_positions", None)
            run = getattr(self, "_prefix_run", None)
            for key in keys:
                # Remove from LRU order tracking
                if key in self._order:
                    del self._order[key]
                # Positions track live cache contents only.
                if posmap is not None:
                    posmap.pop(key, None)
                if run is not None:
                    run.pop(key, None)

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
                of tracked keys should be evicted. Value should be in range [0.0, 1.0].
                For example, 0.1 means roughly 10% of keys should be evicted.
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

            # Calculate target number of keys to evict based on ratio
            target_count = int(len(self._order) * expected_ratio)

            # Ensure at least 1 key if ratio > 0 and we have keys
            if expected_ratio > 0 and target_count == 0 and len(self._order) > 0:
                target_count = 1

            if target_count == 0:
                return []

            # Get keys in LRU order (from beginning - least recently used),
            # skipping keys that fail the filter (e.g. locked keys).
            keys_to_evict: list[ObjectKey] = []

            for key in self._order:
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    # Skip keys that are not eligible for eviction
                    # (e.g. currently locked by read/write operations)
                    continue
                keys_to_evict.append(key)
                if len(keys_to_evict) >= target_count:
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
