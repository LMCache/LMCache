# SPDX-License-Identifier: Apache-2.0
"""SIEVE eviction policy for multiprocess L1 and L2 caches."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


@dataclass(slots=True)
class _SieveEntry:
    key: ObjectKey
    visited: bool = False
    newer: "_SieveEntry | None" = None
    older: "_SieveEntry | None" = None


class SieveEvictionPolicy(EvictionPolicy):
    """SIEVE policy for scan-resistant MP cache eviction.

    Entries stay in FIFO insertion order. Touches set a visited bit without
    reordering the queue. During eviction, a hand walks from older entries
    toward newer entries, clears visited bits to grant second chances, and
    selects the first unvisited entries. This makes one-time scans leave hot
    entries in place without maintaining full recency order on every access.

    L1 reservations are inserted immediately so abandoned writes remain
    evictable. The matching write-complete notification does not count as a
    touch.

    Thread Safety:
        Every state transition is protected by one lock.

    Attributes:
        _entries: Key-to-entry index for constant-time lifecycle updates.
        _head: Newest queue entry.
        _tail: Oldest queue entry.
        _hand: Entry inspected next by an eviction request.
        _pending_admissions: Reserved keys awaiting write completion.
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ) -> None:
        """Initialize an empty SIEVE policy.

        Args:
            default_destination: Destination used when no destination has been
                registered by an eviction controller.
        """
        self._lock = threading.Lock()
        self._entries: dict[ObjectKey, _SieveEntry] = {}
        self._head: _SieveEntry | None = None
        self._tail: _SieveEntry | None = None
        self._hand: _SieveEntry | None = None
        self._pending_admissions: set[ObjectKey] = set()
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        """Register an eviction destination.

        Args:
            destination: Destination to use for returned eviction actions.
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Track newly resident keys or mark existing keys as visited.

        Keys are processed in reverse request order so later suffix chunks are
        older eviction candidates, matching the MP LRU policy. A key already
        inserted by :meth:`on_keys_reserved` is left unvisited because write
        completion is not a cache hit.

        Args:
            keys: Keys that became resident after a completed write.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._pending_admissions:
                    self._pending_admissions.remove(key)
                    continue
                entry = self._entries.get(key)
                if entry is None:
                    self._insert_at_head(key)
                else:
                    entry.visited = True

    def on_keys_reserved(self, keys: list[ObjectKey]) -> None:
        """Track write reservations without treating admission as a hit.

        Args:
            keys: Keys reserved for creation.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                self._pending_admissions.add(key)
                entry = self._entries.get(key)
                if entry is not None:
                    self._remove_entry(entry)
                self._insert_at_head(key)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Give resident keys a second chance without reordering the queue.

        Args:
            keys: Resident keys that were accessed.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.visited = True

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Stop tracking keys confirmed absent from the cache.

        Args:
            keys: Keys deleted by the L1 manager or L2 adapter.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                self._pending_admissions.discard(key)
                entry = self._entries.get(key)
                if entry is not None:
                    self._remove_entry(entry)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Choose victims by advancing the SIEVE hand.

        Selection advances the hand and clears visited bits, but keeps keys
        resident until :meth:`on_keys_removed` confirms deletion. A positive
        ratio selects at least one eligible key when possible.

        Args:
            expected_ratio: Approximate fraction of resident keys to evict,
                clamped to ``[0.0, 1.0]``.
            key_eligible_filter: Optional predicate used to skip locked or
                otherwise ineligible keys.
            cache_salt: Ignored because SIEVE is a global, non-isolated policy.

        Returns:
            Zero or one eviction action containing eligible keys in SIEVE hand
            order.
        """
        del cache_salt
        with self._lock:
            entry_count = len(self._entries)
            if entry_count == 0:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(entry_count * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            selected: set[ObjectKey] = set()
            victims: list[ObjectKey] = []
            inspected = 0
            while len(victims) < target_count and inspected < 2 * entry_count:
                entry = self._hand
                assert entry is not None
                self._hand = self._next_entry(entry)
                inspected += 1

                if entry.key in selected:
                    continue
                if key_eligible_filter is not None and not key_eligible_filter(
                    entry.key
                ):
                    continue
                if entry.visited:
                    entry.visited = False
                    continue
                selected.add(entry.key)
                victims.append(entry.key)

            if not victims:
                return []

            destination = (
                self._destinations[0]
                if self._destinations
                else self._default_destination
            )
            return [EvictionAction(keys=victims, destination=destination)]

    def get_num_tracked_keys(self) -> int:
        """Return the number of resident keys tracked by SIEVE.

        Returns:
            Number of entries currently present in the policy queue.
        """
        with self._lock:
            return len(self._entries)

    def _insert_at_head(self, key: ObjectKey) -> None:
        entry = _SieveEntry(key=key, older=self._head)
        if self._head is None:
            self._tail = entry
            self._hand = entry
        else:
            self._head.newer = entry
        self._head = entry
        self._entries[key] = entry

    def _remove_entry(self, entry: _SieveEntry) -> None:
        next_hand = self._next_entry(entry) if len(self._entries) > 1 else None
        if entry.newer is None:
            self._head = entry.older
        else:
            entry.newer.older = entry.older
        if entry.older is None:
            self._tail = entry.newer
        else:
            entry.older.newer = entry.newer
        if self._hand is entry:
            self._hand = next_hand
        del self._entries[entry.key]

    def _next_entry(self, entry: _SieveEntry) -> _SieveEntry:
        if entry.newer is not None:
            return entry.newer
        assert self._tail is not None
        return self._tail
