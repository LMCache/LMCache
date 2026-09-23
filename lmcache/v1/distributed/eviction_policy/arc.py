# SPDX-License-Identifier: Apache-2.0
"""Adaptive Replacement Cache eviction policy for MP mode."""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Iterator
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class ARCEvictionPolicy(EvictionPolicy):
    """Adaptive Replacement Cache policy for MP L1 and L2 eviction.

    Resident keys are split between ``T1`` (recent) and ``T2`` (frequent).
    ``B1`` and ``B2`` retain key-only history for policy-selected removals from
    ``T1`` and ``T2``. Recreating a key from ``B1`` grows the target size of
    ``T1``; recreating one from ``B2`` shrinks it.

    The MP eviction API requests a fraction of the currently tracked keys and
    does not expose a fixed entry capacity. The policy therefore uses the
    largest observed resident-key count as its entry capacity. This matches
    fixed-size KV chunks and fixed-slot adapters; it is not byte-weighted for
    adapters whose objects have different sizes.

    Unlike the request-coupled ``REPLACE(x)`` operation in the original ARC
    paper, MP asks the policy for pressure-driven eviction batches after writes
    have completed. Following vLLM's CPU-offload ARC variant, an equality at
    the adaptive boundary selects from ``T1`` (``len(T1) >= int(p)``). This
    gives MP a deterministic replacement rule without pretending that an
    eviction action is associated with one particular incoming key.

    Thread Safety:
        Every state transition is protected by one lock.
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ) -> None:
        """Initialize an empty ARC policy.

        Args:
            default_destination: Destination used when no destination has been
                registered by an eviction controller.
        """
        self._lock = threading.Lock()
        self._t1: OrderedDict[ObjectKey, None] = OrderedDict()
        self._t2: OrderedDict[ObjectKey, None] = OrderedDict()
        self._b1: OrderedDict[ObjectKey, None] = OrderedDict()
        self._b2: OrderedDict[ObjectKey, None] = OrderedDict()
        self._pending_evictions: set[ObjectKey] = set()
        self._target_t1_size = 0.0
        self._capacity = 0
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
        """Track newly resident keys and apply ghost-history feedback.

        Existing resident keys are treated as accesses because the MP listener
        currently reports both newly created and updated keys through this
        callback. Keys are processed in reverse request order so later suffix
        chunks remain earlier eviction candidates, matching the MP LRU policy.

        Args:
            keys: Keys that became resident after a completed write.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                self._pending_evictions.discard(key)
                if key in self._t1:
                    del self._t1[key]
                    self._t2[key] = None
                elif key in self._t2:
                    self._t2.move_to_end(key)
                elif key in self._b1:
                    self._target_t1_size = min(
                        self._capacity,
                        self._target_t1_size
                        + self._adaptation_delta(self._b1, self._b2),
                    )
                    del self._b1[key]
                    self._t2[key] = None
                elif key in self._b2:
                    self._target_t1_size = max(
                        0.0,
                        self._target_t1_size
                        - self._adaptation_delta(self._b2, self._b1),
                    )
                    del self._b2[key]
                    self._t2[key] = None
                else:
                    self._remove_key(key)
                    self._t1[key] = None

                self._capacity = max(self._capacity, self._resident_size())
                self._trim_ghost_lists()

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Promote touched resident keys to the frequent list.

        Args:
            keys: Resident keys that were accessed.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._t1:
                    del self._t1[key]
                    self._t2[key] = None
                elif key in self._t2:
                    self._t2.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Apply completed removals to resident and ghost state.

        A key returned by the most recent eviction decision enters the ghost
        list corresponding to its resident list. Other removals are explicit
        lifecycle operations and are forgotten without creating ARC feedback.

        Args:
            keys: Keys confirmed deleted by the L1 manager or L2 adapter.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                if key in self._pending_evictions:
                    self._retire(key)
                else:
                    self._remove_key(key)
                self._pending_evictions.discard(key)
            self._trim_ghost_lists()

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Choose ARC victims for a pressure-driven MP eviction cycle.

        Args:
            expected_ratio: Approximate fraction of resident keys to evict,
                clamped to ``[0.0, 1.0]``.
            key_eligible_filter: Optional predicate used to skip locked or
                otherwise ineligible keys.
            cache_salt: Ignored because ARC is a global, non-isolated policy.

        Returns:
            Zero or one eviction action containing eligible keys in ARC victim
            order. A positive ratio selects at least one key when possible.
        """
        del cache_salt
        with self._lock:
            resident_size = self._resident_size()
            if resident_size == 0:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(resident_size * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            self._capacity = max(self._capacity, resident_size)
            self._target_t1_size = min(self._target_t1_size, self._capacity)
            self._trim_ghost_lists()

            selected: set[ObjectKey] = set()
            victims: list[ObjectKey] = []
            virtual_t1_size = len(self._t1)
            t1_iter = iter(tuple(self._t1))
            t2_iter = iter(tuple(self._t2))

            while len(victims) < target_count:
                candidate = None
                # MP eviction is decoupled from the incoming write, so use the
                # request-independent equality rule used by vLLM's ARC policy.
                if virtual_t1_size >= int(self._target_t1_size):
                    candidate = self._next_eligible(
                        t1_iter, selected, key_eligible_filter
                    )
                    if candidate is not None:
                        virtual_t1_size -= 1
                if candidate is None:
                    candidate = self._next_eligible(
                        t2_iter, selected, key_eligible_filter
                    )
                if candidate is None:
                    candidate = self._next_eligible(
                        t1_iter, selected, key_eligible_filter
                    )
                    if candidate is not None:
                        virtual_t1_size -= 1
                if candidate is None:
                    break

                selected.add(candidate)
                victims.append(candidate)

            if not victims:
                return []

            # A later on_keys_removed callback confirms which requested
            # deletions succeeded before those keys enter ghost history.
            self._pending_evictions.update(victims)
            destination = (
                self._destinations[0]
                if self._destinations
                else self._default_destination
            )
            return [EvictionAction(keys=victims, destination=destination)]

    def get_num_tracked_keys(self) -> int:
        """Return the number of resident keys, excluding ghost history."""
        with self._lock:
            return self._resident_size()

    def get_debug_state(self) -> dict[str, object]:
        """Return a copy of ARC state for tests and diagnostics."""
        with self._lock:
            return {
                "t1": list(self._t1),
                "t2": list(self._t2),
                "b1": list(self._b1),
                "b2": list(self._b2),
                "capacity": self._capacity,
                "target_t1_size": self._target_t1_size,
            }

    def _next_eligible(
        self,
        entries: Iterator[ObjectKey],
        selected: set[ObjectKey],
        key_eligible_filter: Callable[[ObjectKey], bool] | None,
    ) -> ObjectKey | None:
        for key in entries:
            if key in selected:
                continue
            if key_eligible_filter is not None and not key_eligible_filter(key):
                continue
            return key
        return None

    def _retire(self, key: ObjectKey) -> None:
        if key in self._t1:
            del self._t1[key]
            self._b1[key] = None
        elif key in self._t2:
            del self._t2[key]
            self._b2[key] = None

    def _remove_key(self, key: ObjectKey) -> None:
        self._t1.pop(key, None)
        self._t2.pop(key, None)
        self._b1.pop(key, None)
        self._b2.pop(key, None)

    @staticmethod
    def _adaptation_delta(
        hit_list: OrderedDict[ObjectKey, None],
        other_list: OrderedDict[ObjectKey, None],
    ) -> float:
        return max(1.0, len(other_list) / len(hit_list))

    def _trim_ghost_lists(self) -> None:
        if self._capacity <= 0:
            return

        while len(self._t1) + len(self._b1) > self._capacity and self._b1:
            self._b1.popitem(last=False)

        while self._tracked_size() > 2 * self._capacity:
            ghost_list = self._b2 if self._b2 else self._b1
            if not ghost_list:
                break
            ghost_list.popitem(last=False)

    def _resident_size(self) -> int:
        return len(self._t1) + len(self._t2)

    def _tracked_size(self) -> int:
        return self._resident_size() + len(self._b1) + len(self._b2)
