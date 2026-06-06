# SPDX-License-Identifier: Apache-2.0
"""
S3FIFO-like eviction policy adapted for LMCache framework.

This implementation is event-driven (like LRU) and uses:
- Small FIFO (probation queue)
- Main FIFO (protected queue)
- Optional Ghost set (history tracking)

The policy decides eviction only inside get_eviction_actions().
"""

from collections import OrderedDict
from collections.abc import Callable
import threading

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class S3FIFOEvictionPolicy(EvictionPolicy):
    """
    Simplified S3FIFO eviction policy.

    Design:
        - Small queue: new keys enter here (FIFO)
        - Main queue: promoted (hot) keys
        - Ghost set: tracks previously evicted keys (optional hint)

    Notes:
        - Fully event-driven like LRU in this codebase
        - No background processing
        - Eviction is decided only in get_eviction_actions()
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
        small_max_size: int | None = None,
    ):
        self._lock = threading.Lock()

        # FIFO structures
        self._small: OrderedDict[ObjectKey, None] = OrderedDict()
        self._main: OrderedDict[ObjectKey, None] = OrderedDict()

        # Ghost set (optional history tracking)
        self._ghost: set[ObjectKey] = set()

        # Eviction destinations
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

        # Optional tuning parameter (not required by framework)
        self._small_max_size = small_max_size

    # ---------------------------------------------------------------------
    # Required interface methods
    # ---------------------------------------------------------------------

    def register_eviction_destination(self, destination: EvictionDestination):
        """Register eviction destination."""
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]):
        """
        New keys enter the SMALL FIFO (probation stage).
        """
        if not keys:
            return

        with self._lock:
            for key in keys:
                # If already exists anywhere, ignore duplicates safely
                if key in self._small or key in self._main:
                    continue

                # If seen before (ghost hit), promote directly to main
                if key in self._ghost:
                    self._main[key] = None
                    self._ghost.remove(key)
                else:
                    self._small[key] = None

            # Optional: enforce small queue size limit (if configured)
            self._evict_small_if_needed()

    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Access pattern:
        - Small → Main (promotion)
        - Main → move to end (LRU-style freshness inside main)
        """
        if not keys:
            return

        with self._lock:
            for key in keys:
                if key in self._small:
                    # Promote to main
                    del self._small[key]
                    self._main[key] = None

                elif key in self._main:
                    # Refresh position (LRU behavior inside main)
                    self._main.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Remove keys from all internal structures.
        """
        if not keys:
            return

        with self._lock:
            for key in keys:
                self._small.pop(key, None)
                self._main.pop(key, None)
                self._ghost.discard(key)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """
        Decide which keys to evict.

        Strategy:
        1. Evict from SMALL first (FIFO)
        2. Then from MAIN (FIFO/LRU order)
        3. Optionally record in GHOST set
        """
        with self._lock:
            if not self._small and not self._main:
                return []

            # Clamp ratio
            expected_ratio = max(0.0, min(1.0, expected_ratio))

            total = len(self._small) + len(self._main)
            target = int(total * expected_ratio)

            if expected_ratio > 0 and target == 0 and total > 0:
                target = 1

            if target == 0:
                return []

            evicted: list[ObjectKey] = []

            # -------------------------
            # 1. Evict from SMALL (FIFO)
            # -------------------------
            for key in list(self._small.keys()):
                if len(evicted) >= target:
                    break

                if key_eligible_filter and not key_eligible_filter(key):
                    continue

                self._small.pop(key, None)
                evicted.append(key)
                self._ghost.add(key)

            # -------------------------
            # 2. Evict from MAIN (FIFO/LRU)
            # -------------------------
            for key in list(self._main.keys()):
                if len(evicted) >= target:
                    break

                if key_eligible_filter and not key_eligible_filter(key):
                    continue

                self._main.pop(key, None)
                evicted.append(key)
                self._ghost.add(key)

            if not evicted:
                return []

            destination = (
                self._destinations[0]
                if self._destinations
                else self._default_destination
            )

            return [EvictionAction(keys=evicted, destination=destination)]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _evict_small_if_needed(self):
        """
        Optional constraint for SMALL queue size.
        Not required by framework but useful for stability.
        """
        if self._small_max_size is None:
            return

        while len(self._small) > self._small_max_size:
            key, _ = self._small.popitem(last=False)
            self._ghost.add(key)