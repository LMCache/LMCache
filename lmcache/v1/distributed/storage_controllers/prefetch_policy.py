# SPDX-License-Identifier: Apache-2.0
"""
Prefetch policy interface and default implementation for L2-to-L1 load decisions.

The prefetch policy decides which L2 adapter should load each key when multiple
adapters have the same key. It receives lookup results (bitmaps) from all adapters
and produces a load plan mapping each adapter to the key indices it should load.
"""

# Standard
from abc import ABC, abstractmethod

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)


class PrefetchPolicy(ABC):
    """
    Abstract interface for prefetch load-plan decisions.

    The prefetch policy is called by the PrefetchController after all L2
    adapters have completed their lookup_and_lock operations. Given the
    lookup results, it decides which adapter should load which keys.
    """

    @abstractmethod
    def select_load_plan(
        self,
        keys: list[ObjectKey],
        lookup_results: dict[int, Bitmap],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[int]]:
        """
        Decide which adapter loads which keys.

        Args:
            keys: Full list of keys being prefetched from L2.
            lookup_results: Mapping from adapter index to Bitmap.
                A set bit at position i means the adapter has keys[i].
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to list of key indices that
            adapter should load. Each key index appears in at most
            one adapter's list. Keys not in any list will NOT be loaded.
        """


class DefaultPrefetchPolicy(PrefetchPolicy):
    """
    Default prefetch policy: for each key, pick the first adapter
    (lowest index) that has it.
    """

    def select_load_plan(
        self,
        keys: list[ObjectKey],
        lookup_results: dict[int, Bitmap],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[int]]:
        """
        Assign each key to the first adapter (by index) that has it.

        Args:
            keys: Full list of keys being prefetched.
            lookup_results: Adapter index -> Bitmap of lookup hits.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to key indices. Each key goes
            to the lowest-indexed adapter that reported having it.
        """
        plan: dict[int, list[int]] = {}
        assigned: set[int] = set()

        for ad in sorted(adapters, key=lambda a: a.index):
            bitmap = lookup_results.get(ad.index)
            if bitmap is None:
                continue
            for i in range(len(keys)):
                if i not in assigned and bitmap.test(i):
                    plan.setdefault(ad.index, []).append(i)
                    assigned.add(i)

        return plan
