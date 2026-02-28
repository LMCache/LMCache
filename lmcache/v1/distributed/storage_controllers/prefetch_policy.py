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
    ) -> dict[int, Bitmap]:
        """
        Decide which adapter loads which keys.

        Args:
            keys: Full list of keys being prefetched from L2.
            lookup_results: Mapping from adapter index to Bitmap.
                A set bit at position i means the adapter has keys[i].
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to a bitmap telling which keys that
            the adapter should load. The returned bitmaps should not
            overlap, and the union of all returned bitmaps should be a subset
            of the union of the input bitmaps.
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
    ) -> dict[int, Bitmap]:
        """
        Assign each key to the first adapter (by index) that has it.

        Args:
            keys: Full list of keys being prefetched.
            lookup_results: Adapter index -> Bitmap of lookup hits.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to key bitmaps. Each key goes
            to the lowest-indexed adapter that reported having it.
        """
        plan: dict[int, Bitmap] = {}
        global_bitmap = Bitmap(len(keys))
        for bitmap in lookup_results.values():
            global_bitmap |= bitmap

        for ad in sorted(adapters, key=lambda a: a.index):
            curr_bitmap = lookup_results.get(ad.index)
            if curr_bitmap is None:
                continue

            local_bitmap = global_bitmap & curr_bitmap
            global_bitmap &= ~local_bitmap
            if local_bitmap.popcount() == 0:
                continue

            plan[ad.index] = local_bitmap

        return plan
