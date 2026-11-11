# SPDX-License-Identifier: Apache-2.0
"""Minimal external selectors for the multiprocess server.

Select these classes explicitly with ``examples.external_policies.policies:...``
when the checkout is importable, or copy them into an installed Python package.
They demonstrate loading and existing contracts, not a cost-aware algorithm.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import StorePolicy
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor


class KeepL1Only(StorePolicy):
    """Keep L1 data without submitting copies to any L2 adapter."""

    def select_store_targets(
        self, keys: list[ObjectKey], adapters: list[L2AdapterDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        """Return no L2 targets; GPU-to-L1 stores have already happened."""
        return {}

    def select_l1_deletions(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Keep L1 copies; ordinary capacity eviction remains in effect."""
        return []


class RetainLoaded(RetainPrefetchPolicy):
    """Reuse the built-in load selection and retain loaded objects in L1."""
