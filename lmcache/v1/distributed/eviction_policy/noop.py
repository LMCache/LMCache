# SPDX-License-Identifier: Apache-2.0
"""
No-op eviction policy for buffer-only mode.

When L1 is used purely as a write buffer (the StoreController deletes
data from L1 once it is stored to L2 and no reader holds it), there
is no need for the eviction policy to track keys at all.
"""

# Standard
from collections.abc import Callable

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class NoOpEvictionPolicy(EvictionPolicy):
    """
    Eviction policy that performs no tracking and never evicts.

    Designed for buffer-only mode where the StoreController is
    responsible for cleaning up L1 after successful L2 writes.
    This avoids the overhead of maintaining an LRU OrderedDict
    that would otherwise do pointless insert-then-remove cycles.
    """

    def register_eviction_destination(self, destination: EvictionDestination):
        pass

    def on_keys_created(self, keys: list[ObjectKey]):
        pass

    def on_keys_touched(self, keys: list[ObjectKey]):
        pass

    def on_keys_removed(self, keys: list[ObjectKey]):
        pass

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        return []
