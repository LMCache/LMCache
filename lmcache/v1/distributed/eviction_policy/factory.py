# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.eviction_policy.arc import ARCEvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.noop import NoOpEvictionPolicy


def CreateEvictionPolicy(
    eviction_config: EvictionConfig, *, defer_subscription: bool = False
) -> EvictionPolicy:
    """
    Factory method to create an eviction policy based on the provided configuration.

    Args:
        eviction_config (EvictionConfig): The configuration for the eviction policy.
        defer_subscription: Delay LocalValue observation until its owning
            StorageManager commits construction. Legacy policies are unchanged.

    Returns:
        An instance of the specified eviction policy.
    """
    if eviction_config.eviction_policy == "LRU":
        return LRUEvictionPolicy()
    elif eviction_config.eviction_policy == "ARC":
        return ARCEvictionPolicy()
    elif eviction_config.eviction_policy == "IsolatedLRU":
        return IsolatedLRUEvictionPolicy()
    elif eviction_config.eviction_policy == "noop":
        return NoOpEvictionPolicy()
    elif eviction_config.eviction_policy in {"Profiled", "ProfiledValue"}:
        # First Party
        from lmcache.v1.distributed.eviction_policy.profiled import (
            ProfileConfig,
            ProfiledEvictionPolicy,
        )
        from lmcache.v1.mp_observability.event_bus import (
            get_event_bus,
            is_observability_enabled,
        )

        if not eviction_config.profiling_config or not is_observability_enabled():
            raise ValueError(
                "Profiled requires --profiling-config and an enabled event bus"
            )
        if eviction_config.eviction_policy == "ProfiledValue":
            # First Party
            from lmcache.v1.distributed.eviction_policy.local_value import (
                LocalValueConfig,
                LocalValuePolicy,
            )

            return LocalValuePolicy(
                LocalValueConfig.load(eviction_config.profiling_config),
                get_event_bus(),
                register=not defer_subscription,
            )
        return ProfiledEvictionPolicy(
            ProfileConfig.load(eviction_config.profiling_config), get_event_bus()
        )
    else:
        raise ValueError(
            f"Unsupported eviction policy: {eviction_config.eviction_policy}"
        )
