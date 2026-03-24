# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Type

# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.mru import MRUCachePolicy

# Cache policy mapping
POLICY_MAPPING: Dict[str, Type[BaseCachePolicy]] = {
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
}


def get_cache_policy(
    policy_name: str,
    *,
    ssd_gate_min_size_bytes: int = 0,
    ssd_gate_min_access_count: int = 0,
) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.

    Args:
        policy_name: Name of the cache policy (case-insensitive, e.g., "LRU", "lru").
        ssd_gate_min_size_bytes: For LRU only: skip SSD writes smaller than this
            (0 = disabled). Passed to :class:`LRUCachePolicy`.
        ssd_gate_min_access_count: For LRU only: require at least this many
            access-count increments before writing to SSD (0 = disabled).

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()

    try:
        policy_cls = POLICY_MAPPING[upper_policy_name]
    except KeyError:
        raise ValueError(
            f"Unknown cache policy: {upper_policy_name}."
            f" Supported policies are: {list(POLICY_MAPPING.keys())}"
        ) from None

    if upper_policy_name == "LRU":
        return LRUCachePolicy(
            ssd_gate_min_size_bytes=ssd_gate_min_size_bytes,
            ssd_gate_min_access_count=ssd_gate_min_access_count,
        )
    return policy_cls()
