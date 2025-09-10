# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy


def get_cache_policy(policy_name: str) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.
    """
    supported_policies = ["LRU", "LFU", "FIFO"]
    if policy_name == "LRU":
        return LRUCachePolicy()
    elif policy_name == "LFU":
        return LFUCachePolicy()
    elif policy_name == "FIFO":
        return FIFOCachePolicy()
    else:
        raise ValueError(
            f"Unknown cache policy: {policy_name}"
            f" Supported policies are: {supported_policies}"
        )
