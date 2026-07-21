# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Type

# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.cost_aware_policy import (
    CostAwareEvictionPolicy,
)
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
    "COST_AWARE": CostAwareEvictionPolicy,
}


def get_cache_policy(
    policy_name: str,
    config: Any = None,
    **kwargs: Any,
) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.

    Args:
        policy_name: Name of the cache policy (case-insensitive, e.g., "LRU", "lru").
        config: Optional LMCacheEngineConfig instance.
        kwargs: Optional explicit policy parameters.

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()

    if upper_policy_name not in POLICY_MAPPING:
        raise ValueError(
            f"Unknown cache policy: {upper_policy_name}."
            f" Supported policies are: {list(POLICY_MAPPING.keys())}"
        )

    if upper_policy_name == "COST_AWARE":
        half_life = kwargs.get("half_life_seconds")
        alpha = kwargs.get("cost_ewma_alpha")
        if config is not None:
            if half_life is None:
                half_life = getattr(config, "cost_aware_half_life_seconds", None)
                if half_life is None and hasattr(config, "get_extra_config_value"):
                    half_life = config.get_extra_config_value(
                        "cost_aware_half_life_seconds", None
                    )
            if alpha is None:
                alpha = getattr(config, "cost_aware_ewma_alpha", None)
                if alpha is None and hasattr(config, "get_extra_config_value"):
                    alpha = config.get_extra_config_value(
                        "cost_aware_ewma_alpha", None
                    )

        policy_kwargs: Dict[str, Any] = {}
        if half_life is not None:
            policy_kwargs["half_life_seconds"] = float(half_life)
        if alpha is not None:
            policy_kwargs["cost_ewma_alpha"] = float(alpha)

        return CostAwareEvictionPolicy(**policy_kwargs)

    return POLICY_MAPPING[upper_policy_name]()
