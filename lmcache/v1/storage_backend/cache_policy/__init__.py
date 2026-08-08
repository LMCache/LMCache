# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Type

# First Party
from lmcache.v1.storage_backend.cache_policy.admission_control import (
    AdmissionControlledPolicy,
    WindowedAdmissionControlledPolicy,
)
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.cost_aware_policy import (
    CostAwareEvictionPolicy,
)
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.mru import MRUCachePolicy

# Checked before _ADMISSION_PREFIX -- "WINDOWED_ADMISSION_LRU" does not
# start with "ADMISSION_", so order between the two doesn't actually
# matter, but keeping the more specific prefix first reads clearer.
_WINDOWED_ADMISSION_PREFIX = "WINDOWED_ADMISSION_"
_ADMISSION_PREFIX = "ADMISSION_"

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
            Prefixing any supported name with "ADMISSION_" (e.g.
            "ADMISSION_LRU", "ADMISSION_COST_AWARE") wraps it in
            AdmissionControlledPolicy; prefixing with "WINDOWED_ADMISSION_"
            (e.g. "WINDOWED_ADMISSION_LRU") wraps it in
            WindowedAdmissionControlledPolicy instead -- a second, separate
            admission-control design (see that class's docstring for how
            it differs). Either way the inner name is resolved recursively
            through this same function, so any current or future policy
            name can be wrapped.
        config: Optional LMCacheEngineConfig instance.
        kwargs: Optional explicit policy parameters, forwarded to the
            (possibly inner, if admission-controlled) policy's constructor.
            For "ADMISSION_<INNER>" names, ``halve_every`` is instead
            forwarded to ``AdmissionControlledPolicy`` itself (the
            frequency-sketch decay window), not to the inner policy. For
            "WINDOWED_ADMISSION_<INNER>" names, ``halve_every``,
            ``window_capacity``, and ``promotion_threshold`` are similarly
            forwarded to ``WindowedAdmissionControlledPolicy`` itself. For
            "COST_AWARE" (including as an inner policy under either
            admission prefix), ``clock`` is forwarded to
            ``CostAwareEvictionPolicy`` -- callers driving it through a
            deterministic simulation (see
            ``lmcache.tools.cache_policy_bench.runner``) should pass a
            logical clock here instead of relying on the real-time
            default.

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()

    if upper_policy_name.startswith(_WINDOWED_ADMISSION_PREFIX):
        inner_name = upper_policy_name[len(_WINDOWED_ADMISSION_PREFIX) :]
        remaining_kwargs = dict(kwargs)
        windowed_kwargs: Dict[str, Any] = {}
        for param in ("halve_every", "window_capacity", "promotion_threshold"):
            if param in remaining_kwargs:
                windowed_kwargs[param] = remaining_kwargs.pop(param)
        inner_policy = get_cache_policy(inner_name, config=config, **remaining_kwargs)
        return WindowedAdmissionControlledPolicy(inner_policy, **windowed_kwargs)

    if upper_policy_name.startswith(_ADMISSION_PREFIX):
        inner_name = upper_policy_name[len(_ADMISSION_PREFIX) :]
        remaining_kwargs = dict(kwargs)
        admission_kwargs: Dict[str, Any] = {}
        if "halve_every" in remaining_kwargs:
            admission_kwargs["halve_every"] = remaining_kwargs.pop("halve_every")
        inner_policy = get_cache_policy(inner_name, config=config, **remaining_kwargs)
        return AdmissionControlledPolicy(inner_policy, **admission_kwargs)

    if upper_policy_name not in POLICY_MAPPING:
        raise ValueError(
            f"Unknown cache policy: {upper_policy_name}."
            f" Supported policies are: {list(POLICY_MAPPING.keys())}"
        )

    if upper_policy_name == "COST_AWARE":
        half_life = kwargs.get("half_life_seconds")
        alpha = kwargs.get("cost_ewma_alpha")
        clock = kwargs.get("clock")
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
        if clock is not None:
            policy_kwargs["clock"] = clock

        return CostAwareEvictionPolicy(**policy_kwargs)

    return POLICY_MAPPING[upper_policy_name]()
