# SPDX-License-Identifier: Apache-2.0
"""Convert vLLM KV cache group metadata into LMCache's neutral model."""

# Standard
from collections.abc import Sequence
from typing import Any

# First Party
from lmcache.v1.kv_cache_groups import LMCKVCacheGroup, LMCKVCacheGroups


def _vllm_kv_cache_groups(kv_cache_config: Any) -> Sequence[Any]:
    """Return vLLM KV cache groups, or an empty sequence when unavailable."""
    if kv_cache_config is None:
        return ()
    return getattr(kv_cache_config, "kv_cache_groups", ()) or ()


def lmcache_kv_cache_groups_from_vllm(
    kv_cache_config: Any,
    registered_layer_names: Sequence[str] | None = None,
) -> LMCKVCacheGroups:
    """Convert vLLM's KV cache group interface to LMCache's neutral type.

    The vLLM fields used here are from the v1 KV cache interface:
    ``KVCacheConfig.kv_cache_groups`` and ``KVCacheGroupSpec.layer_names``.
    Keep additional vLLM-specific field reads in this module, then normalize
    them into ``LMCKVCacheGroup`` fields before crossing LMCache boundaries.
    """
    layer_to_idx = (
        {name: idx for idx, name in enumerate(registered_layer_names)}
        if registered_layer_names is not None
        else {}
    )

    return LMCKVCacheGroups.from_groups(
        LMCKVCacheGroup(
            engine_kv_cache_group_id=engine_kv_cache_group_id,
            layer_names=tuple(getattr(group, "layer_names", ())),
            layer_indices=tuple(
                layer_to_idx[name]
                for name in getattr(group, "layer_names", ())
                if name in layer_to_idx
            ),
        )
        for engine_kv_cache_group_id, group in enumerate(
            _vllm_kv_cache_groups(kv_cache_config)
        )
    )
