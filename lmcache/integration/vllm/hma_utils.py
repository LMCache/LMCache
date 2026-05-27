# SPDX-License-Identifier: Apache-2.0
"""vLLM Hybrid Memory Allocator compatibility helpers.

The fields used here are part of vLLM's v1 KV cache interface:
``KVCacheConfig.kv_cache_groups`` and ``KVCacheGroupSpec.layer_names``.
Keeping the access in one small module makes it easier to adjust if vLLM
changes that interface.
"""

# Standard
from collections.abc import Mapping, Sequence
from typing import Any


def _kv_cache_groups(kv_cache_config: Any) -> Sequence[Any]:
    """Return vLLM KV cache groups, or an empty sequence when unavailable."""
    if kv_cache_config is None:
        return ()
    return getattr(kv_cache_config, "kv_cache_groups", ()) or ()


def get_num_engine_groups(kv_cache_config: Any) -> int:
    """Return the number of engine-side KV cache groups.

    Non-HMA vLLM configurations have no group metadata from LMCache's point of
    view, so they are treated as a single engine group.
    """
    kv_cache_groups = _kv_cache_groups(kv_cache_config)
    return len(kv_cache_groups) if kv_cache_groups else 1


def build_engine_group_layout_hints(
    kv_cache_config: Any,
    kv_caches: Mapping[str, object],
) -> dict[str, object] | None:
    """Build LMCache layout hints from vLLM KV cache group metadata.

    vLLM groups layers by shared block table. LMCache regroups layers by
    transfer shape, so the worker needs a per-layer ``engine_group_idx`` hint
    to choose the correct engine block-id list for each LMCache transfer group.

    Returns ``None`` when no HMA group mapping is present.
    """
    kv_cache_groups = _kv_cache_groups(kv_cache_config)
    layer_to_pos = {name: idx for idx, name in enumerate(kv_caches)}
    if not kv_cache_groups or not layer_to_pos:
        return None

    per_layer_engine_group_idx = [0] * len(layer_to_pos)
    matched_layers: set[str] = set()
    for engine_group_idx, group in enumerate(kv_cache_groups):
        for name in getattr(group, "layer_names", ()):
            pos = layer_to_pos.get(name)
            if pos is not None:
                per_layer_engine_group_idx[pos] = engine_group_idx
                matched_layers.add(name)

    if matched_layers:
        missing_layers = set(layer_to_pos) - matched_layers
        if missing_layers:
            raise ValueError(
                "vLLM kv_cache_groups did not cover registered KV cache "
                f"layers: {sorted(missing_layers)[:8]}"
            )
        return {"per_layer_engine_group_idx": per_layer_engine_group_idx}

    if len(kv_cache_groups) > 1:
        raise ValueError(
            "Unable to map registered KV cache layers to vLLM kv_cache_groups "
            "for HMA."
        )

    return None
