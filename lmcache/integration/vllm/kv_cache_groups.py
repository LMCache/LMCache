# SPDX-License-Identifier: Apache-2.0
"""Convert vLLM KV cache group metadata into LMCache's neutral model."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.v1.kv_cache_groups import LMCacheKVGroup, LMCacheKVSpec


def create_lmcache_kv_spec_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
) -> LMCacheKVSpec:
    """Build the LMCache KV spec from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    :func:`lmcache.v1.kv_layer_groups.create_lmcache_kv_spec`). vLLM-specific
    field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).

    Returns:
        The ``LMCacheKVSpec`` whose group order is the protocol-visible LMCache
        group order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
    from lmcache.v1.kv_layer_groups import create_lmcache_kv_spec

    # Map each vLLM engine KV cache group to LMCache's neutral group form,
    # resolving layer names to registered tensor indices. This is the only
    # place that reads vLLM ``KVCacheConfig`` fields.
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    engine_kv_spec = LMCacheKVSpec.from_groups(
        LMCacheKVGroup(
            hybrid_block_group_id=hybrid_block_group_id,
            layer_indices=tuple(
                layer_to_idx[name]
                for name in getattr(group, "layer_names", ())
                if name in layer_to_idx
            ),
        )
        for hybrid_block_group_id, group in enumerate(vllm_groups)
    )

    # Split engine groups further by physical transfer identity using the real
    # registered tensors.
    gpu_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
        list(kv_caches.values()),
        EngineType.VLLM,
        layout_hints=layout_hints,
    )
    return create_lmcache_kv_spec(
        normalized_kv_caches,
        gpu_kv_format,
        engine_kv_spec,
    )
