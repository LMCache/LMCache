# SPDX-License-Identifier: Apache-2.0
"""Build LMCache group views from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.v1.multiprocess.group_view import LMCacheGroupView


def create_group_views_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
) -> list[LMCacheGroupView]:
    """Build the LMCache group views from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).

    Returns:
        The list of ``LMCacheGroupView`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        get_num_layers,
        normalize_kv_and_discover_format,
    )
    from lmcache.v1.kv_layer_groups import group_layers_by_identity

    # Inspect the real registered tensors for physical layout and dtype.
    gpu_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
        list(kv_caches.values()),
        EngineType.VLLM,
        layout_hints=layout_hints,
    )
    num_layers = get_num_layers(normalized_kv_caches, gpu_kv_format)

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    per_layer_group_idx: list[int] | None = None
    if vllm_groups:
        per_layer_group_idx = [0] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size, dtype), so
    # every resulting LMCache group can be served by a single copy kernel. It is
    # the shared, engine-neutral primitive the server reuses to reproduce the
    # same grouping from the registered tensors.
    return [
        LMCacheGroupView(
            engine_group_id=identity[4],
            layer_indices=tuple(indices),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            gpu_kv_format,
            num_layers,
            per_layer_group_idx,
        )
    ]


def spec_int_attr(spec: Any, attr: str) -> int:
    """Read an integer ``attr`` off a vLLM ``KVCacheGroupSpec.kv_cache_spec``.

    The value may live directly on the spec, or — for a
    ``UniformTypeKVCacheSpecs`` wrapper that bundles several inner specs — on
    its inner specs (all of which share one value), in which case the first
    inner spec is read. Returns ``0`` when neither carries the attribute (or it
    is ``None``), so callers treat ``0`` uniformly as "absent".
    """
    value = getattr(spec, attr, None)
    if value is None:
        inner_specs = getattr(spec, "kv_cache_specs", None)
        if inner_specs:
            first_inner = next(iter(inner_specs.values()))
            value = getattr(first_inner, attr, None)
    return int(value or 0)


def _per_layer_spec_attr_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    attr: str,
) -> list[int] | None:
    """Map an integer ``KVCacheGroupSpec.kv_cache_spec`` attribute onto layers.

    Reads ``attr`` off each engine group's spec (see :func:`spec_int_attr`) and
    assigns it to every registered layer in that group, indexed by registration
    order.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig``. ``None`` or one without
            ``kv_cache_groups`` yields ``None``.
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order, providing the layer-name -> index mapping.
        attr: The integer spec attribute to read (e.g. ``"block_size"`` or
            ``"sliding_window"``).

    Returns:
        A list of length ``len(kv_caches)`` giving each layer's value, or
        ``None`` when no group reports a positive value (callers then omit the
        hint and the server keeps its legacy/scalar fallback).
    """
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    if not vllm_groups:
        return None

    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    per_layer = [0] * len(layer_to_idx)
    any_positive = False
    for group in vllm_groups:
        value = spec_int_attr(getattr(group, "kv_cache_spec", None), attr)
        if value > 0:
            any_positive = True
        for name in group.layer_names:
            idx = layer_to_idx.get(name)
            if idx is not None:
                per_layer[idx] = value

    return per_layer if any_positive else None


def per_layer_sliding_window_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
) -> list[int] | None:
    """Build the per-registered-layer sliding-window list from vLLM metadata.

    Reads ``sliding_window`` off each engine group's spec and maps it onto every
    registered layer in that group. Enables the SWA-suffix-only optimization:
    layers in a sliding-window group only need their trailing window
    stored/retrieved. Layers whose group has no window get ``0``; an all-full-
    attention model yields ``None`` (no SWA metadata to send).
    """
    return _per_layer_spec_attr_from_vllm(
        kv_cache_config, kv_caches, "sliding_window"
    )


def per_layer_inference_engine_logical_block_size_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
) -> list[int] | None:
    """Build the per-registered-layer logical block size list from vLLM metadata.

    Reads ``block_size`` off each engine group's spec and maps it onto every
    registered layer in that group. Under the hybrid KV cache manager these
    differ across groups (e.g. DeepSeek-V4: 256 for the full-attention MLA
    group, 64/4/8 for the SWA / compressor-state groups) while
    ``cache_config.block_size`` collapses to their GCD; the server uses this
    list to derive each LMCache group's ``compress_ratio`` /
    ``physical_chunk_size`` per group. ``None`` when no group reports a block
    size (server falls back to the scalar ``inference_engine_logical_block_size``
    for every group).
    """
    return _per_layer_spec_attr_from_vllm(kv_cache_config, kv_caches, "block_size")
