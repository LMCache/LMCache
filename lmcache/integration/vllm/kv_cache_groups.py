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
    """Read an integer ``attr`` off a vLLM KVCacheGroupSpec spec.

    Falls back to the first inner spec for ``UniformTypeKVCacheSpecs`` wrappers.
    Returns ``0`` when the attribute is absent or ``None``.
    """
    value = getattr(spec, attr, None)
    if value is None:
        inner_specs = getattr(spec, "kv_cache_specs", None)
        if inner_specs:
            first_inner = next(iter(inner_specs.values()))
            value = getattr(first_inner, attr, None)
    return int(value or 0)


def storage_blocks_per_chunk(
    sliding_window: int,
    logical_block_size: int,
    chunk_tokens: int,
) -> int:
    """Blocks per chunk a group actually stores/retrieves (SWA-suffix-aware).

    Returns the full chunk for full-attention groups (``sliding_window <= 0``),
    or ``ceil(sliding_window / logical_block_size)`` capped at the full chunk
    for sliding-window groups. Shared by the connector's scheduler-side trim
    and its worker-side register hint so the two cannot drift.
    """
    full_bpc = chunk_tokens // logical_block_size
    if sliding_window <= 0:
        return full_bpc
    suffix = (sliding_window + logical_block_size - 1) // logical_block_size
    return min(suffix, full_bpc)


def per_layer_storage_blocks_per_chunk_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    chunk_tokens: int,
) -> list[int] | None:
    """Per-registered-layer stored-blocks-per-chunk derived from vLLM metadata.

    Combines each group's ``block_size`` and ``sliding_window`` via
    :func:`storage_blocks_per_chunk` and maps the result onto every layer in
    that group. Returns ``None`` when no group reports a positive block size.
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
    any_block_size = False
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        logical_bs = spec_int_attr(spec, "block_size")
        if logical_bs <= 0:
            continue
        any_block_size = True
        window = spec_int_attr(spec, "sliding_window")
        bpc = storage_blocks_per_chunk(window, logical_bs, chunk_tokens)
        for name in group.layer_names:
            idx = layer_to_idx.get(name)
            if idx is not None:
                per_layer[idx] = bpc

    return per_layer if any_block_size else None
