# SPDX-License-Identifier: Apache-2.0
"""Build LMCache group views from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.v1.multiprocess.group_view import LMCacheGroupView


def spec_int_attr(spec: Any, attr: str) -> int:
    """Read an integer attribute from a vLLM KV cache spec.

    For ``UniformTypeKVCacheSpec``, the attribute lives on the inner ``kv_type``
    spec; for all others it lives directly on *spec*.

    Args:
        spec: A vLLM ``KVCacheSpec`` (or ``None``).
        attr: Attribute name to read.

    Returns:
        Integer value, or ``0`` when *spec* is ``None`` or the attribute is absent.
    """
    if spec is None:
        return 0
    inner = getattr(spec, "kv_type", None)
    src = inner if inner is not None else spec
    return int(getattr(src, attr, 0) or 0)


def storage_blocks_per_chunk(
    sliding_window: int,
    logical_block_size: int,
    chunk_tokens: int,
) -> int:
    """Blocks per LMCache chunk for a KV group.

    Full-attention groups (``sliding_window == 0``) return the full chunk block
    count. SWA groups return only the trailing blocks that fall inside the
    window, capped at the full count.

    Args:
        sliding_window: Sliding-window size in tokens (0 = full attention).
        logical_block_size: Inference-engine-side logical block size in tokens.
        chunk_tokens: LMCache logical chunk size in tokens.

    Returns:
        Number of blocks per chunk to store/retrieve for this group.
    """
    full_bpc = chunk_tokens // logical_block_size
    if sliding_window == 0:
        return full_bpc
    suffix_bpc = math.ceil(sliding_window / logical_block_size)
    return min(suffix_bpc, full_bpc)


def engine_group_sbpc_from_vllm(
    kv_cache_config: Any,
    chunk_tokens: int,
) -> "dict[int, int]":
    """Compute per-engine-group ``storage_blocks_per_chunk`` from vLLM config.

    Reads ``block_size`` and ``sliding_window`` from each engine group's
    ``kv_cache_spec`` and delegates to :func:`storage_blocks_per_chunk`.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (or ``None``).
        chunk_tokens: LMCache logical chunk size in tokens. ``0`` or missing
            config yields an empty mapping.

    Returns:
        Mapping from engine group id to blocks-per-chunk for that group.
        Empty dict when *kv_cache_config* has no groups or *chunk_tokens* is 0.
    """
    if not chunk_tokens or kv_cache_config is None:
        return {}
    vllm_groups = getattr(kv_cache_config, "kv_cache_groups", ()) or ()
    result: dict[int, int] = {}
    for engine_group_id, group in enumerate(vllm_groups):
        spec = getattr(group, "kv_cache_spec", None)
        logical_bs = spec_int_attr(spec, "block_size")
        sliding_win = spec_int_attr(spec, "sliding_window")
        if logical_bs > 0:
            result[engine_group_id] = storage_blocks_per_chunk(
                sliding_win, logical_bs, chunk_tokens
            )
    return result


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

    For per-group geometry hints (e.g. ``storage_blocks_per_chunk``) use
    :func:`engine_group_sbpc_from_vllm` and embed the result in
    :class:`~lmcache.v1.gpu_connector.utils.LayoutHints` under
    ``per_engine_group_storage_blocks_per_chunk`` before passing to the server.

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
