# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its sliding window size in tokens, or ``-1`` for
        non-sliding-window layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        # ``UniformTypeKVCacheSpecs`` carries per-layer specs in
        # ``kv_cache_specs``; other specs apply to all of the group's layers.
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
    return per_layer_sw_size


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def _cp_token_split_factor(vllm_config: Any) -> int:
    """Token-split factor of one scheduler block under context parallelism.

    Mirrors vLLM's ``cp_token_split_factor``: under (uneven) DCP/PCP one
    "virtual" scheduler block of a token-split group spans this many
    ``block_size`` units (``dcp * pcp`` for the even split, ``sum(ratios)``
    under uneven DCP). Returns 1 when context parallelism is off or the
    parallel config is unavailable.

    Prefers vLLM's own helper (authoritative: it reads the installed CP
    token vectors, which under uneven DCP v4 need not equal
    ``--rank-tp-ratio``); falls back to a config-derived approximation for
    vLLM builds without the helper.
    """
    pc = getattr(vllm_config, "parallel_config", None)
    dcp = getattr(pc, "decode_context_parallel_size", 1) or 1
    pcp = getattr(pc, "prefill_context_parallel_size", 1) or 1
    if dcp * pcp <= 1:
        return 1
    try:
        # Third Party
        from vllm.distributed.utils import cp_token_split_factor

        return int(cp_token_split_factor(dcp, pcp))
    except (ImportError, AttributeError):
        ratios = getattr(pc, "rank_tp_ratio", None)
        if ratios and dcp > 1 and len(ratios) == dcp and len(set(ratios)) > 1:
            return int(sum(ratios)) * pcp
        return dcp * pcp


def _is_token_split_spec(spec: Any) -> bool:
    """True if this group's KV is split along the token axis under CP.

    Mamba / linear-attention groups keep the FULL per-sequence state on
    every rank (no token-axis split), so their allocated block IDs always
    span the raw ``block_size``. Duck-typed on the spec class name to
    avoid importing vLLM here; mirrors vLLM's
    ``isinstance(spec, MambaSpec)`` checks (kv_cache_utils, coordinator,
    gpu_model_runner all special-case exactly MambaSpec).
    """
    return "mamba" not in type(spec).__name__.lower()


def resolve_group_token_spans(
    kv_cache_config: Any,
    vllm_config: Any,
) -> list[int]:
    """Tokens of the GLOBAL sequence covered by ONE allocated block ID,
    per engine group.

    This is the grain of the block IDs vLLM's scheduler hands to the
    connector, and therefore the ONLY correct ``tokens_per_block`` for
    slicing token ranges into per-group block IDs and for deriving
    blocks-per-chunk on the worker. It mirrors vLLM's
    ``KVCacheCoordinator._group_token_span``:

    - Mamba groups: raw ``spec.block_size`` (full per-sequence state on
      every rank; the align solver may have inflated this).
    - Token-split (attention) groups: ``spec.block_size`` when context
      parallelism is off; ``spec.block_size * cp_token_split_factor``
      under (uneven) DCP/PCP -- the scheduler manages these groups in
      "virtual" blocks and the block IDs it reports are virtual.

    Using the raw ``spec.block_size`` for token-split groups under CP
    (the previous behaviour) mis-slices block IDs and mis-computes
    blocks-per-chunk: stores silently write mis-keyed chunk data and
    retrieves crash in scatter with a block-count mismatch.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (or ``None`` -> single
            non-hybrid group from ``cache_config.block_size``).
        vllm_config: vLLM ``VllmConfig`` used for the parallel geometry.

    Returns:
        One token span per engine group, in engine-group order.
    """
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    factor = _cp_token_split_factor(vllm_config)
    if not vllm_groups:
        return [vllm_config.cache_config.block_size * factor]
    return [
        group.kv_cache_spec.block_size
        * (factor if _is_token_split_spec(group.kv_cache_spec) else 1)
        for group in vllm_groups
    ]


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
    vllm_config: Any = None,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

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
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        get_num_layers,
        normalize_kv_and_discover_format,
    )
    from lmcache.v1.kv_layer_groups import (
        EXCLUDED_ENGINE_GROUP,
        group_layers_by_identity,
    )

    # Inspect the real registered tensors for physical layout and dtype.
    engine_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
        list(kv_caches.values()),
        EngineType.VLLM,
        layout_hints=layout_hints,
    )
    num_layers = get_num_layers(normalized_kv_caches, engine_kv_format)

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
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_layer_sw_size = [-1] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        # Tokens of the global sequence covered by one allocated block ID
        # of each group. When the vllm_config is available this is the
        # CP-aware scheduler grain (virtual blocks for token-split groups
        # under DCP/PCP); the raw spec block_size is only correct without
        # context parallelism. The physical slot count per chunk is
        # discovered later from the registered tensors.
        if vllm_config is not None:
            group_spans = resolve_group_token_spans(kv_cache_config, vllm_config)
        else:
            group_spans = [g.kv_cache_spec.block_size for g in vllm_groups]
        for engine_group_id, group in enumerate(vllm_groups):
            group_tokens_per_block[engine_group_id] = group_spans[engine_group_id]
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size, dtype), so
    # every resulting LMCache group can be served by a single copy kernel. It is
    # the shared, engine-neutral primitive the server reuses to reproduce the
    # same grouping from the registered tensors.
    return [
        EngineGroupInfo(
            engine_group_id=identity[4],
            layer_indices=tuple(indices),
            tokens_per_block=group_tokens_per_block.get(identity[4], 0),
            sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            engine_kv_format,
            num_layers,
            per_layer_group_idx,
        )
    ]
