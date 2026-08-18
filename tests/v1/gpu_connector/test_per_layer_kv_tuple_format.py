# SPDX-License-Identifier: Apache-2.0
"""Per-layer ``(K, V)`` tuple KV layout (GPUKVFormat.NL_X_TWO_X_NB_BS_NH_HS).

vLLM-Ascend registers its KV cache as a per-layer list where each layer carries
its own ``(K, V)`` pair of paged ``[num_blocks, block_size, num_heads,
head_size]`` tensors, rather than the cross-layer K/V-major ``[2, NL, ...]``
layout. Discovery keeps that native nested structure intact and classifies it as
``NL_X_TWO_X_NB_BS_NH_HS``.

These tests pin discovery, the format-aware accessors, the interleaved kernel
pointer order, the layer-grouping shape, and the multiprocess layout detection
for that format.
"""

# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout
import lmcache.lmcache_native as lmc_ops

NB, NL, BS, NH, HS = 8, 3, 4, 2, 8
F = lmc_ops.EngineKVFormat


def _per_layer_kv_tuple_caches() -> list[list[torch.Tensor]]:
    """Per-layer ``(K, V)`` tuples as vLLM-Ascend registers them.

    Layers are outermost; each layer is a length-2 ``[K, V]`` pair of paged
    ``[NB, BS, NH, HS]`` tensors.
    """
    torch.manual_seed(0)
    return [[torch.randn(NB, BS, NH, HS) for _ in range(2)] for _ in range(NL)]


def test_discovery_preserves_per_layer_kv_tuple_structure() -> None:
    kv = _per_layer_kv_tuple_caches()
    fmt, normalized = U.normalize_kv_and_discover_format(
        cast(DiscoverableKVCache, kv), EngineType.VLLM
    )
    assert fmt == F.NL_X_TWO_X_NB_BS_NH_HS
    # Native structure preserved: layers outermost, each a 2-element (K, V) pair
    # (normalize may rebuild the containers, so compare structure, not identity).
    assert len(normalized) == NL
    assert all(len(layer) == 2 for layer in normalized)


def test_accessors() -> None:
    fmt, norm = U.normalize_kv_and_discover_format(
        cast(DiscoverableKVCache, _per_layer_kv_tuple_caches()), EngineType.VLLM
    )
    assert fmt == F.NL_X_TWO_X_NB_BS_NH_HS
    assert U.get_num_layers(norm, fmt) == NL
    assert U.get_num_blocks(norm, fmt) == NB
    assert U.get_block_size(norm, fmt) == BS
    assert U.get_num_heads(norm, fmt) == NH
    assert U.get_head_size(norm, fmt) == HS
    assert U.get_hidden_dim_size(norm, fmt) == NH * HS
    assert U.get_page_buffer_size(norm, fmt) == NB * BS
    assert U.get_tokens_per_layer(norm, fmt) == NB * BS
    assert U.get_elements_per_layer(norm, fmt) == NB * BS * NH * HS * 2
    assert U.get_dtype(norm, fmt) == torch.float32
    assert not lmc_ops.is_mla(fmt)


def test_group_data_ptrs_interleaved_kv_order() -> None:
    kv = _per_layer_kv_tuple_caches()
    fmt, normalized = U.normalize_kv_and_discover_format(
        cast(DiscoverableKVCache, kv), EngineType.VLLM
    )
    # Interleaved [k_i, v_i, ...] pointer order the transfer kernel expects.
    ptrs = U.get_group_data_ptrs(normalized, fmt, [0, 2])
    assert ptrs == [
        kv[0][0].data_ptr(),
        kv[0][1].data_ptr(),
        kv[2][0].data_ptr(),
        kv[2][1].data_ptr(),
    ]


def test_layer_grouping_shape_desc() -> None:
    kv = _per_layer_kv_tuple_caches()
    fmt, normalized = U.normalize_kv_and_discover_format(
        cast(DiscoverableKVCache, kv), EngineType.VLLM
    )
    mgr = KVLayerGroupsManager(normalized, engine_kv_formats=[fmt] * NL)
    groups = mgr.kernel_groups
    assert len(groups) == 1
    sd = groups[0].shape_desc
    assert (sd.kv_size, sd.nl, sd.nb, sd.bs, sd.nh, sd.hs) == (
        2,
        NL,
        NB,
        BS,
        NH,
        HS,
    )


def test_compute_kv_layout_detects_per_layer_tuple() -> None:
    raw = _per_layer_kv_tuple_caches()
    src = {f"layer_{i}": (k, v) for i, (k, v) in enumerate(raw)}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
        kv_size,
    ) = compute_kv_layout(cast("dict[str, torch.Tensor]", src), layout_hints=None)
    assert block_size == BS
    assert num_layers == NL
    assert hidden_dim == NH * HS
    assert dtype_str == "float32"
    assert detected_kv_format == F.NL_X_TWO_X_NB_BS_NH_HS
    assert kv_size == 2
