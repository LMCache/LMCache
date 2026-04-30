# SPDX-License-Identifier: Apache-2.0

"""Tests for Modular MAX GPU KV layout discovery and shape helpers."""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_block_size,
    get_concrete_gpu_kv_shape,
    get_dtype,
    get_group_data_ptrs,
    get_head_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    make_page_buffer_shape_desc,
    normalize_kv_and_discover_format,
)
import lmcache.c_ops as lmc_ops


def test_max_layout_detection_and_accessors():
    kv_cache = torch.empty(32, 2, 4, 16, 8, 64, dtype=torch.bfloat16)

    fmt, normalized = normalize_kv_and_discover_format(kv_cache, EngineType.MAX)

    assert fmt == lmc_ops.GPUKVFormat.NB_KV_NL_BS_NH_HS
    assert normalized is kv_cache
    assert get_num_blocks(kv_cache, fmt) == 32
    assert get_num_layers(kv_cache, fmt) == 4
    assert get_block_size(kv_cache, fmt) == 16
    assert get_num_heads(kv_cache, fmt) == 8
    assert get_head_size(kv_cache, fmt) == 64
    assert get_dtype(kv_cache, fmt) == torch.bfloat16
    assert get_concrete_gpu_kv_shape(kv_cache, fmt) == "[32, 2, 4, 16, 8, 64]"


def test_max_layout_shape_desc_uses_kv_dim():
    kv_cache = torch.empty(32, 2, 4, 16, 8, 64, dtype=torch.float16)
    fmt = lmc_ops.GPUKVFormat.NB_KV_NL_BS_NH_HS

    desc = make_page_buffer_shape_desc(
        kv_cache,
        fmt,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )

    assert desc.kv_size == 2
    assert desc.nl == 4
    assert desc.nb == 32
    assert desc.bs == 16
    assert desc.nh == 8
    assert desc.hs == 64
    assert desc.element_size == 2


def test_max_layout_mla_shape_desc_uses_single_kv_plane():
    kv_cache = torch.empty(32, 1, 4, 16, 1, 512, dtype=torch.bfloat16)
    fmt = lmc_ops.GPUKVFormat.NB_KV_NL_BS_NH_HS

    desc = make_page_buffer_shape_desc(
        kv_cache,
        fmt,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )

    assert desc.kv_size == 1
    assert desc.nh == 1
    assert desc.hs == 512


def test_max_layout_group_data_ptrs_returns_single_base():
    kv_cache = torch.empty(32, 2, 4, 16, 8, 64, dtype=torch.float16)
    fmt = lmc_ops.GPUKVFormat.NB_KV_NL_BS_NH_HS

    assert get_group_data_ptrs(kv_cache, fmt, [0, 1, 2, 3]) == [kv_cache.data_ptr()]
