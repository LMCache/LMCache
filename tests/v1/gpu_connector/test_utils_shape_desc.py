# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the format-aware per-layer helpers in
:mod:`lmcache.v1.gpu_connector.utils`.
"""

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PageBufferShapeDesc and GPUKVFormat require the CUDA build",
)

# First Party
from lmcache.v1.gpu_connector.utils import (  # noqa: E402
    get_layer_data_ptrs,
    get_layer_dtype,
    get_layer_kv_caches,
    get_layer_shape_signature,
    make_page_buffer_shape_desc,
)
import lmcache.c_ops as lmc_ops  # noqa: E402


def test_make_shape_desc_vllm_flash_attn_nhd():
    kv_caches = [torch.empty(2, 32, 16, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 2
    assert sd.nl == 4
    assert sd.nb == 32
    assert sd.bs == 16
    assert sd.nh == 8
    assert sd.hs == 64
    assert sd.element_size == 2


def test_make_shape_desc_vllm_flash_infer_nhd():
    kv_caches = [torch.empty(32, 2, 16, 8, 64, dtype=torch.float16) for _ in range(2)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=32,
        block_size=16,
    )
    assert sd.nh == 8
    assert sd.hs == 64
    assert sd.kv_size == 2


def test_make_shape_desc_vllm_mla():
    kv_caches = [torch.empty(32, 16, 512, dtype=torch.bfloat16) for _ in range(3)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        layer_idx=0,
        num_layers_in_group=3,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 1
    assert sd.nh == 1
    assert sd.hs == 512


def test_make_shape_desc_sglang_mla():
    kv_caches = [torch.empty(512, 1, 128, dtype=torch.bfloat16) for _ in range(2)]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 1
    assert sd.nh == 1
    assert sd.hs == 128


def test_make_shape_desc_sglang_mha():
    k = [torch.empty(512, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    v = [torch.empty(512, 8, 64, dtype=torch.bfloat16) for _ in range(4)]
    kv_caches = [k, v]
    sd = make_page_buffer_shape_desc(
        kv_caches,
        lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        layer_idx=0,
        num_layers_in_group=4,
        num_blocks=32,
        block_size=16,
    )
    assert sd.kv_size == 2
    assert sd.nh == 8
    assert sd.hs == 64


def test_get_layer_helpers_per_layer_list():
    kv_caches = [
        torch.randn(2, 32, 16, 8, 64, dtype=torch.float16, device="cuda")
        for _ in range(3)
    ]
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

    rep = get_layer_kv_caches(kv_caches, fmt, layer_idx=1)
    assert isinstance(rep, list) and len(rep) == 1
    assert rep[0].data_ptr() == kv_caches[1].data_ptr()

    ptrs = get_layer_data_ptrs(kv_caches, fmt, layer_idx=2)
    assert ptrs == [kv_caches[2].data_ptr()]

    assert get_layer_dtype(kv_caches, fmt, layer_idx=0) == torch.float16


def test_get_layer_helpers_sglang_mha():
    k = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    v = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    kv_caches = [k, v]
    fmt = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS

    rep = get_layer_kv_caches(kv_caches, fmt, layer_idx=1)
    assert rep[0][0].data_ptr() == k[1].data_ptr()
    assert rep[1][0].data_ptr() == v[1].data_ptr()

    ptrs = get_layer_data_ptrs(kv_caches, fmt, layer_idx=0)
    assert ptrs == [k[0].data_ptr(), v[0].data_ptr()]


def test_shape_signature_equal_for_same_shape():
    kv_caches = [torch.empty(2, 32, 16, 8, 64, dtype=torch.bfloat16) for _ in range(3)]
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    sig0 = get_layer_shape_signature(kv_caches, fmt, layer_idx=0)
    sig1 = get_layer_shape_signature(kv_caches, fmt, layer_idx=1)
    assert sig0 == sig1
    assert sig0 == (2, 8, 64)


def test_shape_signature_distinguishes_num_heads():
    kv_caches = [
        torch.empty(2, 32, 16, 8, 64, dtype=torch.bfloat16),
        torch.empty(2, 32, 16, 16, 64, dtype=torch.bfloat16),
    ]
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    sig0 = get_layer_shape_signature(kv_caches, fmt, layer_idx=0)
    sig1 = get_layer_shape_signature(kv_caches, fmt, layer_idx=1)
    assert sig0 != sig1
    assert sig0[1] == 8 and sig1[1] == 16


def test_shape_signature_mla_forces_nh_one():
    kv_caches = [torch.empty(32, 16, 512, dtype=torch.bfloat16) for _ in range(2)]
    fmt = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    sig = get_layer_shape_signature(kv_caches, fmt, layer_idx=0)
    assert sig == (1, 1, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
