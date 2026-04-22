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
    ensure_contiguous_kv_caches,
    get_device,
    get_group_data_ptrs,
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


def test_get_group_data_ptrs_per_layer_list_flattens_in_order():
    kv_caches = [
        torch.randn(2, 32, 16, 8, 64, dtype=torch.float16, device="cuda")
        for _ in range(4)
    ]
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

    ptrs = get_group_data_ptrs(kv_caches, fmt, [0, 2, 3])
    assert ptrs == [
        kv_caches[0].data_ptr(),
        kv_caches[2].data_ptr(),
        kv_caches[3].data_ptr(),
    ]


def test_get_group_data_ptrs_sglang_mha_groups_k_before_v():
    """SGLang MHA kernel contract: [K0, K1, ..., KN, V0, V1, ..., VN] —
    not per-layer [K0, V0, K1, V1, ...]."""
    k = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    v = [torch.randn(512, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    kv_caches = [k, v]
    fmt = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS

    ptrs = get_group_data_ptrs(kv_caches, fmt, [0, 1, 2])
    expected = [
        k[0].data_ptr(),
        k[1].data_ptr(),
        k[2].data_ptr(),
        v[0].data_ptr(),
        v[1].data_ptr(),
        v[2].data_ptr(),
    ]
    assert ptrs == expected


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


def test_get_group_data_ptrs_cross_layer_returns_single_base():
    """Cross-layer format packs every layer into one tensor; the kernel
    (csrc/mp_mem_kernels.cu) reads paged_buffer_ptrs[0] and computes
    per-layer offsets from shape_desc.nl internally. The group helper
    must return a single base pointer, not num_layers entries."""
    big = torch.empty(32, 80, 2, 16, 8, 64, dtype=torch.bfloat16, device="cuda")
    fmt = lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
    ptrs = get_group_data_ptrs(big, fmt, list(range(80)))
    assert ptrs == [big.data_ptr()]


def test_get_layer_data_ptrs_cross_layer_rejects():
    """No per-layer pointer exists for cross-layer; callers must use
    get_group_data_ptrs instead."""
    big = torch.empty(32, 80, 2, 16, 8, 64, dtype=torch.bfloat16, device="cuda")
    fmt = lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
    with pytest.raises(ValueError, match="cross-layer"):
        get_layer_data_ptrs(big, fmt, layer_idx=0)


def test_ensure_contiguous_preserves_bare_tensor():
    """A bare torch.Tensor input (cross-layer shape) must pass through
    ensure_contiguous_kv_caches unchanged — no list wrapping — so the
    DiscoverableKVCache recursive union is respected end-to-end."""
    big = torch.empty(32, 80, 2, 16, 8, 64, dtype=torch.bfloat16, device="cuda")
    out = ensure_contiguous_kv_caches(big)
    assert isinstance(out, torch.Tensor)
    assert out is big


def test_ensure_contiguous_recurses_all_shapes():
    """ensure_contiguous must descend into every DiscoverableKVCache
    shape and permute non-contiguous tensor leaves."""
    # Build a flash-attention HND-layout tensor (non-contiguous after
    # logical→physical permute exposure — the vLLM HND case).
    nhd_view = (
        torch.empty(2, 32, 8, 16, 64, dtype=torch.bfloat16, device="cuda")
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .permute(0, 1, 3, 2, 4)  # NHD logical view over HND physical
    )
    assert not nhd_view.is_contiguous()

    # bare tensor
    out_tensor = ensure_contiguous_kv_caches(nhd_view, kv_layout="HND")
    assert out_tensor.is_contiguous()

    # flat list
    lst = [nhd_view.clone() for _ in range(2)]
    out_list = ensure_contiguous_kv_caches(lst, kv_layout="HND")
    assert isinstance(out_list, list)
    assert all(t.is_contiguous() for t in out_list)

    # nested list (SGLang-shaped)
    k = [nhd_view.clone() for _ in range(2)]
    v = [nhd_view.clone() for _ in range(2)]
    out_nested = ensure_contiguous_kv_caches([k, v], kv_layout="HND")
    assert isinstance(out_nested, list)
    assert all(t.is_contiguous() for sublist in out_nested for t in sublist)


def test_get_device_handles_every_kvcaches_shape():
    """get_device must work for every DiscoverableKVCache shape without format hints."""
    t = torch.empty(8, dtype=torch.bfloat16, device="cuda")
    assert get_device(t) == t.device

    flat = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    assert get_device(flat) == flat[0].device

    k = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    v = [torch.empty(4, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    assert get_device([k, v]) == k[0].device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
