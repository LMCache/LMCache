# SPDX-License-Identifier: Apache-2.0
"""Blocks-first (BLHNC/BLNHC) declaration-driven detection tests.

Builds per-layer strided views into one buffer exactly the way vLLM
registers them under a blocks-first layout, and checks that detection
classifies them as the existing per-layer CS formats with the views
passed through unchanged -- the block step travels in stride(0).
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.utils import resolve_block_stride_and_log_layout
import lmcache.lmcache_native as lmcache_native

NB, NL, NH, BS, CS = 6, 3, 2, 4, 8


def make_views(order: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    inner = (NH, BS, CS) if order == "BLHNC" else (BS, NH, CS)
    buf = torch.arange(NB * NL * NH * BS * CS, dtype=torch.float32)
    buf = buf.reshape(NB, NL, *inner)
    return buf, [buf[:, layer] for layer in range(NL)]


@pytest.mark.parametrize(
    "order,expected",
    [
        ("BLHNC", lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS),
        ("BLNHC", lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS),
    ],
)
def test_detects_per_layer_format_with_views_unchanged(order, expected):
    buf, views = make_views(order)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": order})
    assert fmt == expected
    assert isinstance(kv, list)
    # Contiguity recovery may re-view; addressing must be untouched.
    for k, v in zip(kv, views, strict=True):
        assert k.data_ptr() == v.data_ptr()
        assert k.shape == v.shape and k.stride() == v.stride()
    assert kv[0].stride(0) == buf.stride(0)


def test_block_stride_is_read_from_the_views():
    buf, views = make_views("BLHNC")
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    stride = resolve_block_stride_and_log_layout(kv, fmt, layer_idx=0, group_idx=0)
    assert stride == buf.stride(0) == NL * NH * BS * CS


def test_compact_views_declared_blocks_first_still_classify():
    """A tight per-layer cache under a blocks-first declaration is
    addressed identically: stride(0) is simply the tight step."""
    views = [torch.zeros(NB, NH, BS, CS) for _ in range(NL)]
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    stride = resolve_block_stride_and_log_layout(kv, fmt, layer_idx=0, group_idx=0)
    assert stride == NH * BS * CS


def test_interleaved_groups_detect():
    """Two groups woven inside each block: each view's stride(0) spans both
    groups' bytes; no cross-view relationship is required."""
    inner = (NH, BS, CS)
    buf = torch.arange(NB * 2 * NL * NH * BS * CS, dtype=torch.float32)
    buf = buf.reshape(NB, 2 * NL, *inner)
    views = [buf[:, 2 * layer] for layer in range(NL)]  # even slots = ours
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    assert kv[0].data_ptr() == views[0].data_ptr()
    assert kv[0].stride(0) == 2 * NL * NH * BS * CS


def test_non_tight_inner_dims_rejected():
    """Inner gaps are rejected by contiguous-view recovery, which accepts
    dim-0 padding only."""
    buf, _ = make_views("BLHNC")
    loose = [
        buf.as_strided((NB, NH, BS // 2, CS), (buf.stride(0), BS * CS, 2 * CS, 1))
        for _ in range(NL)
    ]
    with pytest.raises(ValueError, match="dim-0 padding only"):
        detect_format(loose, EngineType.VLLM, {"kv_layout": "BLHNC"})
