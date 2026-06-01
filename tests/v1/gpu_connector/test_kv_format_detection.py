# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-engine format detection split.

``detect_format`` normalizes a raw ``kv_caches`` and discovers its
``EngineKVFormat``. These cases feed canonical (already-normalized)
structures plus the engine and layout hints, and assert the detected
format -- covering the vLLM NHD/HND branch, the SGLang depth-1/depth-2
branch, and the TRT-LLM cross-layer branch.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
import lmcache.c_ops as lmc_ops

NB, NL, BS, NH, HS = 7, 5, 3, 2, 4
DT = torch.float16
F = lmc_ops.EngineKVFormat


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def test_vllm_cross_layer():
    kv = _t(NB, NL, 2, BS, NH, HS)
    fmt, out = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NB_NL_TWO_BS_NH_HS
    assert out is kv


def test_vllm_flash_attn_nhd_vs_hnd():
    kv = [_t(2, NB, BS, NH, HS) for _ in range(NL)]
    fmt_nhd, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt_nhd == F.NL_X_TWO_NB_BS_NH_HS
    # HND geometry has heads before block-size: [2, NB, NH, BS, HS].
    kv_hnd = [_t(2, NB, NH, BS, HS) for _ in range(NL)]
    fmt_hnd, _ = detect_format(kv_hnd, EngineType.VLLM, {"kv_layout": "HND"})
    assert fmt_hnd == F.NL_X_TWO_NB_NH_BS_HS


def test_vllm_flash_infer_nhd():
    kv = [_t(NB, 2, BS, NH, HS) for _ in range(NL)]
    fmt, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NL_X_NB_TWO_BS_NH_HS


def test_vllm_mla():
    kv = [_t(NB, BS, HS) for _ in range(NL)]
    fmt, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NL_X_NB_BS_HS


def test_sglang_mla_depth1():
    kv = [_t(NB * BS, 1, HS) for _ in range(NL)]
    fmt, _ = detect_format(kv, EngineType.SGLANG, {})
    assert fmt == F.NL_X_NBBS_ONE_HS


def test_sglang_mha_depth2_fused():
    kv = [[_t(NB * BS, NH, HS) for _ in range(NL)] for _ in range(2)]
    fmt, _ = detect_format(kv, EngineType.SGLANG, {})
    assert fmt == F.TWO_X_NL_X_NBBS_NH_HS


def test_sglang_mha_mp_reshape():
    # MP path: flat list of 2*NL 3-D tensors + tokens_per_block hint;
    # detection should un-flatten + reshape to the 4-D inner MP format.
    if not hasattr(F, "TWO_X_NL_X_NB_BS_NH_HS"):
        pytest.skip("extension lacks TWO_X_NL_X_NB_BS_NH_HS")
    flat = [_t(NB * BS, NH, HS) for _ in range(2 * NL)]
    fmt, out = detect_format(flat, EngineType.SGLANG, {"tokens_per_block": BS})
    assert fmt == F.TWO_X_NL_X_NB_BS_NH_HS
    # Canonical depth-2 [K_layers, V_layers], inner reshaped to 4-D.
    assert len(out) == 2 and len(out[0]) == NL
    assert tuple(out[0][0].shape) == (NB, BS, NH, HS)


def test_trtllm_cross_layer_6d():
    kv = _t(NB, NL, 2, NH, BS, HS)
    fmt, _ = detect_format(kv, EngineType.TRTLLM, {})
    assert fmt == F.NB_NL_TWO_NH_BS_HS


def test_unsupported_structure_raises():
    # vLLM depth-1 list of 4-D tensors matches no branch (needs 5-D or 3-D).
    kv = [_t(NB, BS, NH, HS) for _ in range(NL)]
    with pytest.raises(ValueError):
        detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
