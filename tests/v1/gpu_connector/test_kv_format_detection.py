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
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format, extract_kv_cache_shapes
import lmcache.lmcache_native as lmcache_native

NB, NL, BS, NH, HS = 7, 5, 3, 2, 4
DT = torch.float16
F = lmcache_native.EngineKVFormat


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def test_vllm_cross_layer():
    kv = _t(NB, NL, 2, BS, NH, HS)
    fmt, out = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NB_NL_TWO_BS_NH_HS
    assert tuple(out.shape) == tuple(kv.shape)
    assert out.data_ptr() == kv.data_ptr()


# The CPU-HND safeguard forces HND regardless of hint when running on a CPU
# host; bypass it so the hint-driven NHD/HND branch is exercised on any host.
_VLLM_DEV = "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type"
_MOCK_DEVICE_TYPE = "cuda" if torch_device_type == "cpu" else torch_device_type


def test_vllm_flash_attn_hnd(monkeypatch):
    monkeypatch.setattr(_VLLM_DEV, _MOCK_DEVICE_TYPE)
    kv = [_t(2, NB, BS, NH, HS) for _ in range(NL)]
    fmt_hnd, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "HND"})
    assert fmt_hnd == F.NL_X_TWO_NB_NH_BS_HS


def test_vllm_flash_infer_nhd(monkeypatch):
    monkeypatch.setattr(_VLLM_DEV, _MOCK_DEVICE_TYPE)
    kv = [_t(NB, 2, BS, NH, HS) for _ in range(NL)]
    fmt, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NL_X_NB_TWO_BS_NH_HS


def test_vllm_mla():
    kv = [_t(NB, BS, HS) for _ in range(NL)]
    fmt, _ = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})
    assert fmt == F.NL_X_NB_BS_HS


@pytest.mark.parametrize("hint", [{}, {"kv_layout": "NHD"}, {"kv_layout": "HND"}])
def test_vllm_rbln_native_singleton_axis(monkeypatch, hint):
    # vLLM-RBLN's 6-D HND layout is its own format, so it is detected from the
    # shape alone on any host and whatever the hint says (vLLM-RBLN reports no
    # layout, and the format is HND by definition).
    monkeypatch.setattr(_VLLM_DEV, "cuda")
    kv = [_t(2, NB, NH, 1, BS, HS) for _ in range(NL)]
    fmt, out = detect_format(kv, EngineType.VLLM, hint)
    assert fmt == F.NL_X_TWO_NB_NH_ONE_BS_HS
    # Detection must not reshape it away: the singleton axis survives.
    assert tuple(out[0].shape) == (2, NB, NH, 1, BS, HS)
    assert out[0].data_ptr() == kv[0].data_ptr()


def test_vllm_blocks_first_fused_num_heads_2(monkeypatch):
    # Raw 4-D [NB, NH, BS, 2*HS] with NH == 2 (a common GQA config): a 5-D
    # split would make the K/V axis and the head axis both equal 2, ambiguous
    # with flash-infer. Detection must use the rank-4 shape to land on the
    # content-size format, and keep the tensor raw.
    monkeypatch.setattr(_VLLM_DEV, _MOCK_DEVICE_TYPE)
    raw = [_t(NB, 2, BS, 2 * HS) for _ in range(NL)]
    fmt, out = detect_format(raw, EngineType.VLLM, {"kv_layout": "HND"})
    assert fmt == F.NL_X_NB_NH_BS_CS
    assert tuple(out[0].shape) == (NB, 2, BS, 2 * HS)


def test_vllm_blocks_first_fused_hnd(monkeypatch):
    monkeypatch.setattr(_VLLM_DEV, _MOCK_DEVICE_TYPE)
    raw_hnd = [_t(NB, NH, BS, 2 * HS) for _ in range(NL)]
    fmt_hnd, _ = detect_format(raw_hnd, EngineType.VLLM, {"kv_layout": "HND"})
    assert fmt_hnd == F.NL_X_NB_NH_BS_CS


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
    # vLLM depth-1 list of 2-D tensors matches no branch (needs 5-D, 4-D, or
    # 3-D). (4-D is now the blocks-first fused layout, so it no longer raises.)
    kv = [_t(NB, HS) for _ in range(NL)]
    with pytest.raises(ValueError):
        detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})


def test_extract_kv_cache_shapes_single_tensor():
    assert extract_kv_cache_shapes(_t(NB, BS, HS)) == {(NB, BS, HS)}


def test_extract_kv_cache_shapes_uniform_list_dedups():
    kv = [_t(2, NB, BS, NH, HS) for _ in range(NL)]
    assert extract_kv_cache_shapes(kv) == {(2, NB, BS, NH, HS)}


def test_extract_kv_cache_shapes_mixed_nested():
    kv = [
        [_t(NB, BS, NH, HS), _t(NB, BS, HS)],
        [_t(NB, BS, NH, HS)],
    ]
    assert extract_kv_cache_shapes(kv) == {(NB, BS, NH, HS), (NB, BS, HS)}
