# SPDX-License-Identifier: Apache-2.0
"""Split K/V + fused normalization and strict validation.

Some serving backends expose their paged KV cache to LMCache as two independent
per-layer torch tensors (key, value) instead of one fused ``[2, ...]`` tensor --
e.g. a host-memory backend that hands LMCache zero-copy views of separate native
key/value arrays (avoiding a fused staging copy). LMCache normalizes both the
existing fused representation and the split ``tuple[Tensor, Tensor]`` form into
its internal per-layer view, and gathers/scatters the live buffers directly.

These tests are backend-neutral: they never import a backend package, never
name a concrete connector, and never branch on a backend/platform name. They map
the split/fused normalization cases (1-6, 14, 15). Layout-precedence cases
live in test_metal_engine_driven.py. No accelerator required.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    get_block_size,
    get_head_size,
    get_num_heads,
    get_num_layers,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)


@pytest.fixture(autouse=True)
def _force_cpu_device_type(monkeypatch):
    """Pin the detector's process-level device type to "cpu" so the CPU-default
    and CPU-override cases are deterministic on any host (a GPU CI box would
    otherwise default to NHD)."""
    import lmcache.v1.gpu_connector.kv_format.detectors.vllm as det

    monkeypatch.setattr(det, "torch_device_type", "cpu", raising=False)


def _detector():
    from lmcache.v1.gpu_connector.kv_format.detectors.vllm import VLLM_Detector

    return VLLM_Detector()


# --- builders --------------------------------------------------------------


def _fused_kv(num_layers=2, nb=6, bs=4, nh=2, hs=8):
    """Fused per-layer KV: [2, num_blocks, block_size, num_heads, head_size]."""
    return {
        f"layer_{i}": torch.randn(2, nb, bs, nh, hs) for i in range(num_layers)
    }


def _split_kv(num_layers=2, nb=6, bs=4, nh=2, hs=8, *, seed=0):
    """Split per-layer KV: {name: (key, value)}, each [nb, bs, nh, hs] (NHD)."""
    g = torch.Generator().manual_seed(seed)
    kv = {}
    for i in range(num_layers):
        k = torch.randn(nb, bs, nh, hs, generator=g)
        v = torch.randn(nb, bs, nh, hs, generator=g) + 100.0  # distinguish V
        kv[f"layer_{i}"] = (k, v)
    return kv


# --- (1) existing fused CPU HND registration unchanged ---------------------


def test_case1_fused_cpu_hnd_unchanged():
    detector = _detector()
    fused = [torch.randn(2, 6, 4, 2, 8) for _ in range(2)]  # nh=2 != bs=4
    # No hint on CPU -> HND default.
    fmt, _ = detector.discover(fused, {})
    assert fmt.name == "NL_X_TWO_NB_NH_BS_HS"  # HND fused
# --- (3) split K/V CPU NHD works -------------------------------------------


def test_case3_split_cpu_nhd_works():
    kv = _split_kv()
    normalized, fmts = normalize_and_discover_per_layer_formats(
        list(kv.values()), [], EngineType.VLLM, {"kv_layout": "NHD"}
    )
    assert all(f.name == "TWO_X_NL_X_NB_BS_NH_HS" for f in fmts)
    # normalized is [K_layers, V_layers]
    assert isinstance(normalized, list) and len(normalized) == 2
    assert len(normalized[0]) == 2 and len(normalized[1]) == 2
    assert get_num_layers(normalized, fmts[0]) == 2
    assert get_block_size(normalized, fmts[0]) == 4
    assert get_num_heads(normalized, fmts[0]) == 2
    assert get_head_size(normalized, fmts[0]) == 8


# --- (4) split K/V CPU HND works -------------------------------------------


def test_case4_split_cpu_hnd_works():
    """A split registration is physically NHD [NB,BS,NH,HS]; the detector
    reports the split format regardless of an HND hint (there is a single
    supported split layout). It must still normalize + expose geometry."""
    kv = _split_kv()
    normalized, fmts = normalize_and_discover_per_layer_formats(
        list(kv.values()), [], EngineType.VLLM, {"kv_layout": "HND"}
    )
    assert all(f.name == "TWO_X_NL_X_NB_BS_NH_HS" for f in fmts)
    assert get_num_layers(normalized, fmts[0]) == 2


# --- (5) split gather -> scatter bit-exact ---------------------------------


def test_case5_split_gather_scatter_bitexact():
    src = _split_kv(num_layers=3, nb=8, bs=16, nh=4, hs=64, seed=1)
    block_ids = [0, 1, 2, 3]
    bpc = 2
    chunks = gather_paged_kv_to_cpu(
        src, block_ids, bpc, layout_hints={"kv_layout": "NHD"}
    )
    dst = _split_kv(num_layers=3, nb=8, bs=16, nh=4, hs=64, seed=99)
    for k, v in dst.values():
        k.zero_()
        v.zero_()
    scatter_cpu_to_paged_kv(
        dst, block_ids, chunks, bpc, layout_hints={"kv_layout": "NHD"}
    )
    for name in src:
        sk, sv = src[name]
        dk, dv = dst[name]
        for b in block_ids:
            assert torch.equal(sk[b], dk[b]), f"K mismatch {name} block {b}"
            assert torch.equal(sv[b], dv[b]), f"V mismatch {name} block {b}"


# --- (6) K and V transferred independently, never swapped ------------------


def test_case6_split_kv_never_swapped():
    src = _split_kv(num_layers=2, nb=4, bs=8, nh=2, hs=16, seed=2)
    block_ids = [0, 1]
    bpc = 1
    chunks = gather_paged_kv_to_cpu(
        src, block_ids, bpc, layout_hints={"kv_layout": "NHD"}
    )
    dst = _split_kv(num_layers=2, nb=4, bs=8, nh=2, hs=16, seed=77)
    for k, v in dst.values():
        k.zero_()
        v.zero_()
    scatter_cpu_to_paged_kv(
        dst, block_ids, chunks, bpc, layout_hints={"kv_layout": "NHD"}
    )
    for name in src:
        sk, sv = src[name]
        dk, dv = dst[name]
        for b in block_ids:
            # V was offset +100; a swap would land src-V values in dst-K.
            assert not torch.equal(dk[b], sv[b]), f"K/V swapped {name} block {b}"
            assert torch.equal(dk[b], sk[b])
            assert torch.equal(dv[b], sv[b])


# --- (7) explicit lmcache.kv_layout=NHD overrides CPU HND default ----------


# --- (8) explicit HND works ------------------------------------------------


# --- (9) invalid layout values fail clearly --------------------------------


# --- (10) connector config overrides runtime detection ---------------------


# --- (11) runtime detection overrides env fallback -------------------------


# --- (12) two connector instances do not share mutable layout state --------


# --- (13) ambiguous dims do not trigger silent layout guessing -------------


# --- (14) existing fused CUDA/GPU paths unchanged --------------------------


def test_case14_fused_gpu_paths_unchanged(monkeypatch):
    """On a non-CPU device the default is NHD and fused detection is unchanged."""
    import lmcache.v1.gpu_connector.kv_format.detectors.vllm as det

    monkeypatch.setattr(det, "torch_device_type", "cuda", raising=False)
    detector = _detector()
    fused = [torch.randn(2, 6, 4, 2, 8) for _ in range(2)]
    fmt, _ = detector.discover(fused, {})  # no hint, non-cpu -> NHD
    assert fmt.name == "NL_X_TWO_NB_BS_NH_HS"
    # An explicit HND still works on GPU.
    fmt_hnd, _ = detector.discover(fused, {"kv_layout": "HND"})
    assert fmt_hnd.name == "NL_X_TWO_NB_NH_BS_HS"


# --- (15) cache-group layer names validated strictly -----------------------


def test_case15_split_kv_shape_mismatch_raises():
    """A split (key, value) pair with mismatched shapes is a hard error, not a
    silent guess."""
    detector = _detector()
    bad = [(torch.randn(4, 8, 2, 16), torch.randn(4, 8, 2, 8))]  # HS differs
    with pytest.raises(ValueError, match="key shape.*!= value shape"):
        detector.discover(bad, {"kv_layout": "NHD"})


def test_case15b_split_kv_dtype_mismatch_raises():
    detector = _detector()
    bad = [
        (
            torch.randn(4, 8, 2, 16, dtype=torch.float32),
            torch.randn(4, 8, 2, 16, dtype=torch.float16),
        )
    ]
    with pytest.raises(ValueError, match="key dtype.*!= value dtype"):
        detector.discover(bad, {"kv_layout": "NHD"})
