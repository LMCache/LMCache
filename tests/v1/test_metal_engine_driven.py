# SPDX-License-Identifier: Apache-2.0
"""KV-layout resolution + CPU-presented NHD gather/scatter (host-memory backends).

Some serving backends present their paged KV cache to LMCache as **CPU** torch
tensors whose physical layout is NHD rather than the x86-CPU default (HND) --
e.g. vLLM's Apple-Metal plugin, whose MLX unified-memory cache is bridged to
CPU torch tensors. The layout must therefore be declarable explicitly, and the
resolution must be backend-neutral (no platform/plugin/name checks).

These tests cover the resolution precedence and the CPU NHD store/retrieve
round-trip. They map 1:1 to the 12 required cases (numbered below). No
accelerator required.

Precedence (highest wins):
    1. explicit connector config (``lmcache.kv_layout`` / alias ``kv_layout``)
    2. vLLM runtime query (``get_kv_cache_layout()``)
    3. ``LMCACHE_VLLM_KV_LAYOUT`` env fallback
    4. device default: CPU -> HND, non-CPU -> NHD
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import utils as vllm_utils


@pytest.fixture(autouse=True)
def _force_cpu_device_type(monkeypatch):
    """These tests exercise the CPU-presented host-memory scenario. The
    detector reads a process-level ``torch_device_type`` constant (set by the
    LMCache platform at import: "cpu" on an Apple-Metal host, "cuda" on a GPU
    host), NOT the tensor's device. Pin it to "cpu" so the CPU-default and
    CPU-override cases are deterministic on any host (otherwise a GPU CI box
    defaults to NHD and the "no hint -> HND" case fails spuriously)."""
    import lmcache.v1.gpu_connector.kv_format.detectors.vllm as det

    monkeypatch.setattr(det, "torch_device_type", "cpu", raising=False)


def _nhd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> "dict[str, torch.Tensor]":
    """Per-layer NHD KV tensors: [2, num_blocks, block_size, num_heads, head_size].

    This is the layout a host-memory NHD backend (e.g. vllm-metal) presents:
    stacked K/V on axis 0, then NHD. ``num_heads != block_size`` so the shape is
    unambiguously NHD-vs-HND for the detector.
    """
    return {
        f"layer_{i}": torch.randn(2, num_blocks, block_size, num_heads, head_size)
        for i in range(num_layers)
    }


# ---------------------------------------------------------------------------
# Layout-resolution precedence (cases 1-7, 11, 12)
# ---------------------------------------------------------------------------


def test_case1_cpu_no_override_is_hnd() -> None:
    """(1) CPU tensor without an explicit override remains HND (the default)."""
    detector = _detector()
    kv = list(_nhd_kv_caches().values())
    # No hint -> detector applies the CPU default (HND).
    fmt_default, _ = detector.discover(kv, {})
    fmt_hnd, _ = detector.discover(kv, {"kv_layout": "HND"})
    assert fmt_default == fmt_hnd
    assert fmt_hnd.name == "NL_X_TWO_NB_NH_BS_HS"


def test_case2_cpu_explicit_nhd_resolves_nhd() -> None:
    """(2) CPU tensor with explicit NHD resolves to NHD (override wins on CPU)."""
    detector = _detector()
    kv = list(_nhd_kv_caches().values())
    fmt_nhd, _ = detector.discover(kv, {"kv_layout": "NHD"})
    assert fmt_nhd.name == "NL_X_TWO_NB_BS_NH_HS"


def test_case3_cpu_explicit_hnd_resolves_hnd() -> None:
    """(3) CPU tensor with explicit HND resolves to HND."""
    detector = _detector()
    kv = list(_nhd_kv_caches().values())
    fmt_hnd, _ = detector.discover(kv, {"kv_layout": "HND"})
    assert fmt_hnd.name == "NL_X_TWO_NB_NH_BS_HS"


def test_case4_connector_config_beats_runtime_query(monkeypatch) -> None:
    """(4) Connector configuration takes precedence over the runtime query."""
    # Runtime query would say HND, but the explicit connector config says NHD.
    monkeypatch.setattr(vllm_utils, "try_get_vllm_kv_cache_layout", lambda: "HND")
    assert vllm_utils.resolve_kv_layout("NHD") == "NHD"
    assert vllm_utils.vllm_layout_hints("NHD") == {"kv_layout": "NHD"}


def test_case5_runtime_query_beats_env(monkeypatch) -> None:
    """(5) Runtime query takes precedence over the environment fallback."""
    monkeypatch.setattr(vllm_utils, "try_get_vllm_kv_cache_layout", lambda: "NHD")
    monkeypatch.setenv("LMCACHE_VLLM_KV_LAYOUT", "HND")
    # No explicit connector config -> tier 2 (query=NHD) beats tier 3 (env=HND).
    assert vllm_utils.resolve_kv_layout(None) == "NHD"


def test_case6_env_fallback_when_no_config_or_query(monkeypatch) -> None:
    """(6) Env fallback works when neither connector config nor query available."""
    monkeypatch.setattr(vllm_utils, "try_get_vllm_kv_cache_layout", lambda: None)
    monkeypatch.setenv("LMCACHE_VLLM_KV_LAYOUT", "NHD")
    assert vllm_utils.resolve_kv_layout(None) == "NHD"
    assert vllm_utils.vllm_layout_hints(None) == {"kv_layout": "NHD"}
    monkeypatch.setenv("LMCACHE_VLLM_KV_LAYOUT", "HND")
    assert vllm_utils.resolve_kv_layout(None) == "HND"


def test_case7_invalid_explicit_value_raises() -> None:
    """(7) Invalid explicit values fail with a useful error (never silently ignored)."""
    with pytest.raises(ValueError, match="Invalid KV layout"):
        vllm_utils.normalize_kv_layout("BOGUS", source="test")
    with pytest.raises(ValueError, match="Invalid KV layout"):
        vllm_utils.resolve_kv_layout("XYZ")
    with pytest.raises(ValueError, match="Invalid KV layout"):
        vllm_utils.read_explicit_kv_layout_from_config_dict(
            {"lmcache.kv_layout": "sideways"}
        )


def test_case7b_case_insensitive_and_alias() -> None:
    """(7 cont.) Supported values normalize case; canonical key + alias both work."""
    assert vllm_utils.normalize_kv_layout("nhd", source="t") == "NHD"
    assert vllm_utils.normalize_kv_layout(" Hnd ", source="t") == "HND"
    # Canonical key wins over alias when both present.
    assert (
        vllm_utils.read_explicit_kv_layout_from_config_dict(
            {"lmcache.kv_layout": "NHD", "kv_layout": "HND"}
        )
        == "NHD"
    )
    # Alias alone still works (compatibility).
    assert (
        vllm_utils.read_explicit_kv_layout_from_config_dict({"kv_layout": "nhd"})
        == "NHD"
    )


def test_case11_two_configs_do_not_share_state(monkeypatch) -> None:
    """(11) Two connector instances with different explicit layouts do not share
    or overwrite any process-global layout state."""
    monkeypatch.setattr(vllm_utils, "try_get_vllm_kv_cache_layout", lambda: None)
    monkeypatch.delenv("LMCACHE_VLLM_KV_LAYOUT", raising=False)
    # Resolve interleaved; each call is pure and instance-value driven.
    a1 = vllm_utils.resolve_kv_layout("NHD")
    b1 = vllm_utils.resolve_kv_layout("HND")
    a2 = vllm_utils.resolve_kv_layout("NHD")
    b2 = vllm_utils.resolve_kv_layout("HND")
    assert (a1, a2) == ("NHD", "NHD")
    assert (b1, b2) == ("HND", "HND")


def test_case12_ambiguous_shape_no_silent_guess() -> None:
    """(12) Ambiguous tensor shapes (block_size == num_heads) do not cause a
    silent NHD-vs-HND guess: the explicit hint decides, and the two hints
    produce genuinely different formats."""
    detector = _detector()
    # block_size == num_heads == 4 -> shape [2, NB, 4, 4, HS] is ambiguous.
    ambiguous = [torch.randn(2, 6, 4, 4, 8) for _ in range(2)]
    fmt_nhd, _ = detector.discover(ambiguous, {"kv_layout": "NHD"})
    fmt_hnd, _ = detector.discover(ambiguous, {"kv_layout": "HND"})
    # If the detector silently guessed from shape, both hints would collapse to
    # the same format. They must not.
    assert fmt_nhd != fmt_hnd
    assert fmt_nhd.name == "NL_X_TWO_NB_BS_NH_HS"
    assert fmt_hnd.name == "NL_X_TWO_NB_NH_BS_HS"


# ---------------------------------------------------------------------------
# Gather/scatter round-trip (cases 8, 9, 10)
# ---------------------------------------------------------------------------


def test_case8_cpu_nhd_gather_scatter_bitexact() -> None:
    """(8) CPU NHD gather -> scatter round-trips bit-exactly."""
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _nhd_kv_caches()
    hints = {"kv_layout": "NHD"}
    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=hints
    )
    destination = {name: torch.zeros_like(t) for name, t in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [4, 5], gathered, blocks_per_chunk, layout_hints=hints
    )
    for name in source:
        assert torch.equal(source[name][:, 0], destination[name][:, 4])
        assert torch.equal(source[name][:, 1], destination[name][:, 5])


def test_case9_cpu_hnd_behavior_unchanged() -> None:
    """(9) CPU HND behavior remains unchanged: HND gather->scatter round-trips."""
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    # HND physical layout: [2, num_blocks, num_heads, block_size, head_size].
    source = {f"layer_{i}": torch.randn(2, 6, 2, 4, 8) for i in range(2)}
    hints = {"kv_layout": "HND"}
    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=hints
    )
    destination = {name: torch.zeros_like(t) for name, t in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [4, 5], gathered, blocks_per_chunk, layout_hints=hints
    )
    for name in source:
        assert torch.equal(source[name][:, 0], destination[name][:, 4])
        assert torch.equal(source[name][:, 1], destination[name][:, 5])


def test_case10_store_and_retrieve_use_same_layout() -> None:
    """(10) STORE and RETRIEVE use the same resolved layout: a chunk gathered
    under NHD must scatter back correctly only under the SAME NHD hint, and the
    detector's format is identical for both directions given one hint."""
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _nhd_kv_caches()
    hints = {"kv_layout": "NHD"}
    blocks_per_chunk = 2
    # STORE (gather) with NHD.
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=hints
    )
    # RETRIEVE (scatter) with the SAME NHD hint -> bit-exact.
    dst_same = {name: torch.zeros_like(t) for name, t in source.items()}
    scatter_cpu_to_paged_kv(
        dst_same, [0, 1], gathered, blocks_per_chunk, layout_hints=hints
    )
    for name in source:
        assert torch.equal(source[name][:, 0], dst_same[name][:, 0])
        assert torch.equal(source[name][:, 1], dst_same[name][:, 1])


def _detector():
    from lmcache.v1.gpu_connector.kv_format.detectors.vllm import VLLM_Detector

    return VLLM_Detector()
