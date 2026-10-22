# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM KV cache registration view normalization.

Covers the manager-logical re-view of kernel-paged MLA caches for pure
(non-Mamba-hybrid) MLA models: GLM-5.3 NVFP4 DSA keeps 256-token logical
blocks over 64-slot ``FLASHMLA_SPARSE`` kernel pages, and its
``uint8[..., 132]`` values/scales planes must transfer opaquely.
"""

# Standard
from types import SimpleNamespace
from typing import Literal

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="KV cache group edits import vLLM at module top")

# Third Party
from vllm.v1.kv_cache_interface import KVCacheSpecKind  # noqa: E402

# First Party
from lmcache.integration.vllm import kv_cache_group_edits  # noqa: E402
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints  # noqa: E402


def _mla_kind(_spec: object) -> KVCacheSpecKind:
    return KVCacheSpecKind.MLA_ATTENTION


def _pure_mla_config(spec: SimpleNamespace) -> SimpleNamespace:
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["layer.0"])
    # has_mamba_layers=False on purpose: pure MLA models must reach the edit
    # registry without any Mamba gate.
    return SimpleNamespace(kv_cache_groups=[group], has_mamba_layers=False)


def _uniform_mla_config(
    leaf_spec: SimpleNamespace, manager_block_size: int = 256
) -> SimpleNamespace:
    group_spec = SimpleNamespace(
        block_size=manager_block_size,
        page_size_bytes=(
            leaf_spec.page_size_bytes * manager_block_size // leaf_spec.block_size
        ),
        kv_cache_specs={"layer.0": leaf_spec},
    )
    return _pure_mla_config(group_spec)


def _leaf_spec(
    block_size: int, page_size_bytes: int, **overrides: int
) -> SimpleNamespace:
    fields: dict[str, int] = {
        "block_size": block_size,
        "page_size_bytes": page_size_bytes,
        "compress_ratio": 1,
        "tokens_per_state": 1,
        "tq_slot_size": 0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_subpaged_mla_uses_manager_block_size_with_physical_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vLLM leaf page must expand to its manager's logical block."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=64, page_size_bytes=64 * 8)
    cache = torch.arange(8 * 64 * 8, dtype=torch.int64).to(torch.uint8)
    cache = cache.view(8, 64, 8)

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec), {"layer.0": cache}, layout_hints={}
    )["layer.0"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == (2, 256, 8)
    assert edited.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(edited.flatten(), cache.flatten())


def test_subpaged_mla_accepts_logical_leaf_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logical-width leaf spec must retain the existing zero-copy behavior."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=256, page_size_bytes=256 * 8)
    cache = torch.arange(8 * 64 * 8, dtype=torch.int64).to(torch.uint8)
    cache = cache.view(8, 64, 8)

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec), {"layer.0": cache}, layout_hints={}
    )["layer.0"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == (2, 256, 8)
    assert edited.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(edited.flatten(), cache.flatten())


@pytest.mark.parametrize("compression_field", ["compress_ratio", "tokens_per_state"])
def test_declared_mla_compression_is_not_reinterpreted(
    monkeypatch: pytest.MonkeyPatch, compression_field: str
) -> None:
    """A leaf declaring slot compression keeps its declared granularity.

    The manager target shrinks by the same tokens-per-state factor, so a
    leaf spec whose granularity already matches its kernel pages stays
    untouched instead of being reinterpreted as full-width logical pages.
    """
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    # Only the parametrized declaration is present on the spec: tokens_per_state
    # takes precedence over compress_ratio, so setting both would not
    # exercise the legacy fallback.
    fields: dict[str, int] = {"block_size": 64, "page_size_bytes": 64 * 8}
    fields[compression_field] = 4
    leaf_spec = SimpleNamespace(**fields)
    cache = torch.zeros(8, 64, 8, dtype=torch.uint8)

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec), {"layer.0": cache}, layout_hints={}
    )["layer.0"]

    assert edited is cache


def test_declared_state_packing_re_pages_to_declared_granularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared packing is preserved while re-paging to the manager block.

    GLM-5.3-Flash kpool-indexer geometry: a 1152-token manager block over
    32-row kernel pages with tokens_per_state=4 must land at 288 stored
    states per logical page, without marking the opaque DSA hint.
    """
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(
        block_size=1152,
        # States-granularity page bytes: 1152 tokens / tokens_per_state=4
        # leaves 288 stored states at 132 bytes each.
        page_size_bytes=1152 * 132 // 4,
        tokens_per_state=4,
    )
    cache = torch.arange(18 * 1 * 32 * 132, dtype=torch.int64).to(torch.uint8)
    cache = cache.view(18, 1, 32, 132)
    layout_hints: LayoutHints = {}

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec, manager_block_size=1152),
        {"layer.0": cache},
        layout_hints=layout_hints,
    )["layer.0"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == (2, 1, 288, 132)
    assert edited.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert "subpaged_dsa_opaque" not in layout_hints


@pytest.mark.parametrize(
    ("manager_block_size", "num_kernel_pages", "page_size_bytes", "message"),
    [
        (250, 8, 64 * 8, "is not a multiple"),
        (256, 7, 64 * 8, "kernel page count"),
        (256, 8, 64 * 7, "different bytes per token"),
    ],
)
def test_subpaged_mla_rejects_invalid_geometry(
    monkeypatch: pytest.MonkeyPatch,
    manager_block_size: int,
    num_kernel_pages: int,
    page_size_bytes: int,
    message: str,
) -> None:
    """Invalid manager/kernel geometry must fail instead of losing KV bytes."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=64, page_size_bytes=page_size_bytes)
    cache = torch.zeros(num_kernel_pages, 64, 8, dtype=torch.uint8)

    with pytest.raises(ValueError, match=message):
        kv_cache_group_edits.apply_kv_cache_group_edits(
            _uniform_mla_config(leaf_spec, manager_block_size),
            {"layer.0": cache},
            layout_hints={},
        )


def test_subpaged_mla_rejects_noncontiguous_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-contiguous kernel page must not be re-viewed as logical pages."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=64, page_size_bytes=64 * 8)
    cache = torch.zeros(8, 64, 16, dtype=torch.uint8)[:, :, ::2]
    assert not cache.is_contiguous()

    with pytest.raises(ValueError, match="must be contiguous"):
        kv_cache_group_edits.apply_kv_cache_group_edits(
            _uniform_mla_config(leaf_spec),
            {"layer.0": cache},
            layout_hints={},
        )


@pytest.mark.parametrize(
    ("kv_layout", "input_shape", "output_shape"),
    [
        ("NHD", (8, 1, 64, 8), (2, 1, 256, 8)),
        ("HND", (8, 64, 1, 8), (2, 256, 1, 8)),
    ],
)
def test_subpaged_mla_normalizes_rank4_fused_layout(
    monkeypatch: pytest.MonkeyPatch,
    kv_layout: Literal["NHD", "HND"],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    """Rank-4 MLA pages re-page at manager granularity despite stale hints.

    The input's head-slot order and the supplied kv_layout hint disagree on
    the HND case; the block axis is identified from bytes per stored state
    and the singleton head axis stays where the raw tensor put it, so the
    view is correct either way.
    """
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=256, page_size_bytes=256 * 8)
    cache = torch.arange(8 * 64 * 8, dtype=torch.int64).to(torch.uint8)
    cache = cache.view(input_shape)

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec),
        {"layer.0": cache},
        layout_hints={"kv_layout": kv_layout},
    )["layer.0"]

    assert isinstance(edited, torch.Tensor)
    assert edited.shape == output_shape
    assert edited.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(edited.flatten(), cache.flatten())


@pytest.mark.parametrize(
    ("kv_layout", "input_shape", "output_shape"),
    [
        ("NHD", (8, 1, 64, 132), (2, 1, 256, 132)),
        ("HND", (8, 64, 1, 132), (2, 256, 1, 132)),
    ],
)
def test_subpaged_dsa_marks_opaque_uint8_layout(
    monkeypatch: pytest.MonkeyPatch,
    kv_layout: Literal["NHD", "HND"],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    """DSA subpages must stay uint8 planes and select opaque transfer."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=256, page_size_bytes=256 * 132)
    cache = torch.arange(8 * 64 * 132, dtype=torch.int64).to(torch.uint8)
    cache = cache.view(input_shape)
    layout_hints: LayoutHints = {"kv_layout": kv_layout}

    edited = kv_cache_group_edits.apply_kv_cache_group_edits(
        _uniform_mla_config(leaf_spec),
        {"layer.0": cache},
        layout_hints=layout_hints,
    )["layer.0"]

    assert isinstance(edited, torch.Tensor)
    assert edited.dtype == torch.uint8
    assert edited.shape == output_shape
    assert edited.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(edited.flatten(), cache.flatten())
    assert layout_hints["subpaged_dsa_opaque"] is True


def test_subpaged_dsa_rejects_mixed_native_page_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registration-wide opaque hint must not mask a native DSA page."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=256, page_size_bytes=256 * 132)
    group_spec = SimpleNamespace(
        block_size=256,
        page_size_bytes=leaf_spec.page_size_bytes,
        kv_cache_specs={"layer.0": leaf_spec, "layer.1": leaf_spec},
    )
    config = _pure_mla_config(group_spec)
    config.kv_cache_groups[0].layer_names = ["layer.0", "layer.1"]
    subpaged = torch.zeros(8, 1, 64, 132, dtype=torch.uint8)
    native = torch.zeros(2, 256, 132, dtype=torch.uint8)

    with pytest.raises(ValueError, match="cannot mix opaque subpaged DSA"):
        kv_cache_group_edits.apply_kv_cache_group_edits(
            config,
            {"layer.0": subpaged, "layer.1": native},
            layout_hints={"kv_layout": "NHD"},
        )


def test_subpaged_dsa_rejects_external_opaque_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving engines must not forge LMCache's internal DSA format hint."""
    monkeypatch.setattr(kv_cache_group_edits, "get_kv_cache_spec_kind", _mla_kind)
    leaf_spec = _leaf_spec(block_size=256, page_size_bytes=256 * 132)
    cache = torch.zeros(8, 1, 64, 132, dtype=torch.uint8)

    with pytest.raises(ValueError, match="internal LMCache hint"):
        kv_cache_group_edits.apply_kv_cache_group_edits(
            _uniform_mla_config(leaf_spec),
            {"layer.0": cache},
            layout_hints={"subpaged_dsa_opaque": True},
        )
