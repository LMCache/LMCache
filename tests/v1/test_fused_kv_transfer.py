# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process transfer of fused/packed (CS / TWO_HS) KV layouts.

Covers the ``MemObjKVLayout`` contract on ``multi_layer_kv_transfer`` plus the
semantics of both declared layouts:

* Two independent single-direction oracles (D2H and H2D) compare against a
  pure-PyTorch reference built from vLLM's layout definition — each engine
  row packs a head's K in the leading half and V in the trailing half of its
  ``CS == 2 * HS`` content dim. Round-trip alone would be blind to a K/V swap.
* Sentinel values encode all five coordinates (layer, token, head, k_or_v,
  offset), so any axis confusion produces loud mismatches.
* Round-trips run over shuffled slot mappings covering odd and even slots
  (the original defect aliased slot pairs), partial chunks,
  ``skip_prefix_n_tokens`` and non-trivial vectorization widths.

The GPU legs run the compiled kernel when CUDA is available; the CPU legs run
the pure-torch fallback. Both compare against the same reference, which is
deliberately independent of either implementation.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import (
    EngineKVFormat,
    MemObjKVLayout,
    TransferDirection,
)
from lmcache.v1.platform import torch_ops

_FUSED_FORMATS = [
    EngineKVFormat.NL_X_NB_BS_NH_CS,
    EngineKVFormat.NL_X_NB_NH_BS_CS,
    EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
    EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
]
_HND_FORMATS = {
    EngineKVFormat.NL_X_NB_NH_BS_CS,
    EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
}

_CUDA = torch.cuda.is_available()
_BACKENDS = ["py"] + (["cuda"] if _CUDA else [])


@dataclass(frozen=True)
class _Geometry:
    num_layers: int
    num_blocks: int
    block_size: int
    num_heads: int
    hs_logical: int  # per-head width of one K or V half

    @property
    def content_size(self) -> int:
        return 2 * self.hs_logical

    @property
    def page_buffer_size(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def hidden(self) -> int:
        return self.num_heads * self.hs_logical

    def layer_shape(self, fmt: EngineKVFormat) -> tuple:
        if fmt in _HND_FORMATS:
            return (
                self.num_blocks,
                self.num_heads,
                self.block_size,
                2 * self.hs_logical,
            )
        return (self.num_blocks, self.block_size, self.num_heads, 2 * self.hs_logical)


_GEO = _Geometry(num_layers=3, num_blocks=8, block_size=4, num_heads=4, hs_logical=8)


def _shuffled_slot_mapping(geo: _Geometry, num_tokens: int, seed: int = 7):
    """A non-trivial slot mapping that must cover both parities: the original
    corruption halved the slot stride, aliasing slot 2t+1 onto the upper half
    of slot t, so an all-even mapping could round-trip by accident."""
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm(geo.page_buffer_size, generator=gen)[:num_tokens]
    assert (slots % 2 == 0).any() and (slots % 2 == 1).any()
    return slots.to(torch.long)


def _reference_kv_rows(layer_t: torch.Tensor, fmt: EngineKVFormat, slots, geo):
    """K/V rows for ``slots`` by explicit indexing of the engine layout:
    K is the leading, V the trailing half of each head's packed content dim."""
    block_idx = slots // geo.block_size
    block_off = slots % geo.block_size
    if fmt in _HND_FORMATS:
        rows = layer_t[block_idx, :, block_off, :]  # (n, NH, CS)
    else:
        rows = layer_t[block_idx, block_off, :, :]  # (n, NH, CS)
    n = slots.numel()
    k = rows[:, :, : geo.hs_logical].reshape(n, -1)
    v = rows[:, :, geo.hs_logical :].reshape(n, -1)
    return k, v


def _sentinel_layers(geo: _Geometry, fmt: EngineKVFormat, device) -> list:
    """Engine tensors whose int32 values encode (layer, slot, head, kv, off)."""
    layers = []
    for layer_id in range(geo.num_layers):
        t = torch.empty(geo.layer_shape(fmt), dtype=torch.int32, device=device)
        slot = torch.arange(geo.page_buffer_size).view(geo.num_blocks, geo.block_size)
        head = torch.arange(geo.num_heads)
        off = torch.arange(geo.hs_logical)
        kv = torch.arange(2)
        # (NB, BS, NH, 2, HS) sentinel grid, then arrange per format.
        val = (
            1_000_000 * kv.view(1, 1, 1, 2, 1)
            + 100_000 * layer_id
            + 1_000 * slot.view(geo.num_blocks, geo.block_size, 1, 1, 1)
            + 100 * head.view(1, 1, geo.num_heads, 1, 1)
            + off.view(1, 1, 1, 1, geo.hs_logical)
        ).to(torch.int32)
        if fmt in _HND_FORMATS:
            val = val.permute(0, 2, 1, 3, 4)  # -> (NB, NH, BS, 2, HS)
        t.copy_(val.reshape(geo.layer_shape(fmt)).to(device))
        layers.append(t)
    return layers


def _run_transfer(
    backend: str,
    key_value: torch.Tensor,
    layers: list,
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    fmt: EngineKVFormat,
    geo: _Geometry,
    layout: MemObjKVLayout,
    head_size: int,
    skip_prefix_n_tokens: int = 0,
) -> None:
    if backend == "py":
        torch_ops.multi_layer_kv_transfer(
            key_value,
            layers,
            slot_mapping,
            torch.device("cpu"),
            geo.page_buffer_size,
            direction,
            fmt,
            block_size=geo.block_size,
            head_size=head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
            mem_obj_kv_layout=layout,
        )
        return
    # First Party
    import lmcache.cuda_ops as lmc_ops

    ptrs = torch.tensor(
        [t.data_ptr() for t in layers], dtype=torch.int64, device="cuda"
    )
    lmc_ops.multi_layer_kv_transfer(
        key_value,
        ptrs,
        slot_mapping.cuda(),
        torch.device("cuda"),
        geo.page_buffer_size,
        TransferDirection.H2D
        if int(direction) == int(TransferDirection.H2D)
        else TransferDirection.D2H,
        EngineKVFormat(int(fmt)),
        block_size=geo.block_size,
        head_size=head_size,
        skip_prefix_n_tokens=skip_prefix_n_tokens,
        mem_obj_kv_layout=MemObjKVLayout(int(layout)),
    )
    torch.cuda.synchronize()


def _device(backend: str) -> torch.device:
    return torch.device("cuda" if backend == "cuda" else "cpu")


def _dummy_transfer_args() -> dict:
    """Minimal arguments; contract validation fires before any tensor use."""
    return dict(
        key_value=torch.zeros(2, 1, 4, 8),
        key_value_ptrs=[torch.zeros(1, 4, 8)],
        slot_mapping=torch.zeros(4, dtype=torch.long),
        paged_memory_device=torch.device("cpu"),
        page_buffer_size=4,
        direction=TransferDirection.D2H,
        engine_kv_format=EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        block_size=4,
    )


@pytest.mark.parametrize("fmt", _FUSED_FORMATS)
def test_fused_format_requires_explicit_layout(fmt):
    args = _dummy_transfer_args()
    args["engine_kv_format"] = fmt
    with pytest.raises(ValueError, match="explicit mem_obj_kv_layout"):
        torch_ops.multi_layer_kv_transfer(**args)


@pytest.mark.parametrize(
    "layout", [MemObjKVLayout.SPLIT_KV_2LTD, MemObjKVLayout.FUSED_PACKED]
)
def test_non_fused_format_rejects_layout(layout):
    args = _dummy_transfer_args()
    args["mem_obj_kv_layout"] = layout
    with pytest.raises(ValueError, match="must pass UNSPECIFIED"):
        torch_ops.multi_layer_kv_transfer(**args)


@pytest.mark.parametrize(
    "layout,dim0",
    [(MemObjKVLayout.SPLIT_KV_2LTD, 1), (MemObjKVLayout.FUSED_PACKED, 2)],
)
def test_layout_shape_mismatch_rejected(layout, dim0):
    geo = _GEO
    args = _dummy_transfer_args()
    args["engine_kv_format"] = EngineKVFormat.NL_X_NB_BS_NH_CS
    args["mem_obj_kv_layout"] = layout
    args["key_value"] = torch.zeros(dim0, geo.num_layers, 4, geo.hidden)
    args["key_value_ptrs"] = [torch.zeros(geo.layer_shape(args["engine_kv_format"]))]
    args["head_size"] = geo.content_size
    with pytest.raises(ValueError, match="requires a"):
        torch_ops.multi_layer_kv_transfer(**args)


# ---------------------------------------------------------------------------
# Single-direction semantic oracles (sentinel-coded, reference-checked)
# ---------------------------------------------------------------------------

_CS_FORMATS = [EngineKVFormat.NL_X_NB_BS_NH_CS, EngineKVFormat.NL_X_NB_NH_BS_CS]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("fmt", _CS_FORMATS)
def test_split_d2h_matches_reference(backend, fmt):
    geo = _GEO
    dev = _device(backend)
    num_tokens = 16
    layers = _sentinel_layers(geo, fmt, dev)
    slots = _shuffled_slot_mapping(geo, num_tokens)
    memobj = torch.zeros(
        2, geo.num_layers, num_tokens, geo.hidden, dtype=torch.int32, device=dev
    )

    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.D2H,
        fmt,
        geo,
        MemObjKVLayout.SPLIT_KV_2LTD,
        head_size=geo.content_size,
    )

    for layer_id, layer_t in enumerate(layers):
        k_ref, v_ref = _reference_kv_rows(layer_t.cpu(), fmt, slots, geo)
        assert torch.equal(memobj[0, layer_id].cpu(), k_ref)
        assert torch.equal(memobj[1, layer_id].cpu(), v_ref)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("fmt", _CS_FORMATS)
def test_split_h2d_matches_reference(backend, fmt):
    geo = _GEO
    dev = _device(backend)
    num_tokens = 16
    slots = _shuffled_slot_mapping(geo, num_tokens)
    # Sentinel-coded memobj planes: kv * 1e6 + layer * 1e5 + token * 1e3 + i.
    memobj = (
        1_000_000 * torch.arange(2).view(2, 1, 1, 1)
        + 100_000 * torch.arange(geo.num_layers).view(1, -1, 1, 1)
        + 1_000 * torch.arange(num_tokens).view(1, 1, -1, 1)
        + torch.arange(geo.hidden).view(1, 1, 1, -1)
    ).to(torch.int32)
    memobj = memobj.to(dev)
    layers = [
        torch.zeros(geo.layer_shape(fmt), dtype=torch.int32, device=dev)
        for _ in range(geo.num_layers)
    ]

    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.H2D,
        fmt,
        geo,
        MemObjKVLayout.SPLIT_KV_2LTD,
        head_size=geo.content_size,
    )

    for layer_id, layer_t in enumerate(layers):
        k_rows, v_rows = _reference_kv_rows(layer_t.cpu(), fmt, slots, geo)
        assert torch.equal(k_rows, memobj[0, layer_id].cpu())
        assert torch.equal(v_rows, memobj[1, layer_id].cpu())
        # Untouched slots stay zero.
        untouched = torch.tensor(
            [s for s in range(geo.page_buffer_size) if s not in set(slots.tolist())]
        )
        uk, uv = _reference_kv_rows(layer_t.cpu(), fmt, untouched, geo)
        assert not uk.any() and not uv.any()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("fmt", _CS_FORMATS)
def test_packed_d2h_matches_reference(backend, fmt):
    """FUSED_PACKED pins the byte-identical staging behavior (CacheBlend/V3):
    the memobj row is the engine's packed row verbatim."""
    geo = _GEO
    head_size = geo.content_size
    dev = _device(backend)
    num_tokens = 16
    layers = _sentinel_layers(geo, fmt, dev)
    slots = _shuffled_slot_mapping(geo, num_tokens)
    memobj = torch.zeros(
        1,
        geo.num_layers,
        num_tokens,
        geo.num_heads * geo.content_size,
        dtype=torch.int32,
        device=dev,
    )

    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.D2H,
        fmt,
        geo,
        MemObjKVLayout.FUSED_PACKED,
        head_size=head_size,
    )

    for layer_id, layer_t in enumerate(layers):
        block_idx = slots // geo.block_size
        block_off = slots % geo.block_size
        if fmt in _HND_FORMATS:
            rows = layer_t.cpu()[block_idx, :, block_off, :]
        else:
            rows = layer_t.cpu()[block_idx, block_off, :, :]
        expected = rows.reshape(slots.numel(), -1)
        assert torch.equal(memobj[0, layer_id].cpu(), expected)


# ---------------------------------------------------------------------------
# Round-trips (random data, both dtypes, partial chunks, skip prefix)
# ---------------------------------------------------------------------------

_ROUNDTRIP_GEOS = [
    _GEO,
    # Non-trivial vectorization: a 6-element half-head is 12 bytes in fp16 —
    # not divisible by 8, so the transfer unit must narrow.
    _Geometry(num_layers=2, num_blocks=6, block_size=4, num_heads=5, hs_logical=6),
    # Realistic Qwen3-like head geometry.
    _Geometry(num_layers=2, num_blocks=4, block_size=8, num_heads=8, hs_logical=128),
]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("fmt", _FUSED_FORMATS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("geo", _ROUNDTRIP_GEOS)
@pytest.mark.parametrize("num_tokens,skip_prefix", [(16, 0), (11, 0), (16, 3)])
def test_split_roundtrip(backend, fmt, dtype, geo, num_tokens, skip_prefix):
    dev = _device(backend)
    torch.manual_seed(0)
    layers = [
        torch.randn(geo.layer_shape(fmt), dtype=dtype).to(dev)
        for _ in range(geo.num_layers)
    ]
    originals = [t.clone() for t in layers]
    slots = _shuffled_slot_mapping(geo, num_tokens)
    memobj = torch.zeros(
        2, geo.num_layers, num_tokens, geo.hidden, dtype=dtype, device=dev
    )

    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.D2H,
        fmt,
        geo,
        MemObjKVLayout.SPLIT_KV_2LTD,
        head_size=geo.content_size,
        skip_prefix_n_tokens=skip_prefix,
    )

    # Restore into wiped engine tensors and compare the transferred slots.
    for t in layers:
        t.zero_()
    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.H2D,
        fmt,
        geo,
        MemObjKVLayout.SPLIT_KV_2LTD,
        head_size=geo.content_size,
        skip_prefix_n_tokens=skip_prefix,
    )

    live = slots[skip_prefix:]
    for restored, original in zip(layers, originals, strict=True):
        rk, rv = _reference_kv_rows(restored.cpu(), fmt, live, geo)
        ok, ov = _reference_kv_rows(original.cpu(), fmt, live, geo)
        assert torch.equal(rk, ok)
        assert torch.equal(rv, ov)
        # Slots outside the mapping (and skipped prefix slots) stay wiped.
        untouched = torch.tensor(
            [s for s in range(geo.page_buffer_size) if s not in set(live.tolist())]
        )
        uk, uv = _reference_kv_rows(restored.cpu(), fmt, untouched, geo)
        assert not uk.any() and not uv.any()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("fmt", _CS_FORMATS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_roundtrip(backend, fmt, dtype):
    geo = _GEO
    head_size = geo.content_size
    dev = _device(backend)
    torch.manual_seed(1)
    layers = [
        torch.randn(geo.layer_shape(fmt), dtype=dtype).to(dev)
        for _ in range(geo.num_layers)
    ]
    originals = [t.clone() for t in layers]
    num_tokens = 16
    slots = _shuffled_slot_mapping(geo, num_tokens)
    memobj = torch.zeros(
        1,
        geo.num_layers,
        num_tokens,
        geo.num_heads * geo.content_size,
        dtype=dtype,
        device=dev,
    )

    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.D2H,
        fmt,
        geo,
        MemObjKVLayout.FUSED_PACKED,
        head_size=head_size,
    )
    for t in layers:
        t.zero_()
    _run_transfer(
        backend,
        memobj,
        layers,
        slots.to(dev),
        TransferDirection.H2D,
        fmt,
        geo,
        MemObjKVLayout.FUSED_PACKED,
        head_size=head_size,
    )

    for restored, original in zip(layers, originals, strict=True):
        rk, rv = _reference_kv_rows(restored.cpu(), fmt, slots, geo)
        ok, ov = _reference_kv_rows(original.cpu(), fmt, slots, geo)
        assert torch.equal(rk, ok)
        assert torch.equal(rv, ov)
