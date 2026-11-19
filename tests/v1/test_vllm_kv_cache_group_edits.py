# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``lmcache.integration.vllm.kv_cache_group_edits``.

Regression coverage for the vLLM 0.26.0 KV layout change: the mainline
non-MLA attention backends moved from the rank-5 K/V-split KV layout

    ``(num_blocks, 2, block_size, num_kv_heads, head_size)``

to the rank-4 K/V-fused layout

    ``(num_blocks, num_kv_heads, block_size, 2 * head_size)``

(``vllm/v1/attention/backends/flash_attn.py``, ``flashinfer.py``,
``triton_attn.py``, ``flex_attention.py``, ``rocm_aiter_fa.py``). Not every
backend moved -- ``hpc_attn.py`` is still rank 5 -- so both layouts are
covered here. The sub-paged attention edit gated on ``ndim == 5`` and so
silently stopped
firing, leaving the attention group paged at the backend's kernel block size
while LMCache addressed it in scheduler block-id units -- a silent
store/retrieve corruption with no exception and no log line.

These tests run on CPU: the module's vLLM imports are stubbed before import,
so they behave the same whether or not vLLM is installed.
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import TypeAlias
from unittest.mock import patch
import importlib
import sys

# Third Party
import pytest
import torch

# One registered cache value, mirroring the module's own ``RegisteredKVCache``:
# a paged KV tensor, or ``[conv_state, ssm_state]`` for Mamba layers.
_RegisteredKVCache: TypeAlias = torch.Tensor | list[torch.Tensor]

# A reported geometry: a Mamba/GDN hybrid whose attention
# block size vLLM inflated to 1600 to align with the Mamba page, served by a
# backend advertising fixed kernel block sizes (FlashInfer's [16, 32, 64]),
# so ``select_common_block_size`` re-pages at 64 -- the largest advertised
# size dividing 1600.
LOGICAL_BLOCK_SIZE = 1600
KERNEL_BLOCK_SIZE = 64
SUBPAGE_RATIO = LOGICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE  # 25
NUM_LOGICAL_BLOCKS = 4
NUM_KERNEL_PAGES = NUM_LOGICAL_BLOCKS * SUBPAGE_RATIO
NUM_KV_HEADS = 2
HEAD_SIZE = 8
NUM_MAMBA_GROUPS = 4


class _SpecKind(Enum):
    """Stand-in for ``vllm.v1.kv_cache_interface.KVCacheSpecKind``."""

    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    CROSS_ATTENTION = "cross_attention"
    MAMBA = "mamba"
    MLA_ATTENTION = "mla_attention"


@dataclass
class _Spec:
    """Stand-in for a vLLM ``KVCacheSpec`` leaf."""

    kind: _SpecKind
    block_size: int
    page_size_bytes: int
    real_page_size_bytes: int | None = None
    mamba_cache_mode: str = "align"


@dataclass
class _Group:
    layer_names: list[str]
    kv_cache_spec: _Spec


@dataclass
class _Config:
    kv_cache_groups: list[_Group]
    has_mamba_layers: bool = True


_MODULE = "lmcache.integration.vllm.kv_cache_group_edits"


@pytest.fixture(scope="module")
def edits():
    """Import the module under test with its vLLM dependencies stubbed."""
    stub = ModuleType("vllm.v1.kv_cache_interface")
    stub.KVCacheConfig = _Config
    stub.KVCacheSpec = _Spec
    stub.KVCacheSpecKind = _SpecKind
    stub.get_kv_cache_spec_kind = lambda spec: spec.kind

    vllm_pkg = ModuleType("vllm")
    vllm_v1 = ModuleType("vllm.v1")
    # ``from pkg import child`` returns the attribute ``pkg`` already holds, so
    # dropping only the ``sys.modules`` entry would yield whatever binding an
    # earlier importer left behind -- against real vLLM, no rule then matches.
    parent = importlib.import_module("lmcache.integration.vllm")
    previous = sys.modules.get(_MODULE)
    try:
        with patch.dict(
            sys.modules,
            {
                "vllm": vllm_pkg,
                "vllm.v1": vllm_v1,
                "vllm.v1.kv_cache_interface": stub,
            },
        ):
            sys.modules.pop(_MODULE, None)
            module = importlib.import_module(_MODULE)
            assert module.KVCacheSpecKind is _SpecKind
            yield module
    finally:
        sys.modules.pop(_MODULE, None)
        parent.__dict__.pop("kv_cache_group_edits", None)
        if previous is not None:
            sys.modules[_MODULE] = previous
            parent.kv_cache_group_edits = previous


def _attention_page_bytes() -> int:
    """Bytes in one *logical* attention page, for either layout."""
    elems = 2 * LOGICAL_BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    return elems * torch.finfo(torch.bfloat16).bits // 8


def _attention_spec() -> _Spec:
    return _Spec(
        kind=_SpecKind.FULL_ATTENTION,
        block_size=LOGICAL_BLOCK_SIZE,
        page_size_bytes=_attention_page_bytes(),
    )


def _fused_kv_cache(kernel_block_size: int = KERNEL_BLOCK_SIZE) -> torch.Tensor:
    """A vLLM >= 0.26.0 rank-4 fused-K/V attention tensor.

    Shape ``(num_kernel_pages, num_kv_heads, kernel_block_size,
    2 * head_size)`` -- ``flash_attn.py:141`` / ``flashinfer.py:408``.
    """
    pages = NUM_LOGICAL_BLOCKS * (LOGICAL_BLOCK_SIZE // kernel_block_size)
    return torch.arange(
        pages * NUM_KV_HEADS * kernel_block_size * 2 * HEAD_SIZE,
        dtype=torch.bfloat16,
    ).view(pages, NUM_KV_HEADS, kernel_block_size, 2 * HEAD_SIZE)


def _split_kv_cache(kernel_block_size: int = KERNEL_BLOCK_SIZE) -> torch.Tensor:
    """A vLLM <= 0.25.x rank-5 K/V-split attention tensor.

    Shape ``(num_kernel_pages, 2, kernel_block_size, num_kv_heads,
    head_size)`` -- ``flash_attn.py:132`` / ``flashinfer.py:392``.
    """
    pages = NUM_LOGICAL_BLOCKS * (LOGICAL_BLOCK_SIZE // kernel_block_size)
    return torch.arange(
        pages * 2 * kernel_block_size * NUM_KV_HEADS * HEAD_SIZE,
        dtype=torch.bfloat16,
    ).view(pages, 2, kernel_block_size, NUM_KV_HEADS, HEAD_SIZE)


def _mamba_spec(page_bytes: int) -> _Spec:
    return _Spec(
        kind=_SpecKind.MAMBA,
        block_size=LOGICAL_BLOCK_SIZE,
        page_size_bytes=page_bytes,
        mamba_cache_mode="align",
    )


def _mamba_kv_cache(
    prefix_elems: int = 0,
) -> tuple[list[torch.Tensor], int, torch.Tensor]:
    """A Mamba ``[conv_state, ssm_state]`` pair sharing one padded page.

    Returns the pair, the page size in bytes, and the backing storage tensor.
    Per-block stride spans the whole page (``conv | ssm | pad``), which is what
    ``_MambaPageViewEdit.apply`` re-strides over. When ``prefix_elems > 0``,
    bytes ``[0:prefix_elems)`` are filler and conv/ssm use that storage offset
    (vLLM-Ascend v0.30 shared-pool geometry).
    """
    elem = torch.finfo(torch.bfloat16).bits // 8
    # Page must factor as (2, block_size, 1, head_size) for the synthetic view.
    elems_per_page = 2 * LOGICAL_BLOCK_SIZE * 3
    conv_elems, ssm_elems = 2 * LOGICAL_BLOCK_SIZE, 2 * LOGICAL_BLOCK_SIZE

    storage = torch.arange(
        prefix_elems + NUM_LOGICAL_BLOCKS * elems_per_page, dtype=torch.bfloat16
    )
    if prefix_elems:
        storage[:prefix_elems] = -1
    conv_state = storage.as_strided(
        (NUM_LOGICAL_BLOCKS, conv_elems),
        (elems_per_page, 1),
        storage_offset=prefix_elems,
    )
    ssm_state = storage.as_strided(
        (NUM_LOGICAL_BLOCKS, ssm_elems),
        (elems_per_page, 1),
        storage_offset=prefix_elems + conv_elems,
    )
    return [conv_state, ssm_state], elems_per_page * elem, storage


def _slice_strided_mamba_kv_cache() -> tuple[list[torch.Tensor], int]:
    """A Mamba ``[conv_state, ssm_state]`` pair with slice-sized block strides.

    Returns the pair and a hybrid ``page_size_bytes`` for the spec (conv + ssm
    + pad) that is larger than either slice. Each state is a dense contiguous
    view: stepping along dim-0 advances only that state's bytes, which is what
    vLLM-Ascend <= v0.29 registers. That layout matches
    ``_MambaSlicePlaneEdit`` and must not match ``_MambaPageViewEdit``.
    """
    elem = torch.finfo(torch.bfloat16).bits // 8
    conv_nbytes = LOGICAL_BLOCK_SIZE
    ssm_nbytes = 2 * LOGICAL_BLOCK_SIZE
    page_bytes = conv_nbytes + ssm_nbytes + 32
    conv_elems = conv_nbytes // elem
    ssm_elems = ssm_nbytes // elem
    conv_storage = torch.arange(NUM_LOGICAL_BLOCKS * conv_elems, dtype=torch.bfloat16)
    conv_state = conv_storage.view(NUM_LOGICAL_BLOCKS, conv_elems)
    ssm_storage = torch.arange(NUM_LOGICAL_BLOCKS * ssm_elems, dtype=torch.bfloat16)
    ssm_state = ssm_storage.view(NUM_LOGICAL_BLOCKS, ssm_elems)
    return [conv_state, ssm_state], page_bytes


def _hybrid_config(
    attention_cache: torch.Tensor, attention_spec: _Spec | None = None
) -> tuple[_Config, dict[str, _RegisteredKVCache]]:
    """Build the hybrid group layout: 4 Mamba groups + 1 attention group.

    Args:
        attention_cache: Registered tensor for the single attention layer.
        attention_spec: Spec for the attention group; defaults to the
            inflated block size 1600.

    Returns:
        The vLLM-shaped config and the registered ``kv_caches`` mapping.
    """
    mamba_caches, mamba_page_bytes, _ = _mamba_kv_cache()
    groups = [
        _Group(layer_names=[f"mamba.{i}"], kv_cache_spec=_mamba_spec(mamba_page_bytes))
        for i in range(NUM_MAMBA_GROUPS)
    ]
    groups.append(
        _Group(
            layer_names=["attn.0"],
            kv_cache_spec=attention_spec or _attention_spec(),
        )
    )

    kv_caches: dict[str, _RegisteredKVCache] = {
        f"mamba.{i}": list(mamba_caches) for i in range(NUM_MAMBA_GROUPS)
    }
    kv_caches["attn.0"] = attention_cache
    return _Config(kv_cache_groups=groups), kv_caches


def _edit_attention(
    edits, attention_cache: torch.Tensor, attention_spec: _Spec | None = None
) -> torch.Tensor:
    """Run the public entry point and return the edited attention tensor.

    Args:
        edits: The module under test.
        attention_cache: Registered tensor for the attention layer.
        attention_spec: Optional attention spec override.

    Returns:
        The attention layer's entry in the edited mapping. It is the *same
        object* as ``attention_cache`` when no rule matched.
    """
    config, kv_caches = _hybrid_config(attention_cache, attention_spec)
    edited = edits.apply_kv_cache_group_edits(config, kv_caches, layout_hints={})
    return edited["attn.0"]


# --------------------------------------------------------------------------
# Which layouts the sub-paged rule fires on -- the 0.26.0 regression gate
# --------------------------------------------------------------------------


def test_subpaged_edit_fires_on_fused_kv_layout(edits):
    """vLLM >= 0.26.0 rank-4 fused layout must be re-paged.

    Before the fix the rule required ``ndim == 5``, so it silently skipped
    this layout and left the group at the kernel block size.
    """
    cache = _fused_kv_cache()
    assert _edit_attention(edits, cache).shape[2] == LOGICAL_BLOCK_SIZE


def test_subpaged_edit_fires_on_split_kv_layout(edits):
    """vLLM <= 0.25.x rank-5 split layout must still be re-paged."""
    cache = _split_kv_cache()
    assert _edit_attention(edits, cache).shape[2] == LOGICAL_BLOCK_SIZE


@pytest.mark.parametrize("make_cache", [_fused_kv_cache, _split_kv_cache])
def test_subpaged_edit_skips_unsubpaged_tensor(edits, make_cache):
    """A backend paging at the logical block size needs no edit.

    FlashAttention advertises ``MultipleOf(16)``, so ``select_common_block_size``
    returns the logical block size unchanged and the registered block dim
    already equals ``spec.block_size``. The tensor must pass through untouched.
    """
    cache = make_cache(kernel_block_size=LOGICAL_BLOCK_SIZE)
    assert _edit_attention(edits, cache) is cache


def test_subpaged_edit_skips_declared_compression(edits):
    """Declared slot compression belongs to the compression path, not here."""
    spec = _attention_spec()
    spec.compress_ratio = 2
    cache = _fused_kv_cache()
    assert _edit_attention(edits, cache, attention_spec=spec) is cache


# --------------------------------------------------------------------------
# The re-view itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("make_cache", [_fused_kv_cache, _split_kv_cache])
def test_subpaged_edit_restores_block_granularity(edits, make_cache):
    """The edited view's block dim must equal the scheduler block size.

    This is the invariant the design doc states edits exist to restore:
    "the registered tensor's paging granularity must equal the block-id
    granularity".
    """
    cache = make_cache()
    viewed = _edit_attention(edits, cache)

    assert viewed.ndim == 5
    assert viewed.shape[0] == NUM_LOGICAL_BLOCKS
    assert viewed.shape[2] == LOGICAL_BLOCK_SIZE
    # Same storage, never a copy.
    assert viewed.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert viewed.numel() == cache.numel()


@pytest.mark.parametrize("make_cache", [_fused_kv_cache, _split_kv_cache])
def test_subpaged_edit_preserves_byte_order(edits, make_cache):
    """The re-view must be a pure reinterpretation of the same bytes.

    Store and retrieve share this mapping, so byte order is what has to
    round-trip -- the dims are addressing metadata only.
    """
    cache = make_cache()
    torch.testing.assert_close(
        _edit_attention(edits, cache).reshape(-1), cache.reshape(-1)
    )


@pytest.mark.parametrize("make_cache", [_fused_kv_cache, _split_kv_cache])
def test_subpaged_edit_groups_contiguous_kernel_pages(edits, make_cache):
    """Logical block ``n`` must cover kernel pages ``n*k .. n*k+k-1``.

    vLLM expands the worker-side block table the same way
    (``BlockTable.map_to_kernel_blocks``), so this grouping is what makes
    scheduler block IDs address the right bytes.
    """
    cache = make_cache()
    viewed = _edit_attention(edits, cache)

    for block in range(NUM_LOGICAL_BLOCKS):
        expected = cache[block * SUBPAGE_RATIO : (block + 1) * SUBPAGE_RATIO]
        torch.testing.assert_close(viewed[block].reshape(-1), expected.reshape(-1))


def test_subpaged_edit_handles_permuted_fused_registration(edits):
    """vLLM hands back a permute view, which need not be logically contiguous.

    With a KV connector attached and no override, ``get_kv_cache_layout()``
    defaults to NHD; FlashInfer's NHD stride order ``(0, 2, 1, 3)`` means the
    registered rank-4 tensor is a transposed -- non-contiguous -- view of a
    contiguous allocation.
    """
    physical = torch.arange(
        NUM_KERNEL_PAGES * KERNEL_BLOCK_SIZE * NUM_KV_HEADS * 2 * HEAD_SIZE,
        dtype=torch.bfloat16,
    ).view(NUM_KERNEL_PAGES, KERNEL_BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD_SIZE)
    registered = physical.permute(0, 2, 1, 3)

    assert not registered.is_contiguous()
    assert registered.shape[2] == KERNEL_BLOCK_SIZE

    viewed = _edit_attention(edits, registered)
    assert viewed.shape[0] == NUM_LOGICAL_BLOCKS
    assert viewed.shape[2] == LOGICAL_BLOCK_SIZE
    torch.testing.assert_close(viewed.reshape(-1), physical.reshape(-1))


def test_subpaged_edit_rejects_permuted_split_registration(edits):
    """A permuted rank-5 tensor must fail loudly rather than be re-viewed.

    vLLM <= 0.25.x under HND registers rank 5 with stride order
    ``(0, 1, 3, 2, 4)``. Re-viewing it in memory order would succeed and then
    be misread downstream -- the edited view is NHD-shaped, so the HND
    detector branch reads its block size from the synthetic ``num_heads`` axis
    and resolves 1. This rule's whole purpose is to prevent a silently wrong
    block size, so it must not introduce one.
    """
    physical = torch.zeros(
        NUM_KERNEL_PAGES,
        2,
        NUM_KV_HEADS,
        KERNEL_BLOCK_SIZE,
        HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    registered = physical.permute(0, 1, 3, 2, 4)

    assert not registered.is_contiguous()
    assert registered.shape[2] == KERNEL_BLOCK_SIZE
    with pytest.raises(ValueError, match="must be contiguous"):
        _edit_attention(edits, registered)


def test_subpaged_edit_rejects_untiled_page(edits):
    """An undeclared packed layout must fail loudly, not transfer wrongly."""
    spec = _attention_spec()
    spec.page_size_bytes *= 2
    with pytest.raises(ValueError, match="do not tile the logical page"):
        _edit_attention(edits, _fused_kv_cache(), attention_spec=spec)


def test_subpaged_apply_rejects_unknown_rank(edits):
    """Ranks other than 4 and 5 are not attention KV layouts.

    Driven against the rule directly rather than through
    ``apply_kv_cache_group_edits``: the rule's own gate screens on rank, so a
    rank-3 tensor is passed through untouched and can never reach ``apply``
    via the public entry point. The guard is only reachable if a future rank
    is added to the gate without updating ``apply``, which is exactly what
    this pins.
    """
    edit = edits._SubpagedAttentionViewEdit()
    with pytest.raises(ValueError, match="attention KV tensor"):
        edit.apply(_attention_spec(), torch.zeros(4, 8, 8), {})


def test_mamba_page_view_skips_slice_strided_list(edits):
    """Slice-strided conv/ssm must use the plane edit, not the page view.

    When per-block stride equals each slice's byte length rather than
    ``page_size_bytes``, ``_MambaPageViewEdit`` must not match and
    ``_MambaSlicePlaneEdit`` must.
    """
    caches, page_bytes = _slice_strided_mamba_kv_cache()
    spec = _mamba_spec(page_bytes)
    assert not edits._MambaPageViewEdit().matches(spec, caches)
    assert edits._MambaSlicePlaneEdit().matches(spec, caches)


def test_mamba_slice_planes_on_slice_strided_list(edits):
    """Slice-strided conv/ssm are re-viewed as two uint8 format-17 planes.

    End-to-end through ``apply_kv_cache_group_edits``: each state keeps its
    own width (conv and ssm differ), dtype is ``uint8``, and the token axis
    matches ``spec.block_size``.
    """
    caches, page_bytes = _slice_strided_mamba_kv_cache()
    spec = _mamba_spec(page_bytes)
    config = _Config(
        kv_cache_groups=[_Group(layer_names=["mamba.0"], kv_cache_spec=spec)]
    )
    edited = edits.apply_kv_cache_group_edits(config, {"mamba.0": caches}, {})
    planes = edited["mamba.0"]
    assert isinstance(planes, list)
    assert len(planes) == 2
    widths = [int(plane.shape[-1]) for plane in planes]
    assert widths[0] != widths[1]
    for plane in planes:
        assert plane.dtype == torch.uint8
        assert plane.ndim == 4
        assert int(plane.shape[2]) == 1


def test_mamba_slice_planes_skip_indivisible_slice(edits):
    """Non-divisible per-block bytes must not match the slice-plane edit.

    ``_MambaSlicePlaneEdit`` requires ``nbytes_per_block % block_size == 0``
    so ``width`` is integral. An off-by-one ``block_size`` must leave the
    cache unchanged through the public entry point.
    """
    caches, page_bytes = _slice_strided_mamba_kv_cache()
    spec = _mamba_spec(page_bytes)
    spec.block_size = LOGICAL_BLOCK_SIZE + 1
    assert not edits._MambaSlicePlaneEdit().matches(spec, caches)
    config = _Config(
        kv_cache_groups=[_Group(layer_names=["mamba.0"], kv_cache_spec=spec)]
    )
    edited = edits.apply_kv_cache_group_edits(config, {"mamba.0": caches}, {})
    assert edited["mamba.0"] is caches


def test_mamba_page_view_applies_on_packed_page(edits):
    """Packed page-base conv still uses the page-view edit.

    When conv's per-block stride spans the full padded page, only
    ``_MambaPageViewEdit`` may match; the slice-plane rule must not.
    """
    caches, page_bytes, _ = _mamba_kv_cache()
    spec = _mamba_spec(page_bytes)
    assert edits._MambaPageViewEdit().matches(spec, caches)
    assert not edits._MambaSlicePlaneEdit().matches(spec, caches)


# --------------------------------------------------------------------------
# End to end: both edits fire together on the hybrid geometry
# --------------------------------------------------------------------------


def test_both_edits_fire_on_fused_kv_hybrid(edits):
    """The end-to-end assertion for the 0.26.0 regression.

    Before the fix the startup log for this geometry read
    ``KV cache group edits applied: {'mamba-page-view': 4}`` -- the attention
    edit left no key because a non-matching rule never increments the counter.
    Both rules must now fire.
    """
    config, kv_caches = _hybrid_config(_fused_kv_cache())
    edited = edits.apply_kv_cache_group_edits(config, kv_caches, layout_hints={})

    attention = edited["attn.0"]
    assert attention.shape[2] == LOGICAL_BLOCK_SIZE, (
        "attention group still paged at the kernel block size; LMCache would "
        "misread it as slot-compressed"
    )
    for i in range(NUM_MAMBA_GROUPS):
        assert edited[f"mamba.{i}"].shape[2] == LOGICAL_BLOCK_SIZE


def test_edit_counts_include_both_rules(edits):
    """Both rule names must appear in the applied-edits counts.

    This log line is the only startup signal that a rule fired, so it is the
    one place a silently non-matching rule is observable.

    LMCache loggers set ``propagate = False``, so assert on the module logger
    directly rather than via the ``caplog`` (root-logger) fixture.
    """
    config, kv_caches = _hybrid_config(_fused_kv_cache())
    with patch.object(edits.logger, "info") as mock_info:
        edits.apply_kv_cache_group_edits(config, kv_caches, layout_hints={})

    counts = mock_info.call_args.args[1]
    assert counts == {
        "mamba-page-view": NUM_MAMBA_GROUPS,
        "subpaged-attention-view": 1,
    }


def test_every_group_reaches_block_id_granularity(edits):
    """The invariant edits exist to restore, across every group.

    Design doc: "After edits, every registered tensor's block dim equals its
    group's kv_cache_spec.block_size, so the server derives compress_ratio
    == 1 for these groups."
    """
    config, kv_caches = _hybrid_config(_fused_kv_cache())
    edited = edits.apply_kv_cache_group_edits(config, kv_caches, layout_hints={})

    for group in config.kv_cache_groups:
        for name in group.layer_names:
            assert edited[name].shape[2] == group.kv_cache_spec.block_size


def test_flash_attention_hybrid_needs_no_attention_edit(edits):
    """A backend paging at the logical size leaves only the Mamba edit.

    FlashAttention advertises ``MultipleOf(16)``, so its tensor arrives already
    paged at 1600 in both vLLM versions -- there is no 0.25.1 -> 0.26.0 delta
    on that path, and no attention edit to apply.
    """
    config, kv_caches = _hybrid_config(
        _fused_kv_cache(kernel_block_size=LOGICAL_BLOCK_SIZE)
    )
    edited = edits.apply_kv_cache_group_edits(config, kv_caches, layout_hints={})

    # Passed through untouched, and already at block-id granularity.
    assert edited["attn.0"] is kv_caches["attn.0"]
    assert edited["attn.0"].shape[2] == LOGICAL_BLOCK_SIZE


def test_mamba_page_view_handles_nonzero_storage_offset(edits):
    """Packed page view must not require conv to start at storage offset zero.

    vLLM-Ascend v0.30 places hybrid Mamba state after other layers in one pool;
    ``_MambaPageViewEdit.apply`` must re-stride from conv's offset without
    pulling in bytes before ``prefix_elems``.
    """
    prefix_elems = 64
    caches, page_bytes, storage = _mamba_kv_cache(prefix_elems)
    spec = _mamba_spec(page_bytes)
    assert edits._MambaPageViewEdit().matches(spec, caches)
    viewed = edits._MambaPageViewEdit().apply(spec, caches, {})

    elems_per_page = page_bytes // storage.element_size()
    assert viewed.dtype == storage.dtype
    assert tuple(viewed.shape) == (
        NUM_LOGICAL_BLOCKS,
        2,
        LOGICAL_BLOCK_SIZE,
        1,
        elems_per_page // (2 * LOGICAL_BLOCK_SIZE),
    )
    assert int(viewed.storage_offset()) == prefix_elems
    expected = storage.as_strided(
        (NUM_LOGICAL_BLOCKS, elems_per_page),
        (elems_per_page, 1),
        storage_offset=prefix_elems,
    )
    torch.testing.assert_close(viewed.reshape(NUM_LOGICAL_BLOCKS, -1), expected)
    assert int(viewed.reshape(-1)[0]) != -1


def _subpaged_kv_planes(
    *,
    prefix_elems: int = 0,
    interleaved: bool = True,
    contiguous: bool = False,
    num_kv_heads: int = 1,
    num_logical_blocks: int = 2,
    inner_stride: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, _Spec, torch.Tensor]:
    """Rank-4 K and V planes paged below the scheduler block.

    Shape ``(num_kernel_pages, kernel_block_size, num_kv_heads, head_size)``
    with ``kernel_block_size < spec.block_size``. ``interleaved=True`` is the
    vLLM-Ascend v0.30 pool: K's ``stride(0)`` covers one K row plus the
    following V row. ``contiguous=True`` is vLLM-Ascend <= v0.29: every K
    kernel page, then every V kernel page, each plane dense.
    ``interleaved=False`` gives a one-row stride that does not tile a
    logical page.
    """
    kernel_block_size = 8
    logical_block_size = 64
    ratio = logical_block_size // kernel_block_size
    head_size = 4
    num_kernel_pages = num_logical_blocks * ratio
    row_elems = kernel_block_size * num_kv_heads * head_size
    if contiguous:
        kernel_stride_elems = row_elems
        plane_count = 2
    else:
        kernel_stride_elems = 2 * row_elems if interleaved else row_elems
        plane_count = 1
    storage = torch.arange(
        prefix_elems + plane_count * num_kernel_pages * kernel_stride_elems,
        dtype=torch.bfloat16,
    )
    storage[:prefix_elems] = -1
    shape = (num_kernel_pages, kernel_block_size, num_kv_heads, head_size)
    if inner_stride is None:
        stride = (kernel_stride_elems, num_kv_heads * head_size, head_size, 1)
    else:
        stride = inner_stride
    k_plane = storage.as_strided(shape, stride, storage_offset=prefix_elems)
    if contiguous:
        v_offset = prefix_elems + num_kernel_pages * row_elems
        page_elems = 2 * ratio * row_elems
    else:
        v_offset = prefix_elems + (row_elems if interleaved else 0)
        page_elems = ratio * kernel_stride_elems
    v_plane = storage.as_strided(shape, stride, storage_offset=v_offset)
    spec = _Spec(
        kind=_SpecKind.FULL_ATTENTION,
        block_size=logical_block_size,
        page_size_bytes=page_elems * storage.element_size(),
    )
    return k_plane, v_plane, spec, storage


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_subpaged_plane_edit_fires_on_interleaved_kv(edits, num_kv_heads):
    """vLLM-Ascend v0.30 interleaved K/V planes must be re-paged.

    The fused rank-4/5 rule does not match a pair, so the interleaved rule
    is what restores the scheduler block size. The output is the same
    original-dtype rank-5 page as the fused edit (format 2), and the
    contiguous rule must not also match.
    """
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(num_kv_heads=num_kv_heads)
    pair = (k_plane, v_plane)
    interleaved = edits._SubpagedAttentionInterleavedPlaneViewEdit()
    contiguous = edits._SubpagedAttentionContiguousPlaneViewEdit()
    assert interleaved.matches(spec, pair)
    assert not contiguous.matches(spec, pair)
    viewed = _edit_attention(edits, pair, attention_spec=spec)
    assert isinstance(viewed, torch.Tensor)
    ratio = int(spec.block_size) // int(k_plane.shape[1])
    num_blocks = int(k_plane.shape[0]) // ratio
    synthetic_head_size = int(k_plane.shape[2]) * int(k_plane.shape[3])
    page_elems = int(spec.page_size_bytes) // int(viewed.element_size())
    assert viewed.dtype == k_plane.dtype
    assert viewed.ndim == 5
    assert tuple(viewed.shape) == (
        num_blocks,
        2,
        int(spec.block_size),
        1,
        synthetic_head_size,
    )
    assert tuple(viewed.stride()) == (
        page_elems,
        page_elems // 2,
        synthetic_head_size,
        synthetic_head_size,
        1,
    )
    assert viewed.untyped_storage().data_ptr() == k_plane.untyped_storage().data_ptr()
    assert int(viewed.storage_offset()) == int(k_plane.storage_offset())
    assert viewed.numel() == num_blocks * page_elems


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_subpaged_plane_edit_groups_contiguous_kernel_pages(edits, num_kv_heads):
    """Logical block ``n`` must cover kernel pages ``n*k .. n*k+k-1``.

    vLLM-Ascend v0.30 interleaves those pages in one pool, and elements before
    K's storage offset are not part of block 0.
    """
    prefix_elems = 17
    k_plane, v_plane, spec, storage = _subpaged_kv_planes(
        prefix_elems=prefix_elems, num_kv_heads=num_kv_heads
    )
    viewed = _edit_attention(edits, (k_plane, v_plane), attention_spec=spec)
    assert isinstance(viewed, torch.Tensor)

    page_elems = spec.page_size_bytes // storage.element_size()
    for block in range(viewed.shape[0]):
        start = prefix_elems + block * page_elems
        torch.testing.assert_close(
            viewed[block].reshape(-1), storage[start : start + page_elems]
        )
    assert int(viewed.storage_offset()) == prefix_elems
    assert int(storage[0]) != int(viewed.reshape(-1)[0])


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_interleaved_plane_edit_matches_fused_page_view(edits, num_kv_heads):
    """The interleaved pair must edit to the same tensor as the fused rule.

    Both inputs describe ``[K0][V0][K1][V1]...`` in one allocation. The fused
    rule sees one rank-5 kernel-paged tensor; the interleaved rule sees the
    K and V views of that tensor. Their logical-page views must be identical.
    """
    prefix_elems = 17
    k_plane, v_plane, spec, storage = _subpaged_kv_planes(
        prefix_elems=prefix_elems, num_kv_heads=num_kv_heads
    )
    num_kernel_pages = int(k_plane.shape[0])
    kernel_block_size = int(k_plane.shape[1])
    head_size = int(k_plane.shape[3])
    row_elems = kernel_block_size * num_kv_heads * head_size
    fused = storage.as_strided(
        (num_kernel_pages, 2, kernel_block_size, num_kv_heads, head_size),
        (2 * row_elems, row_elems, num_kv_heads * head_size, head_size, 1),
        storage_offset=prefix_elems,
    )
    fused_view = edits._SubpagedAttentionViewEdit().apply(spec, fused, {})
    interleaved_view = edits._SubpagedAttentionInterleavedPlaneViewEdit().apply(
        spec, (k_plane, v_plane), {}
    )
    assert tuple(interleaved_view.shape) == tuple(fused_view.shape)
    assert tuple(interleaved_view.stride()) == tuple(fused_view.stride())
    assert interleaved_view.dtype == fused_view.dtype
    assert int(interleaved_view.storage_offset()) == int(fused_view.storage_offset())
    assert (
        interleaved_view.untyped_storage().data_ptr()
        == fused_view.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(interleaved_view.reshape(-1), fused_view.reshape(-1))


def test_subpaged_plane_edit_rejects_non_dense_interleaved_inner_stride(edits):
    """Interleaved dim-0 stride is not enough without a dense NHD row."""
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(num_kv_heads=4)
    kernel_stride_elems = int(k_plane.stride(0))
    bad_stride = (kernel_stride_elems, 4, 8, 1)
    k_bad = k_plane.as_strided(k_plane.shape, bad_stride, k_plane.storage_offset())
    v_bad = v_plane.as_strided(v_plane.shape, bad_stride, v_plane.storage_offset())
    interleaved = edits._SubpagedAttentionInterleavedPlaneViewEdit()
    assert not interleaved.matches(spec, (k_bad, v_bad))


def test_subpaged_plane_edit_rejects_interleaved_separate_storage(edits):
    """K and V must share one pool; separate allocations are not interleaved."""
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(num_kv_heads=4)
    v_clone = v_plane.clone()
    assert v_clone.untyped_storage().data_ptr() != k_plane.untyped_storage().data_ptr()
    interleaved = edits._SubpagedAttentionInterleavedPlaneViewEdit()
    assert not interleaved.matches(spec, (k_plane, v_clone))


def test_subpaged_plane_edit_rejects_separate_kv_planes(edits):
    """Interleaved layout with a one-row stride must not match either plane rule.

    ``interleaved=False`` gives K/V strides that do not tile
    ``page_size_bytes``; neither Ascend plane edit nor a bare K list may match.
    """
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(interleaved=False)
    pair = (k_plane, v_plane)
    interleaved = edits._SubpagedAttentionInterleavedPlaneViewEdit()
    contiguous = edits._SubpagedAttentionContiguousPlaneViewEdit()
    assert not interleaved.matches(spec, pair)
    assert not contiguous.matches(spec, pair)
    assert not interleaved.matches(spec, [k_plane])
    assert not contiguous.matches(spec, [k_plane])


def test_subpaged_plane_edit_groups_contiguous_kernel_pages_per_plane(edits):
    """Logical block ``n`` covers kernel pages ``n*k .. n*k+k-1`` on each plane.

    vLLM-Ascend <= v0.29 lays those pages out as all K, then all V. Elements
    before K's storage offset are not part of block 0.
    """
    prefix_elems = 17
    k_plane, v_plane, spec, storage = _subpaged_kv_planes(
        prefix_elems=prefix_elems, contiguous=True
    )
    pair = (k_plane, v_plane)
    interleaved = edits._SubpagedAttentionInterleavedPlaneViewEdit()
    contiguous = edits._SubpagedAttentionContiguousPlaneViewEdit()
    assert not interleaved.matches(spec, pair)
    assert contiguous.matches(spec, pair)
    viewed = _edit_attention(edits, pair, attention_spec=spec)
    assert len(viewed) == 2
    assert viewed[0].dtype == k_plane.dtype
    assert viewed[1].dtype == v_plane.dtype
    assert viewed[0].shape[1] == spec.block_size
    k_ptr = k_plane.untyped_storage().data_ptr()
    assert viewed[0].untyped_storage().data_ptr() == k_ptr
    assert (
        viewed[1].untyped_storage().data_ptr() == v_plane.untyped_storage().data_ptr()
    )
    row_elems = int(k_plane[0].numel())
    ratio = int(spec.block_size) // int(k_plane.shape[1])
    span = ratio * row_elems
    v_base = prefix_elems + int(k_plane.shape[0]) * row_elems
    for block in range(viewed[0].shape[0]):
        k_start = prefix_elems + block * span
        v_start = v_base + block * span
        torch.testing.assert_close(
            viewed[0][block].reshape(-1), storage[k_start : k_start + span]
        )
        torch.testing.assert_close(
            viewed[1][block].reshape(-1), storage[v_start : v_start + span]
        )
    assert int(viewed[0].storage_offset()) == prefix_elems
    assert int(viewed[1].storage_offset()) == v_base
    assert int(storage[0]) != int(viewed[0].reshape(-1)[0])


def test_subpaged_plane_edit_accepts_separate_contiguous_allocations(edits):
    """Contiguous K/V in distinct storages still reshape at the scheduler block.

    ``v_plane.clone()`` forces a separate allocation; the contiguous-plane edit
    must still re-page both tensors in place.
    """
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(contiguous=True)
    pair = (k_plane, v_plane.clone())
    viewed = _edit_attention(edits, pair, attention_spec=spec)
    assert viewed[0].shape[1] == spec.block_size
    assert viewed[1].shape[1] == spec.block_size
    torch.testing.assert_close(viewed[1].reshape(-1), pair[1].reshape(-1))


def test_subpaged_plane_edit_accepts_mamba_padded_page(edits):
    """Conv padding on the spec page is not part of the contiguous K/V views.

    vLLM-Ascend sets ``page_size_padded`` so ``page_size_bytes`` equals the
    Mamba page. The reshape still has to run when the tensors equal
    ``real_page_size_bytes``.
    """
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(contiguous=True)
    spec.real_page_size_bytes = spec.page_size_bytes
    spec.page_size_bytes += 128
    viewed = _edit_attention(edits, (k_plane, v_plane), attention_spec=spec)
    assert viewed[0].shape[1] == spec.block_size
    assert viewed[1].shape[1] == spec.block_size
    assert viewed[0].dtype == k_plane.dtype


def test_subpaged_plane_edit_rejects_untiled_contiguous_kv(edits):
    """Inflated ``page_size_bytes`` without matching bytes must fail loudly.

    Doubling the spec page while leaving the tensors unchanged must not match,
    ``apply`` must raise, and the public entry point must pass the pair through.
    """
    k_plane, v_plane, spec, _storage = _subpaged_kv_planes(contiguous=True)
    spec.page_size_bytes *= 2
    pair = (k_plane, v_plane)
    edit = edits._SubpagedAttentionContiguousPlaneViewEdit()
    assert not edit.matches(spec, pair)
    assert not edits._SubpagedAttentionInterleavedPlaneViewEdit().matches(spec, pair)
    with pytest.raises(ValueError, match="do not tile the logical page"):
        edit.apply(spec, pair, {})
    assert _edit_attention(edits, pair, attention_spec=spec) is pair


def test_subpaged_plane_rules_are_ordered_and_exclusive(edits):
    """The fused rule precedes the Ascend plane rules, which stay exclusive.

    Registry order is fused, then interleaved, then contiguous. The fused
    rule only matches a single tensor, so each Ascend ``(K, V)`` fixture
    still matches exactly one plane rule.
    """
    names = [type(edit).__name__ for edit in edits._EDITS]
    interleaved = "_SubpagedAttentionInterleavedPlaneViewEdit"
    contiguous = "_SubpagedAttentionContiguousPlaneViewEdit"
    fused = "_SubpagedAttentionViewEdit"
    assert names.index(fused) < names.index(interleaved)
    assert names.index(interleaved) < names.index(contiguous)
    for contiguous_layout in (False, True):
        k_plane, v_plane, spec, _storage = _subpaged_kv_planes(
            contiguous=contiguous_layout
        )
        pair = (k_plane, v_plane)
        matches = [
            type(edit).__name__ for edit in edits._EDITS if edit.matches(spec, pair)
        ]
        expected = contiguous if contiguous_layout else interleaved
        assert matches == [expected]
