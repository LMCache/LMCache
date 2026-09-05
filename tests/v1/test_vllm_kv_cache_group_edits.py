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

These tests run on CPU and without vLLM installed: the module's vLLM imports
are stubbed before import, mirroring ``test_vllm_kv_cache_groups.py``.
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import TypeAlias
from unittest.mock import patch
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


@dataclass
class _Spec:
    """Stand-in for a vLLM ``KVCacheSpec`` leaf."""

    kind: _SpecKind
    block_size: int
    page_size_bytes: int
    mamba_cache_mode: str = "align"


@dataclass
class _Group:
    layer_names: list[str]
    kv_cache_spec: _Spec


@dataclass
class _Config:
    kv_cache_groups: list[_Group]
    has_mamba_layers: bool = True


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
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm_pkg,
            "vllm.v1": vllm_v1,
            "vllm.v1.kv_cache_interface": stub,
        },
    ):
        sys.modules.pop("lmcache.integration.vllm.kv_cache_group_edits", None)
        # First Party
        from lmcache.integration.vllm import kv_cache_group_edits

        yield kv_cache_group_edits
    sys.modules.pop("lmcache.integration.vllm.kv_cache_group_edits", None)


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


def _mamba_kv_cache() -> tuple[list[torch.Tensor], int]:
    """A Mamba ``[conv_state, ssm_state]`` pair sharing one padded page.

    Returns the pair and the page size in bytes. ``conv_state`` starts at the
    page base and its per-block stride spans the whole page (``conv | ssm |
    pad``), which is what ``_MambaPageViewEdit.apply`` re-strides over.
    """
    elem = torch.finfo(torch.bfloat16).bits // 8
    # Page must factor as (2, block_size, 1, head_size) for the synthetic view.
    elems_per_page = 2 * LOGICAL_BLOCK_SIZE * 3
    conv_elems, ssm_elems = 2 * LOGICAL_BLOCK_SIZE, 2 * LOGICAL_BLOCK_SIZE

    storage = torch.arange(NUM_LOGICAL_BLOCKS * elems_per_page, dtype=torch.bfloat16)
    conv_state = storage.as_strided(
        (NUM_LOGICAL_BLOCKS, conv_elems), (elems_per_page, 1)
    )
    ssm_state = storage.as_strided(
        (NUM_LOGICAL_BLOCKS, ssm_elems), (elems_per_page, 1), storage_offset=conv_elems
    )
    return [conv_state, ssm_state], elems_per_page * elem


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
    mamba_caches, mamba_page_bytes = _mamba_kv_cache()
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
