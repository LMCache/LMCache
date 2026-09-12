# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sub-paged attention edits (GPU-free).

The rules decide purely from a vLLM spec and a registered tensor, so real
specs plus plain CPU tensors cover them; only the ``KVCacheConfig`` is a
duck-typed double, since the edits read just two of its attributes.
"""

# Standard
from dataclasses import dataclass
from typing import Any, cast

# Third Party
import pytest
import torch

kv_iface = pytest.importorskip(
    "vllm.v1.kv_cache_interface",
    reason="kv_cache_group_edits imports vLLM at module top",
)

# First Party
# First Party (after importorskip)
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    apply_kv_cache_group_edits,
)
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]
    has_mamba_layers: bool = True


# One observed hybrid configuration (Qwen3.5 GDN on vLLM 0.27.1): the
# scheduler inflates the attention block to 1600 tokens to align with the
# Mamba page while the backend pages the tensor at 64, so one logical block
# spans 25 kernel pages.
_LOGICAL_BLOCK_SIZE = 1600
_KERNEL_BLOCK_SIZE = 64
_RATIO = _LOGICAL_BLOCK_SIZE // _KERNEL_BLOCK_SIZE
_NUM_LOGICAL_BLOCKS = 3
_NUM_KERNEL_PAGES = _NUM_LOGICAL_BLOCKS * _RATIO
_NUM_HEADS = 4
_HEAD_SIZE = 128
_DTYPE = torch.bfloat16


def _fa_spec(page_size_padded: int | None = None) -> Any:
    """A full-attention spec whose page holds one whole logical block.

    ``page_size_bytes`` is derived by vLLM as ``num_kv_heads * block_size *
    2 * head_size * itemsize``; ``page_size_padded`` overrides it, which the
    byte-accounting tests use to simulate a page the kernel pages cannot tile.
    """
    return kv_iface.FullAttentionSpec(
        block_size=_LOGICAL_BLOCK_SIZE,
        num_kv_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        dtype=_DTYPE,
        page_size_padded=page_size_padded,
    )


def _split_kv_cache() -> torch.Tensor:
    """The rank-5 layout of vLLM < 0.26: K and V get their own axis."""
    return torch.zeros(
        _NUM_KERNEL_PAGES,
        2,
        _KERNEL_BLOCK_SIZE,
        _NUM_HEADS,
        _HEAD_SIZE,
        dtype=_DTYPE,
    )


def _packed_kv_cache(kv_layout: str = "NHD") -> torch.Tensor:
    """The rank-4 layout of vLLM >= 0.26: K/V packed in the content axis."""
    heads_first = kv_layout == "HND"
    return torch.zeros(
        _NUM_KERNEL_PAGES,
        _NUM_HEADS if heads_first else _KERNEL_BLOCK_SIZE,
        _KERNEL_BLOCK_SIZE if heads_first else _NUM_HEADS,
        2 * _HEAD_SIZE,
        dtype=_DTYPE,
    )


def _edit(
    kv_cache: torch.Tensor,
    spec: Any | None = None,
    kv_layout: str = "NHD",
) -> torch.Tensor:
    """Run the registry over one layer and return its (possibly edited) view."""
    config = MockKVCacheConfig(
        kv_cache_groups=[
            MockKVCacheGroup(layer_names=["layer.0"], kv_cache_spec=spec or _fa_spec())
        ]
    )
    edited = apply_kv_cache_group_edits(
        config, {"layer.0": kv_cache}, cast(LayoutHints, {"kv_layout": kv_layout})
    )
    return edited["layer.0"]


def test_page_size_bytes_matches_one_logical_block() -> None:
    """Pin the premise the other tests build on."""
    expected = 2 * _LOGICAL_BLOCK_SIZE * _NUM_HEADS * _HEAD_SIZE * _DTYPE.itemsize
    assert _fa_spec().page_size_bytes == expected


def test_subpaged_split_layout_is_reviewed_at_logical_block_size() -> None:
    """The rank-5 layout re-views to one page per scheduler block."""
    kv_cache = _split_kv_cache()

    edited = _edit(kv_cache)

    assert edited.shape[0] == _NUM_LOGICAL_BLOCKS
    assert edited.shape[2] == _LOGICAL_BLOCK_SIZE
    assert edited.data_ptr() == kv_cache.data_ptr()
    assert edited.numel() == kv_cache.numel()


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_subpaged_packed_layout_is_reviewed_at_logical_block_size(
    kv_layout: str,
) -> None:
    """The rank-4 K/V-packed layout re-views the same way, on both layouts.

    Regression test for the sub-paged full-attention group being left
    unedited on vLLM >= 0.26: the raw kernel geometry then reaches
    ``lmcache.v1.kv_layer_groups``, which reads it as slot compression and
    transfers one of every ``ratio`` kernel pages.
    """
    kv_cache = _packed_kv_cache(kv_layout)

    edited = _edit(kv_cache, kv_layout=kv_layout)

    token_axis = 1 if kv_layout == "NHD" else 2
    head_axis = 2 if kv_layout == "NHD" else 1
    assert edited.ndim == 4
    assert edited.shape[0] == _NUM_LOGICAL_BLOCKS
    assert edited.shape[token_axis] == _LOGICAL_BLOCK_SIZE
    assert edited.shape[head_axis] == 1
    # A whole logical page per block: nothing dropped, same storage.
    page_bytes = edited.shape[1:].numel() * edited.element_size()
    assert page_bytes == _fa_spec().page_size_bytes
    assert edited.data_ptr() == kv_cache.data_ptr()
    assert edited.numel() == kv_cache.numel()


def test_packed_layout_already_at_block_size_is_untouched() -> None:
    """A layer the backend did not re-page must not be edited."""
    kv_cache = torch.zeros(
        _NUM_LOGICAL_BLOCKS,
        _LOGICAL_BLOCK_SIZE,
        _NUM_HEADS,
        2 * _HEAD_SIZE,
        dtype=_DTYPE,
    )

    assert _edit(kv_cache) is kv_cache


def test_packed_layout_that_does_not_tile_the_page_raises() -> None:
    """An undeclared packed layout must fail loudly, not transfer wrongly."""
    padded = _fa_spec().page_size_bytes + _DTYPE.itemsize
    with pytest.raises(ValueError, match="do not tile the logical page"):
        _edit(_packed_kv_cache(), spec=_fa_spec(page_size_padded=padded))


def test_packed_layout_without_a_layout_hint_raises() -> None:
    """The token axis is unknowable without the hint, so refuse to guess."""
    with pytest.raises(ValueError, match="Unsupported kv_layout"):
        _edit(_packed_kv_cache(), kv_layout="none")


def test_non_mamba_config_is_returned_unedited() -> None:
    """The registry is only consulted for Mamba-hybrid models."""
    kv_cache = _packed_kv_cache()
    config = MockKVCacheConfig(
        kv_cache_groups=[
            MockKVCacheGroup(layer_names=["layer.0"], kv_cache_spec=_fa_spec())
        ],
        has_mamba_layers=False,
    )

    edited = apply_kv_cache_group_edits(
        config, {"layer.0": kv_cache}, {"kv_layout": "NHD"}
    )

    assert edited["layer.0"] is kv_cache


def test_mamba_layer_is_not_claimed_by_the_packed_rule() -> None:
    """Rank-4 Mamba state keeps matching ``mamba-unified-view`` first."""
    kv_cache = torch.zeros(
        _NUM_LOGICAL_BLOCKS, 1, 1, _LOGICAL_BLOCK_SIZE * _HEAD_SIZE, dtype=_DTYPE
    )
    spec = kv_iface.MambaSpec(
        block_size=_LOGICAL_BLOCK_SIZE,
        shapes=((_LOGICAL_BLOCK_SIZE, _HEAD_SIZE),),
        dtypes=(_DTYPE,),
        mamba_cache_mode="align",
    )

    edited = _edit(kv_cache, spec=spec)

    assert edited.shape == (_NUM_LOGICAL_BLOCKS, _LOGICAL_BLOCK_SIZE, 1, _HEAD_SIZE)
