# SPDX-License-Identifier: Apache-2.0
"""Kernel-paged unified-layout MLA caches are re-viewed at logical blocks.

GLM-5.3-Flash geometry on Hopper: a 1152-token block, the sparse MLA cache
kernel-paged at 64 rows and the kpool indexer (4 tokens per state) at 32
rows, both in one engine group.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="group edits import vLLM specs")

# Third Party
from vllm.v1.kv_cache_interface import (  # noqa: E402
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    apply_kv_cache_group_edits,
)

BLOCK_SIZE = 1152
NUM_BLOCKS = 2
MLA_WIDTH = 512
MLA_KERNEL_ROWS = 64
INDEXER_WIDTH = 132
INDEXER_KERNEL_ROWS = 32
INDEXER_TOKENS_PER_STATE = 4


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=MLA_WIDTH,
        head_size_v=0,
        dtype=torch.bfloat16,
    )


def _indexer_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=INDEXER_WIDTH,
        head_size_v=0,
        dtype=torch.uint8,
        tokens_per_state=INDEXER_TOKENS_PER_STATE,
    )


def _kernel_paged(num_states: int, kernel_rows: int, width: int, dtype) -> torch.Tensor:
    pages = NUM_BLOCKS * num_states // kernel_rows
    numel = pages * kernel_rows * width
    return torch.arange(numel).to(dtype).reshape(pages, 1, kernel_rows, width)


def _config(groups: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(kv_cache_groups=groups, has_mamba_layers=True)


def _edit(spec, kv_cache: torch.Tensor) -> torch.Tensor:
    config = _config([SimpleNamespace(layer_names=["l"], kv_cache_spec=spec)])
    return apply_kv_cache_group_edits(config, {"l": kv_cache}, layout_hints={})["l"]


def test_mla_kernel_pages_re_viewed_as_logical_block():
    raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH, torch.bfloat16)

    edited = _edit(_mla_spec(), raw)

    assert edited.shape == (NUM_BLOCKS, 1, BLOCK_SIZE, MLA_WIDTH)
    assert edited.data_ptr() == raw.data_ptr()
    ratio = BLOCK_SIZE // MLA_KERNEL_ROWS
    for block in range(NUM_BLOCKS):
        pages = raw[block * ratio : (block + 1) * ratio, 0]
        assert torch.equal(edited[block, 0], pages.reshape(BLOCK_SIZE, MLA_WIDTH))


def test_indexer_keeps_declared_compression():
    num_states = BLOCK_SIZE // INDEXER_TOKENS_PER_STATE
    raw = _kernel_paged(num_states, INDEXER_KERNEL_ROWS, INDEXER_WIDTH, torch.uint8)

    edited = _edit(_indexer_spec(), raw)

    assert edited.shape == (NUM_BLOCKS, 1, num_states, INDEXER_WIDTH)
    assert edited.data_ptr() == raw.data_ptr()


def test_uniform_group_edits_each_layer_by_its_own_spec():
    mla_raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH, torch.bfloat16)
    num_states = BLOCK_SIZE // INDEXER_TOKENS_PER_STATE
    idx_raw = _kernel_paged(num_states, INDEXER_KERNEL_ROWS, INDEXER_WIDTH, torch.uint8)
    spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={"mla": _mla_spec(), "idx": _indexer_spec()},
    )
    config = _config([SimpleNamespace(layer_names=["mla", "idx"], kv_cache_spec=spec)])

    edited = apply_kv_cache_group_edits(
        config, {"mla": mla_raw, "idx": idx_raw}, layout_hints={}
    )

    assert edited["mla"].shape == (NUM_BLOCKS, 1, BLOCK_SIZE, MLA_WIDTH)
    assert edited["idx"].shape == (NUM_BLOCKS, 1, num_states, INDEXER_WIDTH)


def test_logical_block_cache_passes_through():
    raw = _kernel_paged(BLOCK_SIZE, BLOCK_SIZE, MLA_WIDTH, torch.bfloat16)

    assert _edit(_mla_spec(), raw) is raw


def test_mismatched_page_bytes_rejected():
    raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH // 2, torch.bfloat16)

    with pytest.raises(ValueError, match="do not tile"):
        _edit(_mla_spec(), raw)
