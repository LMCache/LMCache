# SPDX-License-Identifier: Apache-2.0
"""Kernel-paged unified-layout MLA caches are re-viewed at logical blocks.

GLM-5.3-Flash geometry on Hopper: a 1152-token block, the sparse MLA cache
kernel-paged at 64 rows and the kpool indexer (4 tokens per state) at 32
rows, both in one engine group.
"""

# Standard
from types import SimpleNamespace
from typing import cast
import inspect

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="group edits import vLLM specs")

# Third Party
from vllm.v1 import kv_cache_interface  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
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


def _supports_declared_slot_compression() -> bool:
    return "tokens_per_state" in inspect.signature(MLAAttentionSpec).parameters


def _padded_kwargs(page_size_padded: int | None) -> dict[str, int]:
    if page_size_padded is None:
        return {}
    if "page_size_padded" not in inspect.signature(MLAAttentionSpec).parameters:
        pytest.skip("MLAAttentionSpec does not expose page_size_padded")
    return {"page_size_padded": page_size_padded}


def _mla_spec(page_size_padded: int | None = None) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=MLA_WIDTH,
        head_size_v=0,
        dtype=torch.bfloat16,
        **_padded_kwargs(page_size_padded),
    )


def _indexer_spec(page_size_padded: int | None = None) -> MLAAttentionSpec:
    if not _supports_declared_slot_compression():
        pytest.skip("MLAAttentionSpec does not expose tokens_per_state")
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=INDEXER_WIDTH,
        head_size_v=0,
        dtype=torch.uint8,
        tokens_per_state=INDEXER_TOKENS_PER_STATE,
        **_padded_kwargs(page_size_padded),
    )


def _kernel_paged(num_states: int, kernel_rows: int, width: int, dtype) -> torch.Tensor:
    pages = NUM_BLOCKS * num_states // kernel_rows
    numel = pages * kernel_rows * width
    return torch.arange(numel).to(dtype).reshape(pages, 1, kernel_rows, width)


def _config(layer_specs: dict[str, KVCacheSpec]) -> KVCacheConfig:
    """Build a version-compatible group and enable the hybrid-layout edit gate."""
    if hasattr(kv_cache_interface, "KVCacheGroup"):
        # Third Party
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

        group = kv_cache_interface.KVCacheGroup(
            kv_cache_specs=layer_specs,
            manager_config=kv_cache_interface.ManagerConfig(
                FullAttentionManager, FullAttentionManager.Params()
            ),
            block_size=BLOCK_SIZE,
        )
    else:
        spec = (
            kv_cache_interface.UniformTypeKVCacheSpecs(
                block_size=BLOCK_SIZE, kv_cache_specs=layer_specs
            )
            if len(layer_specs) > 1
            else next(iter(layer_specs.values()))
        )
        group = SimpleNamespace(layer_names=list(layer_specs), kv_cache_spec=spec)
    return cast(
        KVCacheConfig,
        SimpleNamespace(kv_cache_groups=[group], has_mamba_layers=True),
    )


def _edit(spec: MLAAttentionSpec, kv_cache: torch.Tensor) -> torch.Tensor:
    config = _config({"l": spec})
    edited = apply_kv_cache_group_edits(config, {"l": kv_cache}, layout_hints={})["l"]
    assert isinstance(edited, torch.Tensor)
    return edited


def test_mla_kernel_pages_re_viewed_as_logical_block():
    raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH, torch.bfloat16)

    edited = _edit(_mla_spec(), raw)

    assert edited.shape == (NUM_BLOCKS, 1, BLOCK_SIZE, MLA_WIDTH)
    assert edited.data_ptr() == raw.data_ptr()
    ratio = BLOCK_SIZE // MLA_KERNEL_ROWS
    for block in range(NUM_BLOCKS):
        pages = raw[block * ratio : (block + 1) * ratio, 0]
        assert torch.equal(edited[block, 0], pages.reshape(BLOCK_SIZE, MLA_WIDTH))


@pytest.mark.parametrize("page_size_padded", [None, 524_288])
def test_indexer_keeps_declared_compression(page_size_padded: int | None):
    num_states = BLOCK_SIZE // INDEXER_TOKENS_PER_STATE
    raw = _kernel_paged(num_states, INDEXER_KERNEL_ROWS, INDEXER_WIDTH, torch.uint8)

    edited = _edit(_indexer_spec(page_size_padded=page_size_padded), raw)

    assert edited.shape == (NUM_BLOCKS, 1, num_states, INDEXER_WIDTH)
    assert edited.data_ptr() == raw.data_ptr()


def test_uniform_group_edits_each_layer_by_its_own_spec():
    mla_raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH, torch.bfloat16)
    num_states = BLOCK_SIZE // INDEXER_TOKENS_PER_STATE
    idx_raw = _kernel_paged(num_states, INDEXER_KERNEL_ROWS, INDEXER_WIDTH, torch.uint8)
    config = _config({"mla": _mla_spec(), "idx": _indexer_spec()})

    edited = apply_kv_cache_group_edits(
        config, {"mla": mla_raw, "idx": idx_raw}, layout_hints={}
    )

    assert edited["mla"].shape == (NUM_BLOCKS, 1, BLOCK_SIZE, MLA_WIDTH)
    assert edited["idx"].shape == (NUM_BLOCKS, 1, num_states, INDEXER_WIDTH)


def test_logical_block_cache_passes_through():
    raw = _kernel_paged(BLOCK_SIZE, BLOCK_SIZE, MLA_WIDTH, torch.bfloat16)

    assert _edit(_mla_spec(), raw) is raw


@pytest.mark.parametrize("page_size_padded", [None, 2_097_152])
def test_mismatched_page_bytes_rejected(page_size_padded: int | None):
    raw = _kernel_paged(BLOCK_SIZE, MLA_KERNEL_ROWS, MLA_WIDTH // 2, torch.bfloat16)

    with pytest.raises(ValueError, match="do not tile"):
        _edit(_mla_spec(page_size_padded=page_size_padded), raw)
