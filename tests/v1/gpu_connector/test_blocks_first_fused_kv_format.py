# SPDX-License-Identifier: Apache-2.0
"""Blocks-first, fused-K/V KV layout (GPUKVFormat.NL_X_NB_NH_BS_TWO_HS).

A non-MLA blocks-first attention backend registers its KV cache as the 4D
``[NB, NH, BS, 2 * HS]`` with K/V fused into the trailing dim (as opposed to
the 5D K/V-major ``[2, NB, NH, BS, HS]``). Discovery splits the fused axis into
the canonical 5D ``[NB, NH, BS, 2, HS]`` and classifies it as
``NL_X_NB_NH_BS_TWO_HS``.

These tests pin discovery, the format-aware accessors, and the multiprocess
gather/scatter round-trip for that layout.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
import lmcache.c_ops as lmc_ops

NB, NH, BS, HS, NL = 16, 4, 128, 64, 3
HINTS = {"kv_layout": "HND"}


def _raw_blocks_first_caches() -> list[torch.Tensor]:
    """Per-layer blocks-first tensors as registered: [NB, NH, BS, 2 * HS]."""
    torch.manual_seed(0)
    return [torch.randn(NB, NH, BS, 2 * HS) for _ in range(NL)]


def test_discovery_splits_fused_axis():
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_blocks_first_caches(), EngineType.VLLM, HINTS
    )
    assert fmt == lmc_ops.GPUKVFormat.NL_X_NB_NH_BS_TWO_HS
    # 4D [NB, NH, BS, 2*HS] -> canonical 5D [NB, NH, BS, 2, HS]
    assert tuple(norm[0].shape) == (NB, NH, BS, 2, HS)


def test_discovery_rejects_odd_trailing_dim():
    bad = [torch.randn(NB, NH, BS, 2 * HS + 1) for _ in range(NL)]
    with pytest.raises(ValueError):
        U.normalize_kv_and_discover_format(bad, EngineType.VLLM, HINTS)


def test_accessors():
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_blocks_first_caches(), EngineType.VLLM, HINTS
    )
    assert U.get_num_layers(norm, fmt) == NL
    assert U.get_num_blocks(norm, fmt) == NB
    assert U.get_block_size(norm, fmt) == BS
    assert U.get_num_heads(norm, fmt) == NH
    assert U.get_head_size(norm, fmt) == HS
    assert U.get_hidden_dim_size(norm, fmt) == NH * HS
    assert U.get_page_buffer_size(norm, fmt) == NB * BS
    assert U.get_tokens_per_layer(norm, fmt) == NB * BS
    assert U.get_elements_per_layer(norm, fmt) == NB * NH * BS * HS * 2
    # get_dtype is on the register_kv_caches -> group_layers_by_identity path,
    # so it must recognize this format too.
    assert U.get_dtype(norm, fmt) == _raw_blocks_first_caches()[0].dtype
    assert U.is_hnd(fmt) is True
    assert not U.is_mla(fmt)


def test_mp_gather_scatter_roundtrip():
    blocks_per_chunk = 2
    block_ids = [0, 3, 5, 6]  # 2 chunks
    raw = _raw_blocks_first_caches()
    src = {f"layer_{i}": t for i, t in enumerate(raw)}
    ref = {k: v.clone() for k, v in src.items()}
    idx = torch.tensor(block_ids)

    chunks = gather_paged_kv_to_cpu(
        src, block_ids, blocks_per_chunk, layout_hints=HINTS
    )
    # [K/V, NL, chunk_tokens, NH*HS]
    assert tuple(chunks[0].shape) == (2, NL, blocks_per_chunk * BS, NH * HS)

    # Wipe the gathered blocks, scatter back, and confirm exact recovery.
    dst = {k: v.clone() for k, v in src.items()}
    for k in dst:
        dst[k][idx] = 0.0
    scatter_cpu_to_paged_kv(
        dst, block_ids, chunks, blocks_per_chunk, layout_hints=HINTS
    )

    for k in dst:
        assert torch.equal(dst[k][idx], ref[k][idx])

    # Untouched blocks must be left alone.
    untouched = torch.tensor([b for b in range(NB) if b not in block_ids])
    for k in dst:
        assert torch.equal(dst[k][untouched], ref[k][untouched])
