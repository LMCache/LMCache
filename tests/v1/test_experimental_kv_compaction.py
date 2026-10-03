# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import lmcache.integration.vllm.experimental.kv_compaction as kv_compaction
from lmcache.integration.vllm.experimental.kv_compaction import (
    apply_kv_compaction_moves,
)


def _paged_kv(num_blocks: int, slots_per_block: int, dtype=torch.float32):
    slots = torch.arange(num_blocks * slots_per_block, dtype=torch.float32)
    return (
        slots.view(num_blocks, slots_per_block, 1, 1).repeat(1, 1, 2, 3).to(dtype=dtype)
    )


def _overlap_copies(device="cpu"):
    return torch.tensor(
        [[19, 18], [7, 19], [13, 6]],
        dtype=torch.long,
        device=device,
    )


def _assert_overlap_result(kv_cache):
    compacted = [kv_cache[9, 0], kv_cache[9, 1], kv_cache[3, 0]]
    assert [entry.unique().item() for entry in compacted] == [19, 7, 13]


def test_applies_moves_to_destination_slots():
    kv_cache = _paged_kv(3, 4)
    apply_kv_compaction_moves(
        kv_cache,
        torch.tensor([[6, 0], [9, 1]], dtype=torch.long),
    )
    assert kv_cache[0, 0].unique().tolist() == [6]
    assert kv_cache[0, 1].unique().tolist() == [9]


def test_overlapping_copies_read_original_sources():
    kv_cache = _paged_kv(10, 2)
    apply_kv_compaction_moves(kv_cache, _overlap_copies())
    _assert_overlap_result(kv_cache)


def test_chunking_preserves_copy_order(monkeypatch):
    kv_cache = _paged_kv(10, 2)
    entry_bytes = kv_cache[0, 0].numel() * kv_cache.element_size()
    monkeypatch.setattr(
        kv_compaction,
        "_MAX_COMPACTION_PAYLOAD_BYTES",
        entry_bytes,
    )
    apply_kv_compaction_moves(kv_cache, _overlap_copies())
    _assert_overlap_result(kv_cache)


def test_trailing_payload_dimensions_move_together():
    kv_cache = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).view(2, 2, 2, 3)
    expected = kv_cache[1, 1].clone()
    apply_kv_compaction_moves(
        kv_cache,
        torch.tensor([[3, 0]], dtype=torch.long),
    )
    assert torch.equal(kv_cache[0, 0], expected)


def test_empty_moves_are_noop():
    kv_cache = _paged_kv(2, 2)
    before = kv_cache.clone()
    apply_kv_compaction_moves(
        kv_cache,
        torch.empty((0, 2), dtype=torch.long),
    )
    assert torch.equal(kv_cache, before)


def test_strided_view_updates_backing_storage():
    backing = torch.arange(10 * 2 * 2 * 3, dtype=torch.float32).view(10, 2, 2, 3)
    kv_cache = backing[:, 0]
    assert not kv_cache.is_contiguous()

    expected_backing = backing.clone()
    expected = expected_backing[:, 0]
    for source, destination in _overlap_copies().tolist():
        expected[destination // 2, destination % 2] = expected[
            source // 2,
            source % 2,
        ]

    apply_kv_compaction_moves(kv_cache, _overlap_copies())
    assert torch.equal(backing, expected_backing)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overlapping_copies_on_cuda(dtype):
    kv_cache = _paged_kv(10, 2, dtype=dtype).cuda()
    apply_kv_compaction_moves(kv_cache, _overlap_copies("cuda"))
    _assert_overlap_result(kv_cache)
