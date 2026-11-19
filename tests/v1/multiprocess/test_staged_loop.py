# SPDX-License-Identifier: Apache-2.0
"""The per-batch staged loop: which chunks it stages, which block ids each
kernel launch gets, and the order of staging and kernels. The engine and
the kernel are faked, so these run without CUDA."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import staged
from lmcache.v1.multiprocess.object_group_transfer.engines import TransferEngine
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    resolve_chunk_range,
)
import lmcache.lmcache_native as lmcache_native

CHUNK = 16
BLOCK = 4
BLOCKS_PER_CHUNK = CHUNK // BLOCK
SLOTS = 2
H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H


@pytest.fixture
def harness(monkeypatch):
    """A cache context with ``SLOTS`` staging slots and one kernel group,
    plus a log of ``("stage", chunk names, slots)`` and
    ``("kernel", block ids, blocks to skip)`` entries."""
    log: list[tuple] = []
    cc = MagicMock()
    cc.lmcache_tokens_per_chunk = CHUNK
    cc.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0])],
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda og: True),
        get_subchunk_sw_size_tokens=lambda kg: CHUNK,
    )
    cc.calculate_num_blocks.side_effect = lambda tokens, kg: tokens // BLOCK
    cc.get_temp_object_group_buffer.side_effect = lambda slot, og: slot

    def fake_stage(engine, memory_objs, buffers, direction):
        log.append(("stage", [obj.name for obj in memory_objs], list(buffers)))

    def fake_kernel(pointers, buffers, block_ids, device, direction, *rest):
        log.append(("kernel", block_ids.tolist(), rest[-1]))

    ops = MagicMock()
    ops.multi_layer_block_kv_transfer.side_effect = fake_kernel
    monkeypatch.setattr(staged, "device_ops", ops)
    monkeypatch.setattr(staged, "stage", fake_stage)
    return cc, log


def _objects(count: int) -> list[MemoryObj | None]:
    objs: list[MemoryObj | None] = []
    for index in range(count):
        obj = MagicMock(spec=MemoryObj)
        obj.name = f"c{index}"
        objs.append(obj)
    return objs


def _run(cc, objs, direction, skip_first_n_tokens=0, chunk_range=None) -> None:
    staged.run_staged_loop(
        cc,
        [torch.arange(len(objs) * BLOCKS_PER_CHUNK)],
        objs,
        0,
        SLOTS,
        skip_first_n_tokens,
        direction,
        engine=TransferEngine.NVME_DMA,
        chunk_range=chunk_range,
    )


def test_retrieve_stages_each_batch_before_its_kernel(harness):
    cc, log = harness
    _run(cc, _objects(3), H2D)
    assert log == [
        ("stage", ["c0", "c1"], [0, 1]),
        ("kernel", list(range(0, 8)), 0),
        ("stage", ["c2"], [0]),
        ("kernel", list(range(8, 12)), 0),
    ]


def test_store_stages_each_batch_after_its_kernel(harness):
    cc, log = harness
    _run(cc, _objects(2), D2H)
    assert log == [
        ("kernel", list(range(0, 8)), 0),
        ("stage", ["c0", "c1"], [0, 1]),
    ]


def test_chunk_range_limits_the_chunks_and_their_block_ids(harness):
    cc, log = harness
    _run(cc, _objects(6), H2D, chunk_range=(3, 5))
    assert log == [
        ("stage", ["c3", "c4"], [0, 1]),
        ("kernel", list(range(12, 20)), 0),
    ]


def test_retrieve_does_not_stage_chunks_the_token_skip_covers(harness):
    """The skip covers chunk 0 and one block of chunk 1: chunk 0 is not
    staged, chunk 1 keeps its slot, and the kernel skips five blocks."""
    cc, log = harness
    _run(cc, _objects(4), H2D, skip_first_n_tokens=CHUNK + BLOCK)
    assert log == [
        ("stage", ["c1"], [1]),
        ("kernel", list(range(0, 8)), BLOCKS_PER_CHUNK + 1),
        ("stage", ["c2", "c3"], [0, 1]),
        ("kernel", list(range(8, 16)), 0),
    ]


def test_batch_entirely_before_the_token_skip_is_not_transferred(harness):
    cc, log = harness
    _run(cc, _objects(4), H2D, skip_first_n_tokens=2 * CHUNK)
    assert log == [
        ("stage", ["c2", "c3"], [0, 1]),
        ("kernel", list(range(8, 16)), 0),
    ]


def test_missing_object_raises_on_retrieve_and_skips_its_batch_on_store(harness):
    cc, log = harness
    objs = _objects(4)
    objs[1] = None
    with pytest.raises(ValueError):
        _run(cc, objs, H2D)
    assert log == []
    _run(cc, objs, D2H)
    assert log == [
        ("kernel", list(range(8, 16)), 0),
        ("stage", ["c2", "c3"], [0, 1]),
    ]


@pytest.mark.parametrize(
    "chunk_range, window_skip, expected",
    [
        (None, 0, (0, 8)),
        (None, 3, (3, 8)),
        ((2, 6), 0, (2, 6)),
        ((2, 6), 4, (4, 6)),
        ((5, 8), 2, (5, 8)),
    ],
)
def test_resolve_chunk_range_combines_the_range_with_the_window_skip(
    chunk_range, window_skip, expected
):
    assert resolve_chunk_range(8, chunk_range, window_skip) == expected
