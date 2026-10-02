# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.kv_reclaim import (
    reclaim_compacted_kv_blocks,
)


@dataclass
class _Block:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False


class _Pool:
    def __init__(self) -> None:
        self.null_block = _Block(0, ref_cnt=0, is_null=True)
        self.freed: list[int] = []

    def free_blocks(self, blocks) -> None:
        self.freed.extend(block.block_id for block in blocks)


class _Blocks:
    def __init__(self, row: list[_Block]) -> None:
        self.blocks = (row,)


def _row(num_blocks: int) -> list[_Block]:
    return [_Block(i + 1) for i in range(num_blocks)]


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def test_reclaims_tail_without_extra_headroom() -> None:
    pool = _Pool()
    blocks = _Blocks(_row(8))

    physical_ids = reclaim_compacted_kv_blocks(
        blocks,
        pool,
        logical_kv_length=128,
        physical_kv_length=24,
        block_size=16,
    )

    assert physical_ids == [1, 2]
    assert [block.block_id for block in blocks.blocks[0]] == [1, 2, 0, 0, 0, 0, 0, 0]
    assert set(pool.freed) == {3, 4, 5, 6, 7, 8}


def test_keeps_one_block_when_physical_boundary_arrives_first() -> None:
    pool = _Pool()
    blocks = _Blocks(_row(7))

    physical_ids = reclaim_compacted_kv_blocks(
        blocks,
        pool,
        logical_kv_length=104,
        physical_kv_length=32,
        block_size=16,
    )

    # Physical KV needs its next block on the next token, while vanilla vLLM
    # will not append a logical block for nine tokens.
    assert physical_ids == [1, 2, 3]
    assert [block.block_id for block in blocks.blocks[0]] == [1, 2, 3, 0, 0, 0, 0]
    assert set(pool.freed) == {4, 5, 6, 7}


def test_one_block_headroom_covers_every_block_phase() -> None:
    block_size = 16

    for logical_phase in range(block_size):
        for physical_phase in range(block_size):
            logical = block_size * 20 + logical_phase
            physical = block_size * 4 + physical_phase
            if logical_phase == 0:
                logical = block_size * 20
            if physical_phase == 0:
                physical = block_size * 4

            pool = _Pool()
            row = _row(_cdiv(logical, block_size))
            blocks = _Blocks(row)
            reclaim_compacted_kv_blocks(
                blocks,
                pool,
                logical_kv_length=logical,
                physical_kv_length=physical,
                block_size=block_size,
            )

            next_block_id = 1000
            for _ in range(block_size * 4):
                logical += 1
                physical += 1

                required_logical_blocks = _cdiv(logical, block_size)
                while len(row) < required_logical_blocks:
                    row.append(_Block(next_block_id))
                    next_block_id += 1

                real_blocks = sum(not block.is_null for block in row)
                needed = _cdiv(physical, block_size)
                assert needed <= real_blocks <= needed + 1


def test_rejects_shared_blocks() -> None:
    pool = _Pool()
    row = _row(4)
    row[1].ref_cnt = 2

    with pytest.raises(ValueError, match="private blocks"):
        reclaim_compacted_kv_blocks(
            _Blocks(row),
            pool,
            logical_kv_length=64,
            physical_kv_length=32,
            block_size=16,
        )
