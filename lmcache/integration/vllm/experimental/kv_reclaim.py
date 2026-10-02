# SPDX-License-Identifier: Apache-2.0
"""Reclaim paged KV blocks after in-place compaction."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _steps_to_next_block(num_tokens: int, block_size: int) -> int:
    return block_size - ((num_tokens - 1) % block_size)


def reclaim_compacted_kv_blocks(
    blocks: "KVCacheBlocks",
    block_pool: "BlockPool",
    *,
    logical_kv_length: int,
    physical_kv_length: int,
    block_size: int,
) -> list[int]:
    """Reclaim unused blocks while preserving vLLM's allocation cadence."""
    if physical_kv_length <= 0 or physical_kv_length > logical_kv_length:
        raise ValueError("Physical KV length must be in (0, logical KV length]")
    if block_size <= 0:
        raise ValueError("Block size must be positive")
    if len(blocks.blocks) != 1:
        raise ValueError("KV compaction requires exactly one cache group")

    row = blocks.blocks[0]
    if not isinstance(row, list):
        raise ValueError("KV compaction requires a mutable vLLM block row")

    logical_blocks = _cdiv(logical_kv_length, block_size)
    if len(row) != logical_blocks:
        raise ValueError("KV compaction requires vanilla full-attention allocation")

    real = [(index, block) for index, block in enumerate(row) if not block.is_null]
    if any(block.ref_cnt != 1 for _, block in real):
        raise ValueError("KV compaction requires private blocks")

    live_blocks = _cdiv(physical_kv_length, block_size)

    # Logical and physical block boundaries can have different phases after
    # compaction. Keep one spare only when the physical boundary arrives first;
    # vanilla vLLM replenishes it at the next logical boundary.
    keep_headroom = int(
        _steps_to_next_block(physical_kv_length, block_size)
        < _steps_to_next_block(logical_kv_length, block_size)
    )
    keep_blocks = live_blocks + keep_headroom
    if len(real) < keep_blocks:
        raise ValueError("Not enough resident blocks for the physical KV view")

    freed = [block for _, block in real[keep_blocks:]]
    for index, _ in real[keep_blocks:]:
        row[index] = block_pool.null_block
    block_pool.free_blocks(reversed(freed))

    return [block.block_id for _, block in real[:keep_blocks]]
