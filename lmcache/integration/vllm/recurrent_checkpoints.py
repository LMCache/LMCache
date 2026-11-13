# SPDX-License-Identifier: Apache-2.0
"""Register copied recurrent states in the vLLM prefix cache."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


def publish_recurrent_checkpoints(
    pool: BlockPool,
    request: Request,
    block_size: int,
    checkpoints: dict[int, dict[int, KVCacheBlock]],
) -> None:
    """Publish device checkpoints after every required worker copy is fenced.

    Args:
        pool: vLLM block pool that owns the private destination blocks.
        request: Request whose prefix hashes identify the checkpoints.
        block_size: Tokens per block, equal for all checkpoint groups.
        checkpoints: Aligned token positions mapped to every recurrent
            group and its copied destination block.

    The caller retains its block references and is responsible for freeing
    them. Any incomplete publication is removed before the error propagates.
    """
    if not checkpoints:
        return
    slots = [pool.null_block] * (max(checkpoints) // block_size)
    published = False
    try:
        for position, groups in checkpoints.items():
            index = position // block_size - 1
            for group, block in groups.items():
                slots[index] = block
                # A single-block range emits one event for the actual prefix
                # boundary instead of advertising the holes in a sparse row.
                pool.cache_full_blocks(
                    request, slots, index, index + 1, block_size, group
                )
        published = True
    finally:
        if not published:
            pool.evict_blocks(
                {
                    block.block_id
                    for row in checkpoints.values()
                    for block in row.values()
                }
            )
