# SPDX-License-Identifier: Apache-2.0
"""Allocate an entire history plan without evicting already-local checkpoints."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.recurrent_destinations import RecurrentDestinations
from lmcache.v1.transport.recurrent_history import HistoryResponse

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


def allocate_recurrent_destinations(
    pool: BlockPool,
    request: Request,
    block_size: int,
    plan: HistoryResponse,
    local_pins: list[KVCacheBlock],
) -> RecurrentDestinations:
    """Allocate private blocks for every checkpoint in the server's plan.

    Args:
        pool: Scheduler's block pool.
        request: Request whose primary restore triggered this plan.
        block_size: Common hash and scheduler block size.
        plan: Complete checkpoint positions and recurrent engine groups.
        local_pins: Existing complete local states to protect during allocation.

    Returns:
        The destinations and references owned until the copy is fenced.

    Raises:
        MemoryError: The whole plan does not fit alongside local pins.
    """
    pool.touch(local_pins)
    needed = len(plan.positions) * len(plan.groups)
    try:
        if needed > pool.get_num_free_blocks():
            raise MemoryError("All recurrent history checkpoints do not fit")
        blocks = iter(pool.get_new_blocks(needed))
    except (MemoryError, ValueError):
        pool.free_blocks(local_pins)
        raise
    destinations = {
        pos: {group: next(blocks) for group in plan.groups} for pos in plan.positions
    }
    return RecurrentDestinations(
        pool, request, block_size, plan, destinations, local_pins
    )
