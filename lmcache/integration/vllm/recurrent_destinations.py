# SPDX-License-Identifier: Apache-2.0
"""Private device blocks owned by a pending recurrent history restore."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.recurrent_checkpoints import publish_recurrent_checkpoints
from lmcache.v1.transport.recurrent_history import HistoryResponse

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


@dataclass
class RecurrentDestinations:
    """Retain private destinations and local checkpoint pins through the copy."""

    pool: BlockPool
    request: Request
    block_size: int
    plan: HistoryResponse
    blocks: dict[int, dict[int, KVCacheBlock]]
    local_pins: list[KVCacheBlock]
    canceled: bool = False
    released: bool = False
    receive_completed: bool = False

    def block_ids(self) -> dict[int, dict[int, int]]:
        """Return destination block IDs suitable for the worker copy request."""
        return {
            pos: {group: block.block_id for group, block in row.items()}
            for pos, row in self.blocks.items()
        }

    def settle(self, outcome: HistoryResponse) -> bool:
        """Publish successful copies and free references after the server fence.

        Args:
            outcome: Server outcome for this destination plan.

        Returns:
            Whether the destination references have been released.
        """
        if self.released:
            return True
        if not outcome.settled or outcome.lease != self.plan.lease:
            return False
        try:
            if (
                outcome.successful
                and not self.canceled
                and not self.request.is_finished()
            ):
                publish_recurrent_checkpoints(
                    self.pool, self.request, self.block_size, self.blocks
                )
        finally:
            self.pool.free_blocks(
                [block for row in self.blocks.values() for block in row.values()]
            )
            self.pool.free_blocks(self.local_pins)
            self.released = True
        return True
