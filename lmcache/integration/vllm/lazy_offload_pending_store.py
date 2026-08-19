# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections import defaultdict
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


class LazyOffloadPendingStore:
    """
    Buffering store operations in lazy offload mode.

    Store metadata is accumulated here instead of being immediately submitted.
    When the offload policy decides it's time, a batch of items is drained
    and returned for submission.
    """

    def __init__(
        self,
        configs: dict | None = None,
    ):
        """
        Initialize the pending store queue.

        Args:
            configs: The configuration for the pending store.
        """
        self._policy = self._create_offload_policy(configs)

        # TODO(chunxiaozheng): support more flexible select count
        self._select_count = (
            configs.get("lmcache.mp.lazy_offload_select_count", 10) if configs else 10
        )

        # TODO(chunxiaozheng): use gpu block pool to guide item selection.
        # GPU block pool reference
        self._gpu_block_pool: "BlockPool | None" = None

        # save all request block ids for free
        self._request_block_ids: dict[str, list[int]] = defaultdict(list)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the GPU block pool to the pending store."""
        self._gpu_block_pool = gpu_block_pool

    def add(self, meta: "LMCacheMPRequestMetadata") -> None:
        """Add a pending store meta to the pending store."""
        if self._gpu_block_pool:
            block_hashes = {
                bid: self._gpu_block_pool.blocks[bid].block_hash
                for bid in meta.op.flat_block_ids
            }
            self._policy.add(meta, block_hashes)
        else:
            raise ValueError("gpu block pool not bound")

    def pop_items_for_offload(self) -> list[PendingStoreItem]:
        """Pop items from the queue when the policy's trigger is satisfied.

        An empty result means offload is not currently due.

        Returns:
            Pending store items to submit, or an empty list when no offload is
            due.
        """
        return self._policy.pop_items_for_offload(self._select_count)

    def mark_req_finished(self, req_id: str):
        self._policy.mark_req_finished(req_id)

    def update_request_gpu_block_ids(self, req_id: str, block_ids: list[int]):
        self._request_block_ids[req_id].extend(block_ids)

    def get_request_gpu_block_ids(self, req_id: str) -> list[int]:
        return self._request_block_ids[req_id]

    def remove_request_gpu_block_ids(self, req_id: str):
        if req_id in self._request_block_ids:
            del self._request_block_ids[req_id]

    def _create_offload_policy(self, configs: dict | None) -> OffloadPolicy:
        """Create the configured lazy-offload policy."""
        policy = (
            configs.get("lmcache.mp.lazy_offload_policy", "FIFO") if configs else "FIFO"
        )
        if policy == "FIFO":
            return FIFOOffloadPolicy(configs)
        raise ValueError(f"Unknown offload policy: {policy}")
