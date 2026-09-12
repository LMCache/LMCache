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

        # Track membership at the queue boundary rather than extending the
        # policy interface. This keeps third-party OffloadPolicy subclasses
        # compatible while allowing callers to distinguish a legitimate
        # no-store request from a missing queued item.
        self._pending_request_ids: set[str] = set()

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
            self._pending_request_ids.add(meta.request_id)
        else:
            raise ValueError("gpu block pool not bound")

    def pop_items_for_offload(self) -> list[PendingStoreItem]:
        """Pop items from the queue when the policy's trigger is satisfied.

        An empty result means offload is not currently due.

        Returns:
            Pending store items to submit, or an empty list when no offload is
            due.
        """
        items = self._policy.pop_items_for_offload(self._select_count)
        self._pending_request_ids.difference_update(item.request_id for item in items)
        return items

    def has_inflight_store_work(self) -> bool:
        """Return whether submitted stores are waiting for worker completion.

        Returns:
            True if submitted stores are still holding GPU blocks while they
            wait for worker completion, otherwise False. Queued store metadata
            alone does not require engine keepalive because it can only be
            submitted by a step that schedules model tokens.
        """
        return bool(self._request_block_ids)

    def mark_req_finished(self, req_id: str) -> None:
        """Mark a queued request as finished in the offload policy.

        Args:
            req_id: Identifier of the completed request.

        Raises:
            ValueError: If the FIFO policy has no queued item for ``req_id``.
        """
        self._policy.mark_req_finished(req_id)

    def has_pending_request(self, req_id: str) -> bool:
        """Return whether a request has cache blocks pending offload.

        Args:
            req_id: Identifier of the request to inspect.

        Returns:
            True when the request has at least one queued store operation.
        """
        return req_id in self._pending_request_ids

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
