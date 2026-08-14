# SPDX-License-Identifier: Apache-2.0
"""FIFO lazy-offload policy."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class FIFOOffloadPolicy(OffloadPolicy):
    """Offload finished pending requests in first-in, first-out order."""

    def __init__(self, configs: dict | None = None) -> None:
        """Initialize the policy.

        Args:
            configs: Optional lazy-offload configuration. The
                ``lmcache.mp.lazy_offload_threshold`` value controls how many
                finished requests trigger an offload.
        """
        self._pending_items: dict[str, PendingStoreItem] = {}
        self._threshold = (
            configs.get("lmcache.mp.lazy_offload_threshold", 100) if configs else 100
        )
        self._finished_requests_count = 0
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: dict[int, bytes],
    ) -> None:
        """Queue cache blocks, aggregating multiple entries per request.

        Args:
            meta: Store metadata for a subset of a request's cache blocks.
            block_hashes: Mapping from queued GPU block IDs to block hashes.
        """
        if meta.request_id not in self._pending_items:
            self._pending_items[meta.request_id] = PendingStoreItem(
                request_id=meta.request_id
            )
        self._pending_items[meta.request_id].metadatas.append((meta, block_hashes))

    def mark_req_finished(self, req_id: str) -> bool:
        """Mark a queued request as ready for FIFO offload.

        Args:
            req_id: Identifier of the request that has completed.

        Returns:
            True if the request has queued cache blocks; False if it queued
            none, which happens for a request shorter than one chunk.
        """
        if req_id not in self._pending_items:
            return False
        self._pending_items[req_id].is_finished = True
        self._finished_requests_count += 1
        return True

    def drop_request(self, req_id: str) -> int:
        """Discard a request's queued cache blocks without offloading them.

        Args:
            req_id: Identifier of the request being dropped.

        Returns:
            The number of queued metadata entries discarded.
        """
        item = self._pending_items.pop(req_id, None)
        if item is None:
            return 0
        if item.is_finished:
            self._finished_requests_count -= 1
        return len(item.metadatas)

    def reclaim_finished_request(self, req_id: str) -> bool:
        """Discard a finished predecessor's item when its id is reused.

        Args:
            req_id: The reused request identifier.

        Returns:
            True if a finished item was discarded; False otherwise.
        """
        item = self._pending_items.get(req_id)
        if item is None or not item.is_finished:
            return False
        del self._pending_items[req_id]
        self._finished_requests_count -= 1
        return True

    def pop_items_for_offload(self, count: int) -> list[PendingStoreItem]:
        """Return up to ``count`` finished requests in insertion order.

        Args:
            count: Maximum number of pending items to pop.

        Returns:
            Finished pending items when the threshold is reached; otherwise an
            empty list.
        """
        if count <= 0 or self._finished_requests_count < self._threshold:
            return []

        to_offload = []
        for req_id in list(self._pending_items.keys()):
            if self._pending_items[req_id].is_finished:
                to_offload.append(self._pending_items[req_id])
                del self._pending_items[req_id]
                self._finished_requests_count -= 1
            if len(to_offload) >= count:
                break
        return to_offload
