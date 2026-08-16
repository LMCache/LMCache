# SPDX-License-Identifier: Apache-2.0
"""FIFO lazy-offload drain policy."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import PendingStoreItem
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class FIFOOffloadPolicy:
    """Buffer by request and drain controller-eligible ids in FIFO order."""

    def __init__(self, configs: dict | None = None) -> None:
        self._pending_items: dict[str, PendingStoreItem] = {}
        self._threshold = (
            configs.get("lmcache.mp.lazy_offload_threshold", 100) if configs else 100
        )
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: dict[int, bytes],
        epoch: int = 0,
    ) -> None:
        """Queue one metadata chunk under its request epoch."""
        item = self._pending_items.get(meta.request_id)
        if item is None:
            item = PendingStoreItem(request_id=meta.request_id, epoch=epoch)
            self._pending_items[meta.request_id] = item
        elif item.epoch != epoch:
            raise RuntimeError(
                f"request {meta.request_id!r} mixed store epochs "
                f"{item.epoch} and {epoch}"
            )
        item.metadatas.append((meta, block_hashes))

    def has_pending_request(self, request_id: str) -> bool:
        return request_id in self._pending_items

    def drop_request(self, request_id: str) -> int:
        """Discard chunks invalidated by a tracker reset."""
        item = self._pending_items.pop(request_id, None)
        return len(item.metadatas) if item is not None else 0

    def discard_for_reuse(self, request_id: str) -> int:
        """Discard a predecessor's chunks before its id is reused."""
        item = self._pending_items.pop(request_id, None)
        return len(item.metadatas) if item is not None else 0

    def release_request(self, request_id: str) -> None:
        """FIFO has no non-pending per-request state to release."""

    def pop_items_for_offload(
        self,
        count: int,
        finished_request_ids: set[str],
        blocked_request_ids: set[str] | None = None,
    ) -> list[PendingStoreItem]:
        """Pop eligible finished requests in admission order."""
        if count <= 0:
            return []
        blocked_request_ids = blocked_request_ids or set()
        eligible_ids = finished_request_ids - blocked_request_ids
        eligible_count = sum(
            request_id in self._pending_items for request_id in eligible_ids
        )
        if eligible_count < self._threshold:
            return []

        to_offload: list[PendingStoreItem] = []
        for request_id in list(self._pending_items):
            if request_id not in eligible_ids:
                continue
            to_offload.append(self._pending_items.pop(request_id))
            if len(to_offload) >= count:
                break
        return to_offload
