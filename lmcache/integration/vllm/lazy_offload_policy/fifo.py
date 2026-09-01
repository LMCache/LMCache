# SPDX-License-Identifier: Apache-2.0
"""FIFO lazy-offload drain policy."""

# Standard
from typing import TYPE_CHECKING, cast

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    PendingStoreItem,
)
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class FIFOOffloadPolicy:
    """Buffer by request and drain controller-eligible ids in FIFO order.

    Legacy placeholder policy: a drain happens once enough finished requests
    have accumulated, and releases whole requests in admission order.
    """

    def __init__(self, configs: dict[str, ConfigValue]) -> None:
        """Read the FIFO trigger and batch size from the connector config.

        Args:
            configs: vLLM connector extra configuration. Recognized here:
                ``lmcache.mp.lazy_offload_threshold`` (finished requests
                needed to trigger a drain) and
                ``lmcache.mp.lazy_offload_select_count`` (requests released
                per drain).
        """
        self._pending_items: dict[str, PendingStoreItem] = {}
        self._threshold = int(
            cast(int, configs.get("lmcache.mp.lazy_offload_threshold", 100))
        )
        self._select_count = int(
            cast(int, configs.get("lmcache.mp.lazy_offload_select_count", 10))
        )
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
        epoch: int,
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

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Release eligible finished requests once the threshold is met."""
        eligible_ids = signals.finished_request_ids - signals.blocked_request_ids
        eligible_count = sum(
            request_id in self._pending_items for request_id in eligible_ids
        )
        if eligible_count < self._threshold:
            return LazyOffloadDrain()

        items: list[PendingStoreItem] = []
        for request_id in list(self._pending_items):
            if request_id not in eligible_ids:
                continue
            items.append(self._pending_items.pop(request_id))
            if len(items) >= self._select_count:
                break
        return LazyOffloadDrain(
            items=items,
            emptied_request_ids=[item.request_id for item in items],
        )

    def has_pending_request(self, request_id: str) -> bool:
        """Whether the request currently owns buffered chunks."""
        return request_id in self._pending_items

    def drop_request(self, request_id: str) -> int:
        """Discard chunks invalidated by a tracker reset."""
        item = self._pending_items.pop(request_id, None)
        return len(item.metadatas) if item is not None else 0

    def discard_for_reuse(self, request_id: str) -> int:
        """Discard a predecessor's chunks before its id is reused."""
        return self.drop_request(request_id)

    def release_request(self, request_id: str) -> None:
        """FIFO has no non-pending per-request state to release."""

    def mark_store_failed(self, request_id: str) -> int:
        """FIFO drains a request whole, so nothing of it is left buffered."""
        return 0

    def log_final_stats(self) -> None:
        """FIFO keeps no counters."""
