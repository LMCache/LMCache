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
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class FIFOOffloadPolicy(OffloadPolicy):
    """Buffer by request and drain eligible requests in FIFO order.

    A drain happens once enough finished requests have accumulated, and
    releases whole requests in admission order. It reads no GPU state, so it
    never drops an operation whose blocks were recycled.
    """

    def __init__(self, configs: dict[str, ConfigValue]) -> None:
        """Read ``lazy_offload_threshold`` and ``_select_count`` from configs.

        Args:
            configs: The connector's ``kv_connector_extra_config``. Only the
                two keys above are read; missing keys keep their defaults of
                100 eligible requests and 10 requests per drain.
        """
        self._pending_items: dict[str, PendingStoreItem] = {}
        # int() does the conversion; the cast only narrows ConfigValue to
        # what int() accepts, since a JSON config may carry the number as a
        # string.
        self._threshold = int(
            cast(
                "str | int | float",
                configs.get("lmcache.mp.lazy_offload_threshold", 100),
            )
        )
        self._select_count = int(
            cast(
                "str | int | float",
                configs.get("lmcache.mp.lazy_offload_select_count", 10),
            )
        )
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> None:
        """Queue one store operation under its request id.

        Args:
            meta: The store operation offered by the manager.
            block_hashes: Its admission-time block hashes, kept untouched
                for the manager to re-validate. This policy does not read
                them, so it never drops an operation whose blocks were
                recycled.
        """
        item = self._pending_items.get(meta.request_id)
        if item is None:
            item = PendingStoreItem(request_id=meta.request_id)
            self._pending_items[meta.request_id] = item
        item.metadatas.append((meta, block_hashes))

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Release eligible finished requests once the threshold is met.

        Args:
            signals: This step's signals from the manager. Only the
                request-id sets are read: a request is eligible when it
                is finished and not blocked, and the drain happens once
                enough eligible requests have buffered operations. Block
                pressure is ignored.

        Returns:
            Up to ``select_count`` whole requests in admission order, each
            also reported as emptied.
        """
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
        """Whether the request currently owns buffered operations.

        Args:
            request_id: The request id to query.

        Returns:
            True while at least one of its operations is buffered.
        """
        return request_id in self._pending_items

    def drop_request(self, request_id: str) -> int:
        """Discard operations invalidated by a preemption reset.

        Args:
            request_id: The preempted request.

        Returns:
            The number of buffered operations discarded.
        """
        item = self._pending_items.pop(request_id, None)
        return len(item.metadatas) if item is not None else 0

    def discard_for_reuse(self, request_id: str) -> None:
        """Discard what the finished holder of this id left buffered.

        Args:
            request_id: The id a new request is taking over.
        """
        self.drop_request(request_id)

    def release_request(self, request_id: str) -> None:
        """FIFO has no non-pending per-request state to release.

        Args:
            request_id: The request whose session was torn down. Unused.
        """

    def mark_store_failed(self, request_id: str) -> int:
        """FIFO drains a request whole, so nothing of it is left buffered.

        Args:
            request_id: The request whose submitted store failed. Unused:
                this policy keeps no prefix-chain state.

        Returns:
            Always zero.
        """
        return 0

    def log_final_stats(self) -> None:
        """FIFO keeps no counters."""
