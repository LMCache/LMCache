# SPDX-License-Identifier: Apache-2.0
"""Base types for lazy-offload policies."""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


@dataclass
class PendingStoreItem:
    """A pending cache-store operation for a single request.

    Attributes:
        request_id: The request identifier for the pending operation.
        metadatas: Store metadata and the captured block hashes to submit.
        is_finished: Whether all cache blocks for the request have been queued.
    """

    request_id: str
    metadatas: list[tuple["LMCacheMPRequestMetadata", dict[int, bytes]]] = field(
        default_factory=list
    )
    is_finished: bool = False


class OffloadPolicy(ABC):
    """Abstract interface for scheduler-side lazy-offload policies.

    Worker processes do not create or invoke policies. Implementations aggregate
    cache blocks by request and decide when pending items should be offloaded.
    """

    @abstractmethod
    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: dict[int, bytes],
    ) -> None:
        """Add cache blocks from one request to the pending store.

        Args:
            meta: Store metadata for a subset of a request's cache blocks.
            block_hashes: Mapping from queued GPU block IDs to block hashes.
        """

    @abstractmethod
    def mark_req_finished(self, req_id: str) -> None:
        """Mark the pending store item for ``req_id`` as finished.

        Args:
            req_id: Identifier of the request that has completed.
        """

    @abstractmethod
    def pop_items_for_offload(self, count: int) -> list[PendingStoreItem]:
        """Pop items only when the policy's offload condition is satisfied.

        Args:
            count: Maximum number of pending items to return.

        Returns:
            Pending items to offload, or an empty list when offload is not due.
        """
