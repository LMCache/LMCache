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
    def mark_req_finished(self, req_id: str) -> bool:
        """Mark the pending store item for ``req_id`` as finished.

        Args:
            req_id: Identifier of the request that has completed.

        Returns:
            True if the request has buffered cache blocks awaiting offload,
            so its session must outlive the request; False if it buffered
            none. A request shorter than one chunk finishes without ever
            producing store metadata, which is not an error.
        """

    @abstractmethod
    def drop_request(self, req_id: str) -> int:
        """Discard everything buffered for a request without storing it.

        Called when the engine drops a request before its cache blocks are
        offloaded, so the buffered blocks can no longer be trusted.

        Args:
            req_id: Identifier of the request being dropped.

        Returns:
            The number of buffered entries discarded; 0 if the request had
            nothing buffered.
        """

    @abstractmethod
    def reclaim_finished_request(self, req_id: str) -> bool:
        """Discard a finished predecessor's buffered item on request-id reuse.

        A new request may legally reuse a finished request's id. Inheriting
        the predecessor's buffered item would merge two unrelated requests'
        cache blocks into one store.

        Args:
            req_id: The reused request identifier.

        Returns:
            True if a finished item was discarded, meaning the caller must
            end the predecessor's session now; False if the id carries no
            finished item.
        """

    @abstractmethod
    def pop_items_for_offload(
        self,
        count: int,
        blocked_request_ids: set[str] | None = None,
    ) -> list[PendingStoreItem]:
        """Pop items only when the policy's offload condition is satisfied.

        Args:
            count: Maximum number of pending items to return.
            blocked_request_ids: Requests that already have a submitted batch
                and must stay queued until its completion receipt arrives.

        Returns:
            Pending items to offload, or an empty list when offload is not due.
        """
