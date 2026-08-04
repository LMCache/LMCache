# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

# First Party
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


@dataclass
class PendingStoreItem:
    """
    Represents a pending store operation in the lazy offload queue.

    Attributes:
        metadata: The store metadata to be submitted.
    """

    metadata: "LMCacheMPRequestMetadata"


# TODO: support more offload policies
class OffloadPolicy(ABC):
    """
    Abstract base class for lazy offload policies.

    Subclasses define when to trigger offload (should_offload) and
    which items to return (select_items).
    """

    @abstractmethod
    def add(self, item: PendingStoreItem):
        """Add a pending store item to the queue or other data structures."""
        ...

    @abstractmethod
    def should_offload(self) -> bool:
        """Determine whether the queue should be drained.

        Returns:
            True if offload should be triggered.
        """
        ...

    @abstractmethod
    def select_items(self, count: int) -> Iterator[PendingStoreItem]:
        """Select which items to offload from the queue.

        Args:
            count: The number of items to select.

        Returns:
            A Iterator of PendingStoreItem.
        """
        ...


class FIFOOffloadPolicy(OffloadPolicy):
    """
    FIFO offload policy: triggers when pending count reaches threshold,
    and returns a fixed batch_size number of items from the front.
    """

    def __init__(self, configs: dict | None = None):
        """
        Args:
            configs: The configuration for the FIFO offload policy.
        """
        self.pending_items: list[PendingStoreItem] = []
        self.threshold = (
            configs.get("lmcache.mp.lazy_offload_threshold", 100) if configs else 100
        )
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self.threshold,
        )

    def add(self, item: PendingStoreItem):
        """Add a pending store item to the queue."""
        self.pending_items.append(item)

    def should_offload(self) -> bool:
        """Trigger offload when pending count >= threshold."""
        return len(self.pending_items) >= self.threshold

    def select_items(self, count: int) -> Iterator[PendingStoreItem]:
        """Yield the first batch_size items (FIFO order)."""
        to_offload = self.pending_items[:count]
        self.pending_items = self.pending_items[count:]
        return iter(to_offload)


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
        policy = (
            configs.get("lmcache.mp.lazy_offload_policy", "FIFO") if configs else "FIFO"
        )
        if policy == "FIFO":
            self._policy = FIFOOffloadPolicy(configs)
        else:
            raise ValueError(f"Unknown offload policy: {policy}")

        # TODO: support more flexible select count
        self._select_count = (
            configs.get("lmcache.mp.lazy_offload_select_count", 10) if configs else 10
        )

        # check if the block hashes are the same when trigger offload
        self._request_block_hashes: dict[str, dict[int, bytes]] = {}

    def add(self, item: PendingStoreItem, block_hashes: dict[int, bytes]) -> None:
        """Add a pending store item to the pending store."""
        self._request_block_hashes[item.metadata.request_id] = block_hashes
        self._policy.add(item)

    def should_offload(self) -> bool:
        """Check if the queue should be drained based on the policy."""
        return self._policy.should_offload()

    def select_items(self) -> Iterator[PendingStoreItem]:
        """
        Drain items from the queue according to the policy.

        Returns:
            Iterator of pending store items to be submitted.
        """
        return self._policy.select_items(self._select_count)

    def get_block_hashes(self, request_id: str) -> dict[int, bytes]:
        return self._request_block_hashes.get(request_id, {})

    def remove_block_hashes(self, request_id: str) -> None:
        self._request_block_hashes.pop(request_id, None)
