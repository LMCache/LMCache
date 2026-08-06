# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# First Party
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


@dataclass
class PendingStoreItem:
    """
    Represents a pending store operation in the lazy offload queue.

    Attributes:
        request_id: The request id of the pending store request.
        metadatas: The store metadata to be submitted.
        is_finished: Whether the request is finished.
    """

    request_id: str
    metadatas: list[tuple["LMCacheMPRequestMetadata", dict[int, bytes]]] = field(
        default_factory=list
    )
    is_finished: bool = False


# TODO: support more offload policies
class OffloadPolicy(ABC):
    """
    Abstract base class for lazy offload policies.

    Subclasses define when to trigger offload (should_offload) and
    which items to return (select_items).
    """

    @abstractmethod
    def add(self, meta: "LMCacheMPRequestMetadata", block_hashes: dict[int, bytes]):
        """Add a pending store item to the pending store."""
        ...

    @abstractmethod
    def mark_req_finished(self, req_id: str):
        """Mark the pending store item finished."""
        ...

    @abstractmethod
    def should_offload(self) -> bool:
        """Determine whether the queue should be drained.

        Returns:
            True if offload should be triggered.
        """
        ...

    @abstractmethod
    def select_items(self, count: int) -> list[PendingStoreItem]:
        """Select which items to offload from the queue.

        Args:
            count: The number of items to select.

        Returns:
            A list of PendingStoreItem.
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
        self._pending_items: dict[str, PendingStoreItem] = {}
        self._threshold = (
            configs.get("lmcache.mp.lazy_offload_threshold", 100) if configs else 100
        )
        self._finished_requests_count = 0
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(self, meta: "LMCacheMPRequestMetadata", block_hashes: dict[int, bytes]):
        if meta.request_id not in self._pending_items:
            self._pending_items[meta.request_id] = PendingStoreItem(
                request_id=meta.request_id
            )
        self._pending_items[meta.request_id].metadatas.append((meta, block_hashes))

    def mark_req_finished(self, req_id: str):
        if req_id in self._pending_items:
            self._pending_items[req_id].is_finished = True
            self._finished_requests_count += 1
        else:
            raise ValueError(
                f"mark req finished failed: req_id: {req_id} not in pending_items"
            )

    def should_offload(self) -> bool:
        return self._finished_requests_count >= self._threshold

    def select_items(self, count: int) -> list[PendingStoreItem]:
        to_offload = []
        for req_id in list(self._pending_items.keys()):
            if self._pending_items[req_id].is_finished:
                to_offload.append(self._pending_items[req_id])
                del self._pending_items[req_id]
                self._finished_requests_count -= 1
        return to_offload


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

        # TODO: use gpu block pool to judge should offload and select items
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

    def should_offload(self) -> bool:
        """Check if the queue should be drained based on the policy."""
        return self._policy.should_offload()

    def select_items(self) -> list[PendingStoreItem]:
        """
        Drain items from the queue according to the policy.

        Returns:
            Iterator of pending store items to be submitted.
        """
        return self._policy.select_items(self._select_count)

    def mark_req_finished(self, req_id: str):
        self._policy.mark_req_finished(req_id)

    def update_request_gpu_block_ids(self, req_id: str, block_ids: list[int]):
        self._request_block_ids[req_id].extend(block_ids)

    def get_request_gpu_block_ids(self, req_id: str) -> list[int]:
        return self._request_block_ids[req_id]

    def remove_request_gpu_block_ids(self, req_id: str):
        if req_id in self._request_block_ids:
            del self._request_block_ids[req_id]
