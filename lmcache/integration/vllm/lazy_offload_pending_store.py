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
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


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


# TODO(chunxiaozheng): support more offload policies
class OffloadPolicy(ABC):
    """
    Abstract base class for lazy offload policies.

    Policies run in the scheduler role through :class:`LazyOffloadPendingStore`;
    worker processes do not create or call them. Subclasses decide whether
    offload is due and pop items in one ``pop_items_for_offload`` operation.
    """

    @abstractmethod
    def add(self, meta: "LMCacheMPRequestMetadata", block_hashes: dict[int, bytes]):
        """Add cache blocks from one request to the pending store.

        Args:
            meta: Store metadata for a subset of one request's cache blocks.
                Chunked prefill or the scheduler's ``max-num-batched-tokens``
                limit can schedule one request multiple times, so policies must
                aggregate its metadata by ``meta.request_id``.
            block_hashes: Mapping from the GPU block IDs in ``meta`` to their
                corresponding block hashes captured when the metadata is queued.
        """
        ...

    @abstractmethod
    def mark_req_finished(self, req_id: str):
        """Mark the pending store item finished."""
        ...

    @abstractmethod
    def pop_items_for_offload(self, count: int) -> list[PendingStoreItem]:
        """Pop items to offload only when the policy's condition is satisfied.

        When the condition is not satisfied, this method returns an empty list
        and leaves pending items in the queue.

        Args:
            count: Maximum number of items to pop.

        Returns:
            Popped pending store items, or an empty list when offload is not
            due.
        """
        ...


class FIFOOffloadPolicy(OffloadPolicy):
    """
    FIFO offload policy: when finished request count reaches the threshold,
    pops a fixed number of items from the front.
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

    def pop_items_for_offload(self, count: int) -> list[PendingStoreItem]:
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
