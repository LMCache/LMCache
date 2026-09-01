# SPDX-License-Identifier: Apache-2.0
"""Base types for lazy-offload policies."""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata

#: Value type of vLLM's ``kv_connector_extra_config`` entries.
ConfigValue = str | int | float | bool | list[str] | None

#: Prefix-cache hash of every GPU block covering one store operation, keyed
#: by block id. ``None`` means the block carries no hash.
BlockHashes = dict[int, "BlockHashWithGroupId | None"]


@dataclass
class PendingStoreItem:
    """Store metadata selected for one request epoch by any drain policy."""

    request_id: str
    epoch: int = 0
    metadatas: list[tuple["LMCacheMPRequestMetadata", BlockHashes]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class DrainSignals:
    """One scheduler step's inputs to a drain decision.

    The two block counts are the step's gross allocation and the next step's
    estimate; ``allocated_block_ids`` are the ids allocated or resurrected
    this step, whose operations the drain revalidates. Requests in
    ``blocked_request_ids`` have a batch in flight and stay buffered.
    """

    new_blocks_allocated: int
    est_next_step_blocks: int
    allocated_block_ids: set[int]
    finished_request_ids: set[str]
    blocked_request_ids: set[str]


@dataclass
class LazyOffloadDrain:
    """Policy-neutral output consumed by the lazy-offload controller.

    ``items`` is the metadata to submit now, one entry per request;
    ``emptied_request_ids`` are the requests whose buffer became empty, which
    the controller weighs against phase and batch state before teardown.
    """

    items: list[PendingStoreItem] = field(default_factory=list)
    emptied_request_ids: list[str] = field(default_factory=list)


class OffloadPolicy(Protocol):
    """Scheduler-side decision logic for deferring cache stores.

    Implementations buffer store metadata by request and decide, once per
    step, what to release. ``LazyOffloadManager`` owns every GPU and
    connector side effect and calls these from the scheduler thread only.
    """

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
        epoch: int,
    ) -> None:
        """Buffer one store operation instead of submitting it.

        ``block_hashes`` snapshots every block covering ``meta``'s token
        range now, and ``epoch`` is the store epoch that produced it. An
        operation the policy cannot take custody of -- an unhashed block, or
        a request whose stored chain is already broken -- is dropped here.

        Raises:
            RuntimeError: If the request already has a different epoch
                buffered.
        """
        ...

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Decide which buffered operations one scheduler step releases.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """
        ...

    def has_pending_request(self, request_id: str) -> bool:
        """Whether the request currently owns buffered operations."""
        ...

    def drop_request(self, request_id: str) -> int:
        """Discard operations invalidated by a preemption reset.

        Returns:
            The number of buffered operations discarded.
        """
        ...

    def discard_for_reuse(self, request_id: str) -> int:
        """Discard a predecessor's state before its id is reused.

        Returns:
            The number of buffered operations discarded.
        """
        ...

    def release_request(self, request_id: str) -> None:
        """Forget non-pending state after current-session teardown."""
        ...

    def mark_store_failed(self, request_id: str) -> int:
        """Break the request's stored prefix chain after a failed store.

        Returns:
            The number of buffered operations dropped.
        """
        ...

    def log_final_stats(self) -> None:
        """Write the policy's final counters at connector shutdown."""
        ...
