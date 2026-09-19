# SPDX-License-Identifier: Apache-2.0
"""Base types for lazy-offload policies.

Terms used throughout this package:

- **Store operation**: one contiguous token range of one request whose KV is
  ready to be written to LMCache. vLLM's request tracker produces these; a
  lazy-offload policy buffers them instead of submitting them at once.
- **Buffered**: held by the policy, not yet handed to the worker. A buffered
  operation still points at live GPU blocks and is only valid while those
  blocks hold the data snapshotted at admission.
- **Emitted**: chosen by a drain and handed back to ``LazyOffloadManager``,
  which pins the blocks and submits the store to the worker.
- **Prefix chain**: the sequence of a request's stored token ranges, in token
  order. LMCache retrieval walks it from token zero, so a range whose
  earlier neighbour was never stored is unreachable; once a range is lost,
  the chain is *broken* and every later range of that request is worthless.
- **Store batch**: the emitted operations of one request, coalesced into the
  one store the worker accepts per request. A batch is *in flight* from its
  submission until every worker has reported it, and the request's later
  operations stay buffered for as long as that lasts.
- **Session**: the LMCache-side state a request owns while it is storing.
  ``LazyOffloadManager`` ends it once the request has finished and has
  nothing buffered and nothing in flight.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata

#: Value type of vLLM's ``kv_connector_extra_config`` entries.
ConfigValue = str | int | float | bool | list[str] | None

#: Prefix-cache hash of every GPU block covering one store operation, keyed
#: by ``(engine_group_id, block_id)``. Block IDs are group-local for hybrid
#: models and must never be flattened into one numeric namespace. ``None``
#: means the block carries no hash.
BlockHashes = dict[tuple[int, int], "BlockHashWithGroupId | None"]

#: One independently scheduled cache-retention group of a request.
RetentionGroupKey = tuple[str, int]


@dataclass
class PendingStoreItem:
    """Store metadata selected for one request by any drain policy.

    Attributes:
        request_id: The vLLM request id these operations belong to. One item
            carries operations of exactly one request, so the manager can
            coalesce them into a single store submission.
        retention_group_id: Scheduler-side group whose cache state shares one
            retirement mechanism and LMCache object-group commit boundary.
        metadatas: The selected operations in token order, each paired with
            the block-hash snapshot taken when it was buffered. The manager
            re-reads the hashes before submitting to prove the blocks still
            hold the same data.
    """

    request_id: str
    retention_group_id: int = 0
    metadatas: list[tuple["LMCacheMPRequestMetadata", BlockHashes]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class DrainSignals:
    """One scheduler step's inputs to a drain decision.

    Attributes:
        new_blocks_allocated: GPU blocks vLLM handed out during this step,
            counted gross. Drives the policy's estimate of how fast the free
            queue is being consumed.
        est_next_step_blocks: Blocks the next step is expected to allocate,
            derived from the tokens this step scheduled. Used so a burst
            after an idle period is not underestimated.
        finished_request_ids: Requests whose generation has ended. Their
            buffered operations are still storable; the id is reported so a
            policy may treat them differently from running requests.
        blocked_request_ids: Legacy request-wide in-flight blocks. Kept for
            compatibility with callers that cannot identify retention groups.
        blocked_retention_groups: Request/group pairs that already have a
            store operation in flight. Other groups of the same request may
            still drain independently.
        request_progress_tokens: Projected computed-token count after this
            scheduler step, keyed by request id. Windowed groups use it to
            submit state before the next scheduler step can retire its blocks.
    """

    new_blocks_allocated: int
    est_next_step_blocks: int
    finished_request_ids: set[str]
    blocked_request_ids: set[str] = field(default_factory=set)
    blocked_retention_groups: set[RetentionGroupKey] = field(default_factory=set)
    request_progress_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class LazyOffloadDrain:
    """Policy-neutral output consumed by ``LazyOffloadManager``.

    Attributes:
        items: What to submit now, one entry per request.
        emptied_request_ids: Requests whose buffer became empty during this
            drain, whether by emission or by dropping invalidated
            operations. This is a buffer fact only: the manager combines it
            with the request's phase and in-flight batch before tearing a
            session down.
    """

    items: list[PendingStoreItem] = field(default_factory=list)
    emptied_request_ids: list[str] = field(default_factory=list)


class OffloadPolicy(ABC):
    """Scheduler-side decision logic for deferring cache stores.

    Implementations buffer store metadata by request and decide, once per
    step, what to release. ``LazyOffloadManager`` owns every GPU and
    connector side effect and calls these from the scheduler thread only.
    """

    @abstractmethod
    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
        *,
        requires_prefix: bool = True,
        retire_at_token: int | None = None,
    ) -> None:
        """Buffer one store operation instead of submitting it.

        One request's operations arrive in token order, each starting
        where the previous one ended, and a policy must keep them in that
        order: they are submitted as one coalesced store.

        An operation the policy cannot take custody of is dropped here
        rather than buffered: one covered by a block that carries no
        prefix-cache hash (its later eviction would be undetectable), or one
        belonging to a request whose prefix chain is already broken.

        Args:
            meta: The store operation: the request id, the token range, and
                the GPU blocks holding that range's KV.
            block_hashes: Prefix-cache hash of every block covering
                ``meta``'s token range, read now. The policy keeps this
                snapshot and compares against it later; a block whose hash
                changed was recycled, so the buffered data is gone.
            requires_prefix: Whether losing one range makes all later ranges
                of this retention group unreachable. Full attention requires
                this; sliding-window and recurrent groups do not.
            retire_at_token: Projected request progress at which the operation
                must be emitted before this group's live blocks can retire.
                ``None`` means free-queue pressure is the only trigger.
        """

    @abstractmethod
    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Decide which buffered operations one scheduler step releases.

        Args:
            signals: This step's block-allocation pressure and the request
                ids the manager reports as finished or blocked.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """

    @abstractmethod
    def has_pending_request(self, request_id: str) -> bool:
        """Whether the request currently owns buffered operations.

        Args:
            request_id: The vLLM request id to query.

        Returns:
            True while at least one of its operations is still buffered.
        """

    @abstractmethod
    def drop_request(self, request_id: str) -> int:
        """Discard operations invalidated by a preemption reset.

        Called when vLLM preempts a request and frees its GPU blocks: every
        buffered operation of that request points at blocks it no longer
        owns. The request keeps its id and restarts from token zero, so the
        policy must also forget that its prefix chain was broken.

        Args:
            request_id: The preempted request whose buffer is discarded.

        Returns:
            The number of buffered operations discarded.
        """

    @abstractmethod
    def discard_for_reuse(self, request_id: str) -> None:
        """Discard what a finished request left behind before its id is reused.

        Called when a new, unrelated request arrives under a request id whose
        previous holder has finished but whose policy state is still around
        (its stores were deferred, or its teardown waits on a store receipt).
        Without this, the arriving request would inherit the finished one's
        buffered operations and broken-chain marker.

        Args:
            request_id: The reused request id, whose previous holder's state
                is discarded.
        """

    @abstractmethod
    def release_request(self, request_id: str) -> None:
        """Forget a request's leftover state once its session is torn down.

        Called by the manager when it ends the session of a request that
        has nothing buffered and nothing in flight. Only state outside the
        pending buffer is affected -- the broken-chain marker -- so that
        finished request ids do not accumulate.

        Args:
            request_id: The request whose session is being torn down.
        """

    @abstractmethod
    def mark_store_failed(
        self,
        request_id: str,
        retention_group_id: int = 0,
        *,
        requires_prefix: bool = True,
    ) -> int:
        """Apply a failed store to one retention group's prefix state.

        The failed range will never be stored, and the request's tracker
        only moves forward, so every later range of this request would be
        unreachable on retrieval. A policy that keeps prefix-chain state
        drops those ranges and refuses the request's further operations; one
        that drains a request whole has nothing left to drop and returns 0.

        Args:
            request_id: The request whose submitted store failed.
            retention_group_id: Independently stored group that failed.
            requires_prefix: Whether later ranges depend on the failed range.

        Returns:
            The number of buffered operations dropped.
        """

    @abstractmethod
    def log_final_stats(self) -> None:
        """Write the policy's final counters at connector shutdown."""
