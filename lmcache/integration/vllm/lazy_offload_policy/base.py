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
- **Held**: a buffered operation whose GPU blocks the manager pinned while
  the request's batch was in flight, so they cannot be recycled behind a
  slow store. A held operation cannot lose its data; it is submitted once
  the batch settles, and the batch takes over the pin.
- **Protected**: a buffered operation an imminent admission would recycle.
  The manager moves its blocks to the free-queue tail instead of pinning
  them, so the admission still has every free block to allocate from, and
  the next drain emits it. Only when the allocation would reach the tail
  too are they held instead.
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
#: by block id. ``None`` means the block carries no hash.
BlockHashes = dict[int, "BlockHashWithGroupId | None"]


@dataclass
class PendingStoreItem:
    """Store metadata selected for one request by any drain policy.

    Attributes:
        request_id: The vLLM request id these operations belong to. One item
            carries operations of exactly one request, so the manager can
            coalesce them into a single store submission.
        metadatas: The selected operations in token order, each paired with
            the block-hash snapshot taken when it was buffered. The manager
            re-reads the hashes before submitting to prove the blocks still
            hold the same data.
    """

    request_id: str
    metadatas: list[tuple["LMCacheMPRequestMetadata", BlockHashes]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class DrainSignals:
    """One scheduler step's inputs to a drain decision.

    Attributes:
        next_step_blocks: Upper bound on the blocks the requests now running
            can allocate before the next drain. Admissions and resumed
            loads are not in it: the manager protects buffered blocks from
            those right before they allocate.
        resuming_blocks: Upper bound on the blocks that requests whose load
            finished, but that no step has scheduled yet, allocate once
            scheduled. The protection when their load finished assumed the
            next step; until a step schedules them, the head moves on and
            the danger depth counts them too.
        finished_request_ids: Requests whose generation has ended. Their
            buffered operations are still storable; the id is reported so a
            policy may treat them differently from running requests.
        blocked_request_ids: Requests that already have a store batch in
            flight. The worker tracks one store future per request id, so
            these must stay buffered until their receipt arrives; a policy
            may ask the manager to hold their blocks meanwhile.
        preempted_request_ids: Requests this step preempted. Their blocks
            are freed, and once rescheduled they restart from token zero,
            which discards whatever is still buffered for them.
        running_blocks_allocated: Blocks this step allocated to requests
            that were already running, which the previous drain's
            ``next_step_blocks`` bounds. Read only to check that bound.
    """

    next_step_blocks: int
    resuming_blocks: int
    finished_request_ids: set[str]
    blocked_request_ids: set[str]
    preempted_request_ids: set[str]
    running_blocks_allocated: int


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
        holds: Operations of blocked requests whose blocks the manager must
            pin now, one entry per request. Each entry extends the request's
            held operations, which always form a prefix of its buffer. They
            stay buffered and are emitted by a later drain.
        to_tail: Operations at risk that ``max_drain_per_step`` left
            buffered, whose blocks the manager must move to the free-queue
            tail now. They are due, and a later drain emits them.
    """

    items: list[PendingStoreItem] = field(default_factory=list)
    emptied_request_ids: list[str] = field(default_factory=list)
    holds: list[PendingStoreItem] = field(default_factory=list)
    to_tail: list[PendingStoreItem] = field(default_factory=list)


@dataclass
class AdmissionProtection:
    """What the manager must do before an admission allocates.

    Attributes:
        to_tail: Operations to move to the free-queue tail, one entry per
            request, in prefix order. They stay allocatable, so the
            admission is not refused for them, and the next drain emits
            them unless their request is blocked, which holds them.
        holds: Operations to pin, because the allocation would reach the
            tail too. Like ``LazyOffloadDrain.holds``, each entry extends
            the request's held prefix; pinning may make vLLM defer the
            admission, which is cheaper than losing the store.
    """

    to_tail: list[PendingStoreItem] = field(default_factory=list)
    holds: list[PendingStoreItem] = field(default_factory=list)


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

    def protect(self, num_blocks: int) -> AdmissionProtection:
        """Protect what an imminent allocation of ``num_blocks`` would recycle.

        Called right before vLLM allocates blocks for a request it admits,
        which evicts from the free-queue head before the step's drain runs.
        A policy that never lets buffered blocks be recycled returns nothing.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.

        Returns:
            The operations to move to the tail and the ones to pin.
        """
        return AdmissionProtection()

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
    def mark_store_failed(self, request_id: str) -> int:
        """Break the request's prefix chain after a failed store.

        The failed range will never be stored, and the request's tracker
        only moves forward, so every later range of this request would be
        unreachable on retrieval. A policy that keeps prefix-chain state
        drops those ranges and refuses the request's further operations; one
        that drains a request whole has nothing left to drop and returns 0.

        Args:
            request_id: The request whose submitted store failed.

        Returns:
            The number of buffered operations dropped.
        """

    @abstractmethod
    def log_final_stats(self) -> None:
        """Write the policy's final counters at connector shutdown."""
