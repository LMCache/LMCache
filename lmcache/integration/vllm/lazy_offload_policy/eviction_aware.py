# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy (gates 1 and 3 of the store decision).

Implements the drain policy described in
``docs/design/integration/vllm/lazy_offload_decision_model.md``: store
operations are buffered instead of submitted eagerly, and are released only
when the GPU blocks holding their data are about to be evicted (gate 1,
"replace prediction with timing") and the covered prefix is long enough for
the store to beat recomputation (gate 3, static break-even threshold).

This module is pure policy: it never touches vLLM at runtime (vLLM types
appear only in annotations) and performs no I/O, so it is unit-testable
without a GPU or a vLLM installation. The connector owns execution: taking
block-hash snapshots at admission, calling :meth:`EvictionAwareStoreQueue.
observe_step` / :meth:`EvictionAwareStoreQueue.collect_due` once per
scheduler step, pinning (``touch``) the blocks of emitted operations, and
submitting them to the worker.
"""

# Standard
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol
import enum
import math

# First Party
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Smoothing factor for the per-step block-consumption EMA. Not a config knob:
# the horizon (in steps) is the tunable quantity; the EMA only smooths noise.
_EMA_ALPHA = 0.3


class BlockPoolReader(Protocol):
    """Read-only view over the GPU block pool required by the policy.

    The production implementation is :class:`GPUBlockPoolView`; tests provide
    a fake. Both must be side-effect free: the policy never mutates pool
    state.
    """

    def free_queue_ranks(self, max_depth: int) -> dict[int, int]:
        """Return the head of the free queue in eviction order.

        Args:
            max_depth: How many blocks to report, counted from the eviction
                head. The policy never compares a rank against anything
                larger, and this runs once per scheduler step on the
                critical path while the queue holds every free block in the
                pool, so an implementation must not walk past the bound.

        Returns:
            Mapping from block id to its rank in the free queue, where rank
            0 is the next eviction victim, for the first ``max_depth``
            blocks only. Blocks that are not in the free queue (in use, or
            pinned) are absent from the mapping, as are blocks past the
            bound; both mean the same thing to the policy -- not at risk
            this step.
        """
        ...

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current prefix-cache hash of a GPU block.

        Args:
            block_id: The GPU block id to inspect.

        Returns:
            The block's current hash, or None if the block holds no cached
            (full, hashed) content -- e.g. it was evicted and reallocated,
            or never completed.
        """
        ...


class GPUBlockPoolView:
    """Production :class:`BlockPoolReader` over a bound vLLM ``BlockPool``.

    All accesses are read-only. ``free_queue_ranks`` costs O(the depth it is
    asked for) rather than O(the free queue), which matters because it runs
    on the scheduler thread once per step: a pool sized to fill the GPU
    keeps tens of thousands of blocks free.
    """

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a vLLM block pool obtained via ``bind_gpu_block_pool``.

        Args:
            block_pool: The scheduler's GPU block pool.
        """
        self._block_pool = block_pool

    def free_queue_ranks(self, max_depth: int) -> dict[int, int]:
        """Snapshot the head of the free queue into block id -> rank.

        Walks the queue's links and stops at the bound, instead of calling
        ``get_all_free_blocks()``, which materialises the whole queue into a
        list (its own docstring in vLLM says it is mainly for testing).

        Args:
            max_depth: Blocks to report, counted from the eviction head.
                Zero or less reports nothing.

        Returns:
            Mapping from block id to rank (0 = next eviction victim) for the
            first ``max_depth`` free blocks.

        Raises:
            RuntimeError: If the free list's fake head has no successor,
                i.e. the queue is not in the shape vLLM maintains.
        """
        if max_depth <= 0:
            return {}
        block = self._block_pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        ranks: dict[int, int] = {}
        rank = 0
        # The fake tail is the one node with no successor, so this stops
        # before reaching it.
        while block.next_free_block is not None and rank < max_depth:
            ranks[block.block_id] = rank
            block = block.next_free_block
            rank += 1
        return ranks

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached.

        Args:
            block_id: The GPU block id to inspect.
        """
        return self._block_pool.blocks[block_id].block_hash

    def num_free_blocks(self) -> int:
        """Return the number of blocks currently in the free queue."""
        return self._block_pool.get_num_free_blocks()


class AdmitResult(enum.Enum):
    """Outcome of admitting a store operation into the lazy queue.

    The connector maps each outcome to an action:

    - ADMITTED: nothing to do now; the operation will be emitted later.
    - REJECTED_UNHASHED_BLOCK: a covered block has no hash, so eviction of
      that block could not be detected later (a reallocated block would also
      read None, masking the loss). The connector must skip the store and
      warn. Because the caller's tracker has already advanced past the
      skipped range, the request's later chunks are unreachable and will be
      rejected as prefix-broken. With plain prefix caching, chunk-aligned
      ranges never cover unhashed blocks; hybrid-attention models (sliding
      window, mamba) can place hash-less null blocks in block tables.
    - REJECTED_PREFIX_BROKEN: an earlier chunk of this request was already
      dropped, so this chunk would be unreachable on retrieval. The
      connector must skip the store entirely.
    - DEDUPLICATED: identical content (same salt, range, and block-hash
      chain) is already buffered under another request. Nothing to do: the
      content will be stored -- or dropped -- with the operation that
      buffered it, and this operation must not defer its own request's
      session teardown.
    """

    ADMITTED = enum.auto()
    REJECTED_UNHASHED_BLOCK = enum.auto()
    REJECTED_PREFIX_BROKEN = enum.auto()
    DEDUPLICATED = enum.auto()


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        request_id: The vLLM request this operation belongs to.
        store_metadata: The ready-to-send store metadata produced by
            ``LMCacheMPRequestMetadata.GetStoreMetadata``; opaque to the
            policy.
        block_hashes: Hash of every GPU block covering the operation's token
            range, snapshotted at admission. All values are non-None
            (enforced by admission); a later mismatch against the pool means
            the block was evicted or reallocated.
        prefix_start_tokens: Token index of the start of this operation's
            range. Used to detect holes in a request's pending list: after
            a deduplicated chunk, the next operation does not start where
            the previous pending one ended, and an emitted batch must never
            span such a hole (it is coalesced into one contiguous store).
        prefix_end_tokens: Token index one past the end of this operation's
            range, i.e. the request-prefix length covered once this
            operation and all earlier ones are stored.
        cache_salt: The request's cache salt, part of the operation's
            content identity for deduplication (two requests with the same
            block hashes but different salts store under different keys).
    """

    request_id: str
    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: dict[int, "BlockHashWithGroupId"]
    prefix_start_tokens: int
    prefix_end_tokens: int
    cache_salt: str = ""


def _content_key(
    op: PendingStoreOp,
) -> tuple[str, int, tuple["BlockHashWithGroupId", ...]]:
    """Content identity of an operation, independent of its request.

    Two operations with equal keys cover the same token range with the same
    cached content: the block-hash chain encodes the token prefix, and the
    salt separates cache namespaces.
    """
    return (op.cache_salt, op.prefix_end_tokens, tuple(op.block_hashes.values()))


def _contiguous_front_run(ops: list[PendingStoreOp]) -> list[PendingStoreOp]:
    """Front slice of ops up to (excluding) the first token-range hole.

    Deduplication can leave a hole in a request's pending list: the missing
    chunk is buffered under another request. An emitted batch is coalesced
    into a single store operation with one contiguous token range, so it
    must never span a hole; ops past the hole stay pending and are emitted
    in a later batch once the front run's completion receipt arrives.
    """
    for index in range(1, len(ops)):
        if ops[index].prefix_start_tokens != ops[index - 1].prefix_end_tokens:
            return ops[:index]
    return ops


@dataclass(frozen=True)
class LazyOffloadPolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        horizon_steps: How many scheduler steps of estimated block
            consumption to treat as "imminent eviction". Larger values drain
            earlier (closer to eager, fewer drops); smaller values drain
            later (better filtering, more drops).
        min_prefix_tokens: Break-even prefix length (gate 3): a request
            whose known prefix is shorter than this when its blocks come due
            is dropped instead of stored. 0 disables the gate.
        max_drain_per_step: Upper bound on operations emitted per step, to
            bound the D2H burst. Must be >= 1. There is no safe static
            lower bound: a prefilling request buffers about one operation
            per step, so a cap below the number of concurrently prefilling
            requests cannot keep up and the backlog is lost to eviction
            rather than merely delayed. Sizing it therefore needs the
            workload, and the runtime sensor for having sized it wrong is
            ``LazyOffloadCounters.throttled_drains``.
    """

    horizon_steps: float = 2.0
    min_prefix_tokens: int = 0
    max_drain_per_step: int = 64

    def __post_init__(self) -> None:
        """Validate field ranges.

        Raises:
            ValueError: If any field is outside its documented range.
        """
        if self.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {self.horizon_steps}")
        if self.min_prefix_tokens < 0:
            raise ValueError(
                f"min_prefix_tokens must be >= 0, got {self.min_prefix_tokens}"
            )
        if self.max_drain_per_step < 1:
            raise ValueError(
                f"max_drain_per_step must be >= 1, got {self.max_drain_per_step}"
            )


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    ``dropped_evicted`` is the gate-1 quality sensor (drop rate): operations
    lost because their blocks were evicted before the policy drained them.
    ``rejected_short_prefix`` counts gate-3 rejections.

    ``throttled_drains`` is the sizing sensor for
    :attr:`LazyOffloadPolicyConfig.max_drain_per_step`: drains that left a
    due operation unemitted because the cap ran out. One of these is
    harmless -- the operation is emitted a step later -- but a cap below
    the number of concurrently prefilling requests never works the backlog
    off, so the count rising alongside ``dropped_evicted`` is the signature
    of a cap set too low. Counted per drain, not per operation, so it is
    comparable with the number of steps rather than with the other
    counters.
    """

    admitted: int = 0
    emitted: int = 0
    dropped_evicted: int = 0
    rejected_short_prefix: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_failed_store: int = 0
    dropped_id_reuse: int = 0
    deduplicated: int = 0
    throttled_drains: int = 0


@dataclass
class DrainResult:
    """Operations released by one :meth:`EvictionAwareStoreQueue.collect_due`.

    Attributes:
        to_store: Operations to submit now, ordered by eviction imminence
            across requests and by prefix order within a request. The
            connector must pin (``touch``) their blocks before the store and
            unpin after completion.
        dropped_evicted: Operations whose data was lost (block evicted or
            reallocated before drain), including later same-request
            operations dropped for prefix closure.
        dropped_short_prefix: Operations dropped by gate 3 (request prefix
            below the break-even length at the time its blocks came due).
        released_requests: Finished requests that no longer have any pending
            operations after this drain; the connector may now end their
            sessions.
        ops_held_back: Operations this drain found due but did not emit
            because ``max_drain_per_step`` ran out. They stay pending and
            are emitted by a later drain if their blocks survive that long.
            Counts only the segment the cap cut, so it is a lower bound:
            candidates the loop never reached are not counted, their
            due-ness being unevaluated.
    """

    to_store: list[PendingStoreOp] = field(default_factory=list)
    dropped_evicted: list[PendingStoreOp] = field(default_factory=list)
    dropped_short_prefix: list[PendingStoreOp] = field(default_factory=list)
    released_requests: list[str] = field(default_factory=list)
    ops_held_back: int = 0


class EvictionAwareStoreQueue:
    """Buffers store operations and releases them by eviction imminence.

    Gate 1 realization: an operation is emitted when any of its blocks sits
    within the *danger depth* of the free queue -- the number of blocks the
    engine is expected to consume within ``horizon_steps`` steps, estimated
    from an EMA of observed per-step allocation and a one-step feedforward
    supplied by the connector. An idle engine (no allocation pressure) never
    triggers a drain; operations whose blocks are evicted before they come
    due are dropped and counted, never stored stale.

    Admission deduplicates by content: an operation whose salt, range, and
    block-hash chain match a pending operation of another request is not
    buffered again. This bounds the queue by the amount of unique cached
    content on the GPU -- without it, every request over a hot shared
    prefix (blocks that never enter the free queue, so never come due)
    would buffer its own copy indefinitely. A hit is validated against the
    pool: the covering op -- and every earlier pending op of its request,
    whose loss would prefix-close over the cover on the next drain -- must
    still hold its admission-time snapshot; otherwise the new copy is
    admitted instead and takes over the content key. Deduplication is still
    optimistic past that check: if the covering operation is dropped later,
    chunks the deduplicated request stores past that point are unreachable
    until a future request re-buffers the missing prefix -- wasted storage,
    never corruption.
    A deduplicated chunk also leaves a hole in its request's pending list;
    emission never spans a hole (each batch is one contiguous store), so
    the ops on each side of it go out in separate batches.

    Not thread-safe: all methods must be called from the scheduler thread
    (the vLLM connector scheduler-side call pattern).
    """

    def __init__(self, config: LazyOffloadPolicyConfig, pool: BlockPoolReader) -> None:
        """Create an empty queue.

        Args:
            config: Policy tunables.
            pool: Read-only view of the GPU block pool.
        """
        self._config = config
        self._pool = pool
        # Per-request pending operations in prefix (admission) order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Requests whose prefix chain was broken by a drop; further chunks
        # of these requests are unreachable and must not be admitted.
        self._prefix_broken: set[str] = set()
        # Requests reported finished by the engine; used to compute
        # DrainResult.released_requests.
        self._finished: set[str] = set()
        # Requests with an emitted batch whose store completion has not been
        # reported yet. The worker tracks one in-flight store per request, so
        # further emissions for these requests are held back until
        # notify_stored().
        self._in_flight: set[str] = set()
        # In-flight batches invalidated by drop_request (preemption reset).
        # Ops admitted after the reset are re-produced from token zero and do
        # not depend on such a batch, so its failure must not drop them.
        self._stale_in_flight: set[str] = set()
        # Content key -> the pending operation buffering that content. An
        # operation whose content is already buffered under another request
        # is deduplicated at admission: without this, every request over a
        # hot shared prefix (blocks that never enter the free queue) would
        # buffer its own copy and the queue would grow without bound. With
        # deduplication the queue is bounded by the amount of unique cached
        # content on the GPU. The covering op is kept (not just the key) so
        # a dedup hit can verify its snapshot is still live: a doomed op can
        # sit in the pending list ahead of its eviction drop, and it must
        # not absorb a live copy of the content.
        self._pending_content: dict[
            tuple[str, int, tuple["BlockHashWithGroupId", ...]], PendingStoreOp
        ] = {}
        self._blocks_per_step_ema: float = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        self._counters = LazyOffloadCounters()

    def admit(self, op: PendingStoreOp) -> AdmitResult:
        """Admit a store operation into the pending queue.

        Args:
            op: The operation to buffer. ``op.block_hashes`` must cover every
                GPU block of the operation's token range.

        Returns:
            The admission outcome; see :class:`AdmitResult` for the action
            the caller must take on each value.
        """
        if op.request_id in self._prefix_broken:
            self._counters.rejected_prefix_broken += 1
            return AdmitResult.REJECTED_PREFIX_BROKEN
        if any(block_hash is None for block_hash in op.block_hashes.values()):
            # The caller's tracker has already advanced past this range, so
            # the request's later chunks would be stored without their prefix
            # (unreachable): reject them like any other broken chain.
            self._prefix_broken.add(op.request_id)
            self._counters.rejected_unhashed += 1
            return AdmitResult.REJECTED_UNHASHED_BLOCK
        content_key = _content_key(op)
        covering = self._pending_content.get(content_key)
        if covering is not None and self._chain_intact(covering):
            self._counters.deduplicated += 1
            return AdmitResult.DEDUPLICATED
        # No covering op, or it is doomed (its own blocks were recycled, or
        # an earlier sibling's were, so the next drain prefix-closes over
        # it): buffer the live copy and make it the new cover. The doomed
        # op stays pending and is dropped by collect_due().
        self._pending.setdefault(op.request_id, []).append(op)
        self._pending_content[content_key] = op
        self._counters.admitted += 1
        return AdmitResult.ADMITTED

    def observe_step(
        self, new_blocks_allocated: int, est_next_step_blocks: int
    ) -> None:
        """Record one scheduler step's block-consumption signals.

        Must be called once per step, before :meth:`collect_due`.

        Args:
            new_blocks_allocated: GPU blocks newly allocated in the step
                that just finished scheduling (gross allocation, counted
                from the scheduler output).
            est_next_step_blocks: Estimated blocks the next step will
                allocate (e.g. scheduled tokens divided by block size).
        """
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = est_next_step_blocks

    def mark_request_finished(self, request_id: str) -> bool:
        """Record that the engine finished a request.

        Args:
            request_id: The finished request.

        Returns:
            True if the request still has pending operations (the caller
            must defer session teardown until the request appears in a
            :class:`DrainResult`'s ``released_requests``); False if nothing
            is pending and the caller may tear down immediately.
        """
        if request_id in self._pending or request_id in self._in_flight:
            self._finished.add(request_id)
            return True
        self._prefix_broken.discard(request_id)
        return False

    def drop_request(self, request_id: str) -> int:
        """Discard all pending operations of a request.

        Called when the buffered state becomes stale: today only when a
        preempted request's tracker is reset (after resume it re-produces
        store metadata from token zero, overlapping anything still
        buffered). An abort does not drop: it routes through
        :meth:`mark_request_finished` and the buffered ops stay storable.
        An in-flight batch is deliberately not forgotten: it
        stays tracked until its completion receipt arrives via
        :meth:`notify_stored`, so an operation re-admitted after the drop
        cannot be emitted while the worker still holds an outstanding
        store for the request (one in-flight batch per request). Such a
        batch is marked stale: operations admitted after the reset do not
        depend on it, so its failure no longer breaks their prefix chain.

        Precondition: the request is not finished with deferred teardown
        (:meth:`mark_request_finished` returned True and no release arrived
        yet). The drop discards the finished marker without emitting a
        release, so violating this would leak the caller's session. The only
        call site today -- the preemption tracker reset -- satisfies it:
        a finished request is never rescheduled, hence never preempted, and
        a reused id is stripped of its predecessor's marker by
        :meth:`reclaim_finished_request` before the successor can be
        preempted.

        Args:
            request_id: The request to discard.

        Returns:
            The number of operations discarded.
        """
        dropped = self._pending.pop(request_id, [])
        self._forget_content(dropped)
        self._counters.dropped_on_request_drop += len(dropped)
        self._finished.discard(request_id)
        self._prefix_broken.discard(request_id)
        if request_id in self._in_flight:
            self._stale_in_flight.add(request_id)
        return len(dropped)

    def reclaim_finished_request(self, request_id: str) -> bool:
        """Release a finished predecessor's residual state on id reuse.

        In lazy mode the engine frees a finished request's id immediately
        (``request_finished`` returns False), so a client may submit a new
        request under an id whose previous owner still has buffered
        operations or an in-flight batch (teardown deferred). Must be
        called when the caller first sees such a new request; without it
        the two requests' state conflates: the predecessor's eviction drop
        would prefix-close over the successor's intact operations, and the
        deferred session release would fire while the successor is live.

        The predecessor's buffered operations are discarded along with its
        finished marker. An in-flight batch is additionally marked stale.
        The marker must not survive the reclaim: the successor is live by
        definition (the reclaim is triggered by its arrival), so a kept
        marker would authorize a premature session teardown at the batch's
        completion receipt -- or, worse, ride until the successor's own
        receipt or eviction drop and tear down mid-request. The id-keyed
        session instead covers both requests and ends once, through the
        successor's own lifecycle (:meth:`mark_request_finished` re-creates
        the marker when the successor finishes); the predecessor's receipt
        only clears the in-flight hold.

        Args:
            request_id: The reused request id.

        Returns:
            True if the caller must end the predecessor's session now;
            False if there was nothing to reclaim or the session teardown
            merges into the successor's lifecycle (in-flight batch).
        """
        if request_id not in self._finished:
            return False
        dropped = self._pending.pop(request_id, [])
        self._forget_content(dropped)
        self._counters.dropped_id_reuse += len(dropped)
        self._prefix_broken.discard(request_id)
        self._finished.discard(request_id)
        if request_id in self._in_flight:
            self._stale_in_flight.add(request_id)
            return False
        return True

    def mark_store_failed(self, request_id: str) -> int:
        """Record that the request's in-flight store batch failed.

        The request's stored prefix chain is broken: its held-back pending
        operations are dropped (stored without the failed prefix they would
        be unreachable) and further admissions are rejected. The finished
        and in-flight markers are left untouched, so the completion receipt
        that accompanies the failure still tears the request down through
        :meth:`notify_stored` as usual.

        A failure of a batch made stale by :meth:`drop_request` is ignored:
        operations admitted after the reset were re-produced from token
        zero and do not depend on the failed prefix.

        Args:
            request_id: The request whose store failed.

        Returns:
            The number of pending operations dropped.
        """
        if request_id in self._stale_in_flight:
            return 0
        dropped = self._pending.pop(request_id, [])
        self._forget_content(dropped)
        self._counters.dropped_failed_store += len(dropped)
        self._prefix_broken.add(request_id)
        return len(dropped)

    def num_pending_ops(self) -> int:
        """Return the total number of buffered store operations."""
        return sum(len(ops) for ops in self._pending.values())

    def stats(self) -> LazyOffloadCounters:
        """Return a copy of the cumulative policy counters."""
        return replace(self._counters)

    def collect_due(self) -> DrainResult:
        """Release the operations whose blocks face imminent eviction.

        For every pending request, first drops the suffix of its operation
        list starting at the first operation whose data is already lost
        (current block hash differs from the admission snapshot) -- storing
        a later chunk without its prefix would be unreachable. Then, if any
        surviving operation has a block within the danger depth of the free
        queue, the request's operations are released from the front up to
        the last due one (prefix closure), subject to gate 3. The released
        segment is additionally cut at the first deduplication hole: the
        batch is coalesced into one contiguous store operation, so ops past
        the hole wait for a later batch.

        Emitting a segment pins its blocks out of the free queue, which
        moves every block behind them toward the head by the segment's size
        before the next step's allocation runs. Each candidate is therefore
        checked against ``danger_depth`` extended by the blocks emitted
        earlier in this call, so a request that an emission teleports into
        the danger window drains now instead of losing the race to the next
        allocation. The first emission still requires a plain
        ``danger_depth`` hit: an idle system never starts draining.

        Returns:
            The operations to store and to drop this step; see
            :class:`DrainResult`.
        """
        result = DrainResult()
        if not self._pending:
            return result

        danger_depth = self._danger_depth()
        # Read only as deep as this call can compare against: the danger
        # depth, plus every block the call could pin out of the queue ahead
        # of a later candidate (the emission shift described above). A zero
        # danger depth makes nothing due, so nothing is read at all -- the
        # loss check below reads block hashes, not ranks, and still runs.
        ranks = (
            self._pool.free_queue_ranks(danger_depth + self._pending_blocks())
            if danger_depth > 0
            else {}
        )

        # Per request: (min in-queue rank, request id, surviving ops).
        # Iterate over a copy: helpers may drop entries from self._pending.
        candidates: list[tuple[int, str, list[PendingStoreOp]]] = []
        for request_id, ops in list(self._pending.items()):
            if request_id in self._in_flight:
                # One in-flight store batch per request (worker constraint);
                # held-back ops are re-examined once notify_stored() arrives.
                continue
            surviving = self._drop_evicted_suffix(request_id, ops, result)
            if not surviving:
                continue
            op_ranks = [
                rank
                for op in surviving
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            if not op_ranks:
                # No block in the free queue (all in use or pinned): the
                # request cannot be due, shifted or not.
                continue
            candidates.append((min(op_ranks), request_id, surviving))

        # Most imminent requests first. The cap may split a segment, but the
        # emitted part is a front slice of it, so within-request prefix order
        # is never violated; the rest stays pending for a later step.
        candidates.sort(key=lambda cand: cand[0])
        budget = self._config.max_drain_per_step
        emitted_blocks = 0
        for min_rank, request_id, surviving in candidates:
            if budget <= 0:
                break
            segment = self._due_front_segment(
                surviving, ranks, danger_depth + emitted_blocks
            )
            if segment is None:
                # Candidates are rank-ordered and the threshold only grows
                # with emissions, so no later candidate can be due either.
                break
            _, due_ops = segment
            # Never emit across a deduplication hole: the batch is coalesced
            # into one contiguous store. The request keeps its due urgency;
            # the post-hole ops follow in a later batch.
            due_ops = _contiguous_front_run(due_ops)
            if self._fails_economy_gate(surviving):
                # Gate 3: the whole known prefix is below break-even. The due
                # front is about to die, which breaks the prefix chain for
                # the rest -- drop everything, not just the due segment.
                # Dropped blocks stay in the free queue, so they do not
                # extend the emission shift.
                result.dropped_short_prefix.extend(surviving)
                self._forget_content(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._prefix_broken.add(request_id)
                self._replace_pending(request_id, [], result)
                continue
            emitted = due_ops[:budget]
            result.ops_held_back += len(due_ops) - len(emitted)
            budget -= len(emitted)
            result.to_store.extend(emitted)
            self._forget_content(emitted)
            self._counters.emitted += len(emitted)
            emitted_blocks += sum(len(op.block_hashes) for op in emitted)
            # Mark in flight before updating pending state so that a request
            # fully drained by this emission is not released until the store
            # completion arrives via notify_stored().
            self._in_flight.add(request_id)
            remaining = self._pending[request_id][len(emitted) :]
            self._replace_pending(request_id, remaining, result)
        if result.ops_held_back:
            self._counters.throttled_drains += 1
        return result

    def notify_stored(self, request_id: str) -> bool:
        """Record that a request's in-flight store batch completed (or was
        drained by an unhealthy worker).

        Re-enables emission of the request's remaining pending operations.

        Args:
            request_id: The request whose store completion was reported.

        Returns:
            True if the request is finished and has nothing pending -- the
            caller may now safely tear down its session; False otherwise.
        """
        self._in_flight.discard(request_id)
        self._stale_in_flight.discard(request_id)
        if request_id in self._pending:
            return False
        if request_id in self._finished:
            self._finished.discard(request_id)
            self._prefix_broken.discard(request_id)
            return True
        return False

    def _pending_blocks(self) -> int:
        """GPU blocks covered by every buffered operation.

        An upper bound on how far one :meth:`collect_due` call can shift the
        free queue by pinning what it emits, since it can only emit
        operations that are already buffered.
        """
        return sum(len(op.block_hashes) for ops in self._pending.values() for op in ops)

    def _danger_depth(self) -> int:
        """Free-queue depth considered at risk within the horizon.

        Expected consumption below half a block over the whole horizon is
        treated as idle (depth 0): the EMA decays asymptotically after a
        burst and would otherwise keep a ceil'd depth of 1 forever.
        """
        per_step = max(self._blocks_per_step_ema, float(self._next_step_estimate))
        horizon_blocks = per_step * self._config.horizon_steps
        if horizon_blocks < 0.5:
            return 0
        return math.ceil(horizon_blocks)

    def _drop_evicted_suffix(
        self,
        request_id: str,
        ops: list[PendingStoreOp],
        result: DrainResult,
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch on any covered block means the block was evicted (or
        reallocated); the op and every later op of the request are dropped
        for prefix closure, and further admissions are rejected.
        """
        first_lost = len(ops)
        for index, op in enumerate(ops):
            if not self._snapshot_intact(op):
                first_lost = index
                break
        if first_lost == len(ops):
            return ops
        dropped = ops[first_lost:]
        result.dropped_evicted.extend(dropped)
        self._forget_content(dropped)
        self._counters.dropped_evicted += len(dropped)
        self._prefix_broken.add(request_id)
        surviving = ops[:first_lost]
        self._replace_pending(request_id, surviving, result)
        return surviving

    def _due_front_segment(
        self,
        ops: list[PendingStoreOp],
        ranks: dict[int, int],
        danger_depth: int,
    ) -> tuple[int, list[PendingStoreOp]] | None:
        """Find the front segment of ops to release for one request.

        An op is due when any of its blocks is within ``danger_depth`` of
        the free-queue head. Blocks absent from ``ranks`` (in use or pinned)
        are not at risk. The released segment runs from the front to the
        last due op, so a stored chunk never lacks its stored prefix.

        Returns:
            (min rank across the segment's due blocks, the segment), or
            None when no op is due.
        """
        if danger_depth <= 0:
            return None
        last_due = -1
        min_rank = danger_depth
        for index, op in enumerate(ops):
            op_ranks = [
                rank
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            due_ranks = [rank for rank in op_ranks if rank < danger_depth]
            if due_ranks:
                last_due = index
                min_rank = min(min_rank, min(due_ranks))
        if last_due < 0:
            return None
        return min_rank, ops[: last_due + 1]

    def _fails_economy_gate(self, ops: list[PendingStoreOp]) -> bool:
        """Gate 3: is the request's known prefix below break-even length?"""
        if self._config.min_prefix_tokens == 0:
            return False
        known_prefix = ops[-1].prefix_end_tokens
        return known_prefix < self._config.min_prefix_tokens

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        A mismatch on any block means it was evicted (or reallocated): the
        operation's data is lost and it must not be stored or deduplicated
        against.
        """
        return all(
            self._pool.block_hash(block_id) == snapshot
            for block_id, snapshot in op.block_hashes.items()
        )

    def _chain_intact(self, op: PendingStoreOp) -> bool:
        """Whether the op and its pending prefix chain are all intact.

        A valid deduplication cover must be more than intact itself: if an
        earlier pending op of its request has lost a block, the next drain
        drops the cover too (prefix closure), so it must not absorb a live
        copy of the content. Later siblings do not matter: their loss
        prefix-closes from their own position, leaving the cover storable.
        """
        for sibling in self._pending.get(op.request_id, []):
            if not self._snapshot_intact(sibling):
                return False
            if sibling is op:
                return True
        # Unreachable while the _pending_content invariant holds (every
        # cover is a pending op of its request); a missing cover is treated
        # as doomed, the safe direction.
        return False

    def _forget_content(self, ops: list[PendingStoreOp]) -> None:
        """Release the content keys of operations leaving the pending queue.

        Must be called on every path that removes operations from
        ``self._pending`` (emission, eviction drop, gate-3 drop, request
        drop), so identical content becomes admissible again. A key is only
        released if the leaving op still owns it: a corpse whose key was
        taken over by a live copy at admission must not release that copy's
        key.
        """
        for op in ops:
            key = _content_key(op)
            if self._pending_content.get(key) is op:
                del self._pending_content[key]

    def _replace_pending(
        self,
        request_id: str,
        remaining: list[PendingStoreOp],
        result: DrainResult,
    ) -> None:
        """Update a request's pending list, releasing it if drained empty."""
        if remaining:
            self._pending[request_id] = remaining
            return
        self._pending.pop(request_id, None)
        if request_id in self._finished and request_id not in self._in_flight:
            self._finished.discard(request_id)
            self._prefix_broken.discard(request_id)
            result.released_requests.append(request_id)
