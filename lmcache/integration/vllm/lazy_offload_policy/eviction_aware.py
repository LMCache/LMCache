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
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol
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

    def free_queue_block_ids(self) -> Iterator[int]:
        """Iterate the free queue from the eviction head, lazily.

        The policy consumes only as many blocks as the step's decisions
        actually compare against, so the iterator must be lazy: this runs
        once per scheduler step on the critical path while the queue holds
        every free block in the pool, and materialising it would be
        O(free blocks) -- tens of thousands on a pool sized to fill the GPU.

        Returns:
            Block ids in eviction order, the next victim first. The position
            of a block in this sequence is its rank; blocks the caller never
            reaches, and blocks that are not in the free queue at all, mean
            the same thing to the policy -- not at risk this step.
        """
        ...

    def is_free(self, block_id: int) -> bool:
        """Whether the block currently sits in the free queue.

        Answers in O(1) what walking to the block's rank would answer in
        O(rank): a block the policy is about to pin leaves the queue and
        shifts every block behind it toward the head, and that shift has to
        be counted whether or not the block is inside the window the step
        happened to read.

        Args:
            block_id: The GPU block id to inspect.

        Returns:
            True when the block is evictable (in the free queue), False when
            it is referenced by a live request or otherwise out of the queue.
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

    All accesses are read-only. ``free_queue_block_ids`` is a generator, so
    it costs O(the depth the caller consumes) rather than O(the free queue),
    which matters because it runs on the scheduler thread once per step: a
    pool sized to fill the GPU keeps tens of thousands of blocks free.
    """

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a vLLM block pool obtained via ``bind_gpu_block_pool``.

        Args:
            block_pool: The scheduler's GPU block pool.
        """
        self._block_pool = block_pool

    def free_queue_block_ids(self) -> Iterator[int]:
        """Walk the free queue's links from the eviction head, lazily.

        Yields ids instead of calling ``get_all_free_blocks()``, which
        materialises the whole queue into a list (its own docstring in vLLM
        says it is mainly for testing), and instead of snapshotting a fixed
        depth, which would charge the step for ranks it never compares.

        Yields:
            Block ids in eviction order, the next victim first.

        Raises:
            RuntimeError: If the free list's fake head has no successor,
                i.e. the queue is not in the shape vLLM maintains.
        """
        block = self._block_pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        # The fake tail is the one node with no successor, so this stops
        # before reaching it.
        while block.next_free_block is not None:
            yield block.block_id
            block = block.next_free_block

    def is_free(self, block_id: int) -> bool:
        """Whether the block is an eviction candidate.

        vLLM keeps exactly the blocks with no live reference in the free
        queue (``BlockPool.touch`` removes a block from it on the same
        condition), so the reference count answers queue membership without
        walking the list. The null block of a hybrid-attention model is
        excluded: it is popped out of the queue at construction and its
        count is not maintained.

        Args:
            block_id: The GPU block id to inspect.
        """
        block = self._block_pool.blocks[block_id]
        return block.ref_cnt == 0 and not block.is_null

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached.

        Args:
            block_id: The GPU block id to inspect.
        """
        return self._block_pool.blocks[block_id].block_hash

    def num_free_blocks(self) -> int:
        """Return the number of blocks currently in the free queue."""
        return self._block_pool.get_num_free_blocks()


class _FreeQueueWindow:
    """The head of the free queue, materialised only as deep as it is used.

    A drain compares ranks against the danger depth, extended by the blocks
    the drain itself pins out of the queue. How deep that reaches is not
    known before the drain runs, and the number it *could* reach --
    ``max_drain_per_step`` operations at the largest pending size -- is the
    wrong thing to pay for: it is the per-step cost of a burst that almost
    never happens, charged to every step. This window starts at the danger
    depth and is extended block by block as emissions actually widen the
    threshold, so a step reads the ranks it compares and no more.
    """

    def __init__(self, block_ids: Iterator[int]) -> None:
        """Open an empty window over a lazy free-queue walk.

        Args:
            block_ids: Free-queue block ids in eviction order, consumed on
                demand. An empty iterator opens a window that never grows.
        """
        self._block_ids = block_ids
        self._exhausted = False
        self.ranks: dict[int, int] = {}

    def depth(self) -> int:
        """Return how many blocks the window currently holds."""
        return len(self.ranks)

    def extend_to(self, depth: int) -> dict[int, int]:
        """Walk the queue until the window holds ``depth`` blocks.

        Args:
            depth: Target depth, counted from the eviction head. A target at
                or below the current depth reads nothing.

        Returns:
            The entries this call revealed, block id -> rank, in ascending
            rank order. Empty when the window already reached the target or
            the queue ended first.
        """
        revealed: dict[int, int] = {}
        while not self._exhausted and len(self.ranks) < depth:
            block_id = next(self._block_ids, None)
            if block_id is None:
                self._exhausted = True
                break
            rank = len(self.ranks)
            self.ranks[block_id] = rank
            revealed[block_id] = rank
        return revealed


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
        epoch: Store epoch that produced this operation.
        cache_salt: The request's cache salt, part of the operation's
            content identity for deduplication (two requests with the same
            block hashes but different salts store under different keys).
    """

    request_id: str
    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: dict[int, "BlockHashWithGroupId"]
    prefix_start_tokens: int
    prefix_end_tokens: int
    epoch: int = 0
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


DEFAULT_HORIZON_STEPS = 2.5


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

    horizon_steps: float = DEFAULT_HORIZON_STEPS
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

    ``drain_steps``, ``free_queue_blocks_read``, ``requests_validated`` and
    ``blocks_validated`` are the cost sensors for the per-step decision
    itself, which runs on the scheduler's critical path and is therefore
    paid by every token's decode latency. Divided by ``drain_steps`` they
    give the mean free-queue depth a step walks and the mean number of
    block-hash comparisons it makes; both are properties of the workload
    and the pending backlog, not of anything the policy is configured with,
    so a rise in either is the signature of the decision loop -- rather
    than the offload itself -- becoming the cost.
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
    drain_steps: int = 0
    free_queue_blocks_read: int = 0
    requests_validated: int = 0
    blocks_validated: int = 0

    def decisions(self) -> tuple[int, ...]:
        """The counters that only a policy decision moves.

        The cost sensors advance on every drain whether or not the policy
        decided anything, so a caller watching the counters for change --
        the ledger log does -- has to watch these instead, or it never goes
        quiet on an engine that is merely running.

        Returns:
            Every counter except the four per-step cost sensors, in
            declaration order.
        """
        return (
            self.admitted,
            self.emitted,
            self.dropped_evicted,
            self.rejected_short_prefix,
            self.rejected_unhashed,
            self.rejected_prefix_broken,
            self.dropped_on_request_drop,
            self.dropped_failed_store,
            self.dropped_id_reuse,
            self.deduplicated,
            self.throttled_drains,
        )


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
        emptied_requests: Requests whose pending operations became empty in
            this drain. The controller combines this fact with request phase
            and submitted-batch state before ending a session.
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
    emptied_requests: list[str] = field(default_factory=list)
    ops_held_back: int = 0


class _PendingOperations:
    """Own pending operations and every index derived from them."""

    def __init__(self) -> None:
        self._by_request: dict[str, list[PendingStoreOp]] = {}
        self._content: dict[
            tuple[str, int, tuple["BlockHashWithGroupId", ...]], PendingStoreOp
        ] = {}
        self._requests_by_block: dict[int, set[str]] = {}
        self._request_block_refs: dict[tuple[str, int], int] = {}
        self._request_order: dict[str, int] = {}
        self._next_request_order = 0
        self._requests_to_validate: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self._by_request)

    def contains_request(self, request_id: str) -> bool:
        return request_id in self._by_request

    def get(self, request_id: str) -> list[PendingStoreOp] | None:
        return self._by_request.get(request_id)

    def covering_op(self, op: PendingStoreOp) -> PendingStoreOp | None:
        return self._content.get(_content_key(op))

    def add(self, op: PendingStoreOp) -> None:
        """Add one admitted operation and all of its index entries."""
        if op.request_id not in self._by_request:
            self._request_order[op.request_id] = self._next_request_order
            self._next_request_order += 1
        self._by_request.setdefault(op.request_id, []).append(op)
        self._content[_content_key(op)] = op
        for block_id in op.block_hashes:
            ref_key = (op.request_id, block_id)
            refs = self._request_block_refs.get(ref_key, 0) + 1
            self._request_block_refs[ref_key] = refs
            if refs == 1:
                self._requests_by_block.setdefault(block_id, set()).add(op.request_id)

    def _forget(self, ops: list[PendingStoreOp]) -> None:
        """Remove content and block index entries for departed ops."""
        for op in ops:
            key = _content_key(op)
            if self._content.get(key) is op:
                del self._content[key]
            for block_id in op.block_hashes:
                ref_key = (op.request_id, block_id)
                refs = self._request_block_refs[ref_key] - 1
                if refs > 0:
                    self._request_block_refs[ref_key] = refs
                    continue
                del self._request_block_refs[ref_key]
                requests = self._requests_by_block[block_id]
                requests.discard(op.request_id)
                if not requests:
                    del self._requests_by_block[block_id]

    def pop_request(self, request_id: str) -> list[PendingStoreOp]:
        """Atomically remove a request and all entries derived from its ops."""
        departed = self._by_request.pop(request_id, [])
        self._forget(departed)
        self._finish_request(request_id)
        return departed

    def replace_request(
        self,
        request_id: str,
        departed: list[PendingStoreOp],
        remaining: list[PendingStoreOp],
    ) -> None:
        """Atomically remove a front/suffix and install the surviving list."""
        self._forget(departed)
        if remaining:
            self._by_request[request_id] = remaining
            return
        self._by_request.pop(request_id, None)
        self._finish_request(request_id)

    def num_ops(self) -> int:
        return sum(len(ops) for ops in self._by_request.values())

    def _finish_request(self, request_id: str) -> None:
        self._request_order.pop(request_id, None)
        self._requests_to_validate.discard(request_id)

    def observe_allocations(
        self,
        allocated_block_ids: set[int] | None,
    ) -> None:
        """Mark requests whose snapshots require validation this step."""
        if allocated_block_ids is None:
            self._requests_to_validate.update(self._by_request)
            return
        for block_id in allocated_block_ids:
            self._requests_to_validate.update(self._requests_by_block.get(block_id, ()))

    def requests_for_blocks(self, block_ids: Iterable[int]) -> set[str]:
        return {
            request_id
            for block_id in block_ids
            for request_id in self._requests_by_block.get(block_id, ())
        }

    def requests_to_check(self, block_ids: Iterable[int]) -> set[str]:
        return self._requests_to_validate | self.requests_for_blocks(block_ids)

    def validation_complete(self, request_id: str) -> None:
        self._requests_to_validate.discard(request_id)

    def admission_order(self, request_id: str) -> int:
        return self._request_order[request_id]


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
        # Primary pending storage and every derived secondary index share one
        # owner so departure paths cannot update one without the other.
        self._pending_ops = _PendingOperations()
        # Prefix validity is a policy concern. Request phase, epochs, and
        # submitted batches are owned by the controller.
        self._broken_prefixes: set[str] = set()
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
        existing = self._pending_ops.get(op.request_id)
        if existing is not None and existing[0].epoch != op.epoch:
            raise RuntimeError(
                f"request {op.request_id!r} mixed store epochs "
                f"{existing[0].epoch} and {op.epoch}"
            )
        if op.request_id in self._broken_prefixes:
            self._counters.rejected_prefix_broken += 1
            return AdmitResult.REJECTED_PREFIX_BROKEN
        if any(block_hash is None for block_hash in op.block_hashes.values()):
            # The caller's tracker has already advanced past this range, so
            # the request's later chunks would be stored without their prefix
            # (unreachable): reject them like any other broken chain.
            self._broken_prefixes.add(op.request_id)
            self._counters.rejected_unhashed += 1
            return AdmitResult.REJECTED_UNHASHED_BLOCK
        covering = self._pending_ops.covering_op(op)
        if covering is not None and self._chain_intact(covering):
            self._counters.deduplicated += 1
            return AdmitResult.DEDUPLICATED
        # No covering op, or it is doomed (its own blocks were recycled, or
        # an earlier sibling's were, so the next drain prefix-closes over
        # it): buffer the live copy and make it the new cover. The doomed
        # op stays pending and is dropped by collect_due().
        self._pending_ops.add(op)
        self._counters.admitted += 1
        return AdmitResult.ADMITTED

    def observe_step(
        self,
        new_blocks_allocated: int,
        est_next_step_blocks: int,
        allocated_block_ids: set[int] | None = None,
    ) -> None:
        """Record one scheduler step's block-consumption signals.

        Must be called once per step, before :meth:`collect_due`.

        Args:
            new_blocks_allocated: GPU blocks newly allocated in the step
                that just finished scheduling (gross allocation, counted
                from the scheduler output).
            est_next_step_blocks: Estimated blocks the next step will
                allocate (e.g. scheduled tokens divided by block size).
            allocated_block_ids: Block ids allocated or resurrected in this
                step. Requests indexed by these ids are revalidated during
                the drain. None asks for a full validation pass and is kept
                for callers that cannot provide the incremental signal.
        """
        self._pending_ops.observe_allocations(allocated_block_ids)
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = est_next_step_blocks

    def has_pending_request(self, request_id: str) -> bool:
        """Whether this request currently owns buffered operations."""
        return self._pending_ops.contains_request(request_id)

    def drop_request(self, request_id: str) -> int:
        """Discard buffered operations invalidated by a tracker reset."""
        dropped = self._pending_ops.pop_request(request_id)
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_on_request_drop += len(dropped)
        return len(dropped)

    def discard_for_reuse(self, request_id: str) -> int:
        """Discard a finished predecessor's buffered policy state."""
        dropped = self._pending_ops.pop_request(request_id)
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_id_reuse += len(dropped)
        return len(dropped)

    def release_request(self, request_id: str) -> None:
        """Forget non-pending policy state after current-session teardown."""
        self._broken_prefixes.discard(request_id)

    def mark_store_failed(self, request_id: str) -> int:
        """Record that the request's in-flight store batch failed.

        The request's stored prefix chain is broken: its held-back pending
        operations are dropped (stored without the failed prefix they would
        be unreachable) and further admissions are rejected. Receipt and
        request lifecycle remain entirely controller-owned.

        The controller only calls this method for a batch from the current
        store epoch. Failures from an epoch made stale by reset or id reuse
        are filtered there because they do not break the current prefix.

        Args:
            request_id: The request whose store failed.

        Returns:
            The number of pending operations dropped.
        """
        dropped = self._pending_ops.pop_request(request_id)
        self._counters.dropped_failed_store += len(dropped)
        self._broken_prefixes.add(request_id)
        return len(dropped)

    def num_pending_ops(self) -> int:
        """Return the total number of buffered store operations."""
        return self._pending_ops.num_ops()

    def stats(self) -> LazyOffloadCounters:
        """Return a copy of the cumulative policy counters."""
        return replace(self._counters)

    def collect_due(self, blocked_request_ids: set[str] | None = None) -> DrainResult:
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

        The free-queue read follows that threshold instead of anticipating
        it: the window opens at the danger depth and widens only by a shift
        an emission has already caused, so a step reads the ranks its
        decisions compare rather than the ranks a full-budget drain could
        have compared. Whether an emitted block shifts the queue is asked of
        the pool directly, not of the window, so a pin deeper than the
        window still counts and the widening cannot stall behind itself.

        Args:
            blocked_request_ids: Requests that already have a store batch in
                flight. They are left pending, and any validation this
                step's allocations asked for stays pending with them.

        Returns:
            The operations to store and to drop this step; see
            :class:`DrainResult`.
        """
        result = DrainResult()
        blocked_request_ids = blocked_request_ids or set()
        if not self._pending_ops:
            return result
        self._counters.drain_steps += 1

        danger_depth = self._danger_depth()
        # A zero danger depth makes nothing due, so the free queue is not
        # walked at all -- the loss check below reads block hashes, not
        # ranks, and still runs.
        window = _FreeQueueWindow(
            self._pool.free_queue_block_ids() if danger_depth > 0 else iter(())
        )
        window.extend_to(danger_depth)

        # Requests due now, ascending by eviction imminence, and the cursor
        # of the first one this drain has not decided yet.
        candidates: list[tuple[int, str, list[PendingStoreOp]]] = []
        cursor = 0
        candidate_ids: set[str] = set()
        # Survivors of the loss check, kept so that a request the window
        # reveals later is not validated a second time in the same step.
        surviving_by_request: dict[str, list[PendingStoreOp]] = {}

        def discover(request_ids: set[str]) -> None:
            """Validate these requests and queue the ones now due.

            Args:
                request_ids: Requests whose outcome may have changed --
                    touched by this step's allocations, or holding a block
                    the window just revealed. Ones already queued as
                    candidates are skipped.
            """
            fresh: list[tuple[int, str, list[PendingStoreOp]]] = []
            for request_id in request_ids:
                if request_id in candidate_ids:
                    continue
                surviving = surviving_by_request.get(request_id)
                if surviving is None:
                    ops = self._pending_ops.get(request_id)
                    if not ops:
                        self._pending_ops.validation_complete(request_id)
                        surviving_by_request[request_id] = []
                        continue
                    if request_id in blocked_request_ids:
                        # One in-flight store batch per request (worker
                        # constraint). Keep an allocation-triggered
                        # validation pending: after the receipt, the
                        # held-back ops still need their snapshots checked
                        # even if their recycled blocks are no longer free.
                        continue
                    surviving = self._drop_evicted_suffix(request_id, ops, result)
                    self._pending_ops.validation_complete(request_id)
                    surviving_by_request[request_id] = surviving
                if not surviving:
                    continue
                op_ranks = [
                    rank
                    for op in surviving
                    for block_id in op.block_hashes
                    if (rank := window.ranks.get(block_id)) is not None
                ]
                if not op_ranks:
                    # No block inside the window: the request cannot be due
                    # yet. A later widening can still bring one into view.
                    continue
                fresh.append((min(op_ranks), request_id, surviving))
                candidate_ids.add(request_id)
            if not fresh:
                return
            # Most imminent first. Sorting only the undecided tail keeps the
            # order the emission loop relies on as the window widens.
            candidates.extend(fresh)
            candidates[cursor:] = sorted(
                candidates[cursor:],
                key=lambda cand: (
                    cand[0],
                    self._pending_ops.admission_order(cand[1]),
                ),
            )

        # Only requests touched by this step's allocations or represented in
        # the window can have changed outcome. The reverse index avoids a
        # full pending-queue scan on every scheduler step.
        discover(self._pending_ops.requests_to_check(window.ranks))

        budget = self._config.max_drain_per_step
        # Blocks this drain has pinned out of the free queue, and so the
        # distance every block behind them moves toward the head. A shared
        # block shifts the queue only on its first pin; a block that was
        # already out of the queue does not shift it at all.
        shift_blocks = 0
        pinned_free_blocks: set[int] = set()
        while budget > 0:
            threshold = danger_depth + shift_blocks
            if window.depth() < threshold:
                revealed = window.extend_to(threshold)
                if revealed:
                    discover(self._pending_ops.requests_for_blocks(revealed))
            if cursor >= len(candidates):
                break
            min_rank, request_id, surviving = candidates[cursor]
            if min_rank >= threshold:
                # Candidates are rank-ordered and the threshold only grows
                # with emissions, so no later candidate can be due either.
                break
            cursor += 1
            segment = self._due_front_segment(surviving, window.ranks, threshold)
            if segment is None:
                continue
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
                self._counters.rejected_short_prefix += len(surviving)
                self._broken_prefixes.add(request_id)
                self._replace_pending(request_id, surviving, [], result)
                continue
            emitted = due_ops[:budget]
            result.ops_held_back += len(due_ops) - len(emitted)
            budget -= len(emitted)
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            newly_pinned = {
                block_id
                for op in emitted
                for block_id in op.block_hashes
                if block_id not in pinned_free_blocks and self._pool.is_free(block_id)
            }
            pinned_free_blocks.update(newly_pinned)
            shift_blocks += len(newly_pinned)
            remaining = surviving[len(emitted) :]
            self._replace_pending(request_id, emitted, remaining, result)
        self._counters.free_queue_blocks_read += window.depth()
        if result.ops_held_back:
            self._counters.throttled_drains += 1
        return result

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
        self._counters.requests_validated += 1
        first_lost = len(ops)
        for index, op in enumerate(ops):
            if not self._snapshot_intact(op):
                first_lost = index
                break
        if first_lost == len(ops):
            return ops
        dropped = ops[first_lost:]
        result.dropped_evicted.extend(dropped)
        self._counters.dropped_evicted += len(dropped)
        self._broken_prefixes.add(request_id)
        surviving = ops[:first_lost]
        self._replace_pending(request_id, dropped, surviving, result)
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
        checked = 0
        for block_id, snapshot in op.block_hashes.items():
            checked += 1
            if self._pool.block_hash(block_id) != snapshot:
                self._counters.blocks_validated += checked
                return False
        self._counters.blocks_validated += checked
        return True

    def _chain_intact(self, op: PendingStoreOp) -> bool:
        """Whether the op and its pending prefix chain are all intact.

        A valid deduplication cover must be more than intact itself: if an
        earlier pending op of its request has lost a block, the next drain
        drops the cover too (prefix closure), so it must not absorb a live
        copy of the content. Later siblings do not matter: their loss
        prefix-closes from their own position, leaving the cover storable.
        """
        for sibling in self._pending_ops.get(op.request_id) or []:
            if not self._snapshot_intact(sibling):
                return False
            if sibling is op:
                return True
        # Unreachable while the content-index invariant holds (every
        # cover is a pending op of its request); a missing cover is treated
        # as doomed, the safe direction.
        return False

    def _replace_pending(
        self,
        request_id: str,
        departed: list[PendingStoreOp],
        remaining: list[PendingStoreOp],
        result: DrainResult,
    ) -> None:
        """Replace pending ops and report requests whose buffer became empty."""
        self._pending_ops.replace_request(request_id, departed, remaining)
        if not remaining:
            result.emptied_requests.append(request_id)
