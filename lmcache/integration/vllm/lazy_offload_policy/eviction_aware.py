# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered instead of being submitted as soon as their
tokens are computed, and are released when the GPU blocks holding their data
approach the free queue's eviction head, or when they hit the configured
deferral deadline. See ``docs/design/integration/vllm/lazy_offload_policy/
eviction_aware.md``.

This module is pure policy: vLLM types appear only in annotations and it
decides without acting. ``LazyOffloadManager`` owns execution -- pinning the
blocks of emitted operations and submitting them.
"""

# Standard
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol, cast
import math
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    PendingStoreItem,
)
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

DEFAULT_HORIZON_STEPS = 2.5

# Smoothing factor for the per-step block-consumption EMA. Not a config
# knob: the horizon (in steps) is the tunable quantity, the EMA only
# smooths noise.
_EMA_ALPHA = 0.3

# Rank given to a request released by the deferral deadline rather than by
# proximity to the free-queue head. It sorts ahead of every real rank (free
# queue positions, never negative) and is always below the emission
# threshold, so an overdue request is decided first and is due even on a
# step whose danger depth is zero.
_OVERDUE_RANK = -1

#: Minimum seconds between periodic counter-ledger log lines.
_STATS_LOG_INTERVAL_S = 5.0

#: Most dropped ops named in the aggregate drop line; the rest are counted.
_DROP_LOG_SAMPLE_OPS = 8

#: Counters that advance on every drain whether or not the policy decided
#: anything, so a change watcher has to skip them.
_COST_SENSOR_FIELDS = frozenset(
    {
        "drain_steps",
        "free_queue_blocks_read",
        "requests_validated",
        "blocks_validated",
    }
)


class BlockPoolReader(Protocol):
    """Read-only view over the GPU block pool required by the policy.

    The production implementation is :class:`GPUBlockPoolView`; tests
    provide a fake. Both must be side-effect free.
    """

    def free_queue_block_ids(self) -> Iterator[int]:
        """Iterate the free queue from the eviction head, lazily.

        Must stay lazy: it runs on the scheduler's critical path once per
        step while the queue holds every free block in the pool.

        Returns:
            Block ids in eviction order, the next victim first. A block's
            position is its rank.
        """
        ...

    def is_free(self, block_id: int) -> bool:
        """Whether the block currently sits in the free queue.

        Answers in O(1) what walking to the block's rank would answer in
        O(rank): pinning a block shifts every block behind it toward the
        head, and that shift counts whether or not the block was inside the
        window the step read.
        """
        ...

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the block's prefix-cache hash, or None if it holds none.

        None means never completed, or evicted and reallocated.
        """
        ...


class GPUBlockPoolView:
    """Production :class:`BlockPoolReader` over a vLLM ``BlockPool``."""

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a block pool obtained via ``bind_gpu_block_pool``."""
        self._block_pool = block_pool

    def free_queue_block_ids(self) -> Iterator[int]:
        """Walk the free queue's links from the eviction head, lazily.

        Yields ids rather than calling ``get_all_free_blocks()``, which
        materialises the whole queue on every step.

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

        vLLM keeps exactly the unreferenced blocks in the free queue, so the
        reference count answers queue membership without walking the list.
        The null block of a hybrid-attention model is excluded: it is popped
        out of the queue at construction and its count is not maintained.
        """
        block = self._block_pool.blocks[block_id]
        return block.ref_cnt == 0 and not block.is_null

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached."""
        return self._block_pool.blocks[block_id].block_hash


class _FreeQueueWindow:
    """The head of the free queue, materialised only as deep as it is used.

    A drain compares ranks against the danger depth, extended by the blocks
    the drain itself pins out of the queue. How deep that reaches is not
    known before the drain runs, so the window opens at the danger depth and
    is extended as emissions actually widen the threshold.
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

    def extend_to(self, depth: int) -> dict[int, int]:
        """Walk the queue until the window holds ``depth`` blocks.

        Args:
            depth: Target depth from the eviction head; at or below the
                current depth this reads nothing.

        Returns:
            The entries this call revealed, block id -> rank, empty when the
            target was already met or the queue ended first.
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


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        request_id: The vLLM request this operation belongs to.
        store_metadata: Ready-to-send store metadata from
            ``LMCacheMPRequestMetadata.GetStoreMetadata``; opaque here.
        block_hashes: Hash of every GPU block covering the operation's token
            range, snapshotted at admission. :meth:`EvictionAwareStoreQueue
            .add` rejects any None value, so a later mismatch against the
            pool means the block was recycled.
        prefix_start_tokens: Token index of the start of this range.
        prefix_end_tokens: Token index one past the end of this range.
        epoch: Store epoch that produced this operation.
        admitted_at_drain: Drain counter at admission. Its distance from the
            drain counter at emission is the deferral the policy achieved.
        admitted_at_time: Wall clock of the admitting step, read by the
            last :meth:`EvictionAwareStoreQueue.drain`.
    """

    request_id: str
    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: BlockHashes
    prefix_start_tokens: int
    prefix_end_tokens: int
    epoch: int = 0
    admitted_at_drain: int = 0
    admitted_at_time: float = 0.0


def _format_drop_sample(dropped: list[PendingStoreOp]) -> str:
    """Render dropped ops for the aggregate drop line, truncating the tail.

    Args:
        dropped: The operations dropped by one drain, in drop order.

    Returns:
        ``request (prefix N)`` for at most ``_DROP_LOG_SAMPLE_OPS`` ops, with
        a ``+N more`` suffix when the list was truncated.
    """
    sample = ", ".join(
        f"{op.request_id} (prefix {op.prefix_end_tokens})"
        for op in dropped[:_DROP_LOG_SAMPLE_OPS]
    )
    omitted = len(dropped) - _DROP_LOG_SAMPLE_OPS
    if omitted > 0:
        sample += f", +{omitted} more"
    return sample


@dataclass(frozen=True)
class LazyOffloadPolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        horizon_steps: How many scheduler steps of estimated block
            consumption count as "imminent eviction". Larger values drain
            earlier (closer to eager, fewer drops); smaller values drain
            later (longer deferral, more drops).
        max_drain_per_step: Upper bound on operations emitted per step, to
            bound the device-to-host burst. Must be >= 1. A prefilling
            request buffers about one operation per step, so a cap below the
            number of concurrently prefilling requests loses the backlog to
            eviction instead of merely delaying it; the sensor for having
            sized it wrong is ``LazyOffloadCounters.throttled_drains``.
        max_deferral_seconds: Upper bound on how long an operation may wait
            between admission and emission; 0.0 (the default) leaves emission
            entirely to the danger window. The window says when the block
            dies on the GPU, not when the content is asked for again, so set
            this below the reuse interval the workload has to beat. Sensor:
            ``LazyOffloadCounters.emitted_overdue`` against ``emitted``.
    """

    horizon_steps: float = DEFAULT_HORIZON_STEPS
    max_drain_per_step: int = 64
    max_deferral_seconds: float = 0.0

    @classmethod
    def from_configs(cls, configs: dict[str, ConfigValue]) -> "LazyOffloadPolicyConfig":
        """Read the tunables from vLLM's ``kv_connector_extra_config``.

        Args:
            configs: Connector extra configuration. Each field is read from
                the key ``lmcache.mp.lazy_offload_<field name>``; missing
                keys keep the field default.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: If a value is outside its documented range.
        """

        def number(name: str, default: float) -> float:
            key = f"lmcache.mp.lazy_offload_{name}"
            return float(cast("str | int | float", configs.get(key, default)))

        return cls(
            horizon_steps=number("horizon_steps", DEFAULT_HORIZON_STEPS),
            max_drain_per_step=int(number("max_drain_per_step", 64)),
            max_deferral_seconds=number("max_deferral_seconds", 0.0),
        )

    def __post_init__(self) -> None:
        """Validate field ranges.

        Raises:
            ValueError: If any field is outside its documented range.
        """
        if self.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {self.horizon_steps}")
        if self.max_drain_per_step < 1:
            raise ValueError(
                f"max_drain_per_step must be >= 1, got {self.max_drain_per_step}"
            )
        if self.max_deferral_seconds < 0.0:
            raise ValueError(
                f"max_deferral_seconds must be >= 0.0, got {self.max_deferral_seconds}"
            )


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    The operation counts close as a ledger: ``admitted`` equals the pending
    depth plus ``emitted`` plus every drop counter. The rest are weights and
    sensors beside that equation: ``dropped_evicted_tokens`` measures the
    losses in the unit the cache is sized in; ``emitted_overdue`` against
    ``emitted`` says whether the deferral deadline or the danger window is
    the binding clock; the ``*_deferral_drains`` sums divided by ``emitted``
    and ``dropped_evicted`` give the mean deferral in drains, the direct
    measure of what the policy buys; ``throttled_drains`` counts drains that
    left a due operation unemitted because ``max_drain_per_step`` ran out;
    and ``_COST_SENSOR_FIELDS`` names the sensors for the per-step decision's
    own cost, paid on the scheduler's critical path.
    """

    admitted: int = 0
    emitted: int = 0
    emitted_overdue: int = 0
    emitted_deferral_drains: int = 0
    dropped_evicted: int = 0
    dropped_evicted_tokens: int = 0
    dropped_deferral_drains: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_failed_store: int = 0
    dropped_id_reuse: int = 0
    throttled_drains: int = 0
    drain_steps: int = 0
    free_queue_blocks_read: int = 0
    requests_validated: int = 0
    blocks_validated: int = 0

    def decisions(self) -> tuple[int, ...]:
        """The counters that only a policy decision moves.

        Returns:
            Every counter except the cost sensors, in declaration order.
        """
        return tuple(
            value
            for name, value in asdict(self).items()
            if name not in _COST_SENSOR_FIELDS
        )


@dataclass
class DrainResult:
    """Internal outcome of one drain, before it is shaped for the caller.

    Attributes:
        to_store: Operations to submit now, ordered by eviction imminence
            across requests and by prefix order within a request. The
            connector must pin (``touch``) their blocks before the store and
            unpin after completion.
        dropped_evicted: Operations whose data was lost (block evicted or
            reallocated before drain), including later same-request
            operations dropped for prefix closure.
        emptied_requests: Requests whose pending operations became empty in
            this drain. The controller combines this with request phase and
            batch state before ending a session.
    """

    to_store: list[PendingStoreOp] = field(default_factory=list)
    dropped_evicted: list[PendingStoreOp] = field(default_factory=list)
    emptied_requests: list[str] = field(default_factory=list)


class _PendingOperations:
    """Own pending operations and every index derived from them."""

    def __init__(self) -> None:
        # Insertion order is admission order, and a request re-enters at the
        # back when it comes back after emptying, so the dict is also the
        # tie-break order the drain sorts by.
        self._by_request: dict[str, list[PendingStoreOp]] = {}
        self._requests_by_block: dict[int, set[str]] = {}
        self._requests_to_validate: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self._by_request)

    def contains_request(self, request_id: str) -> bool:
        return request_id in self._by_request

    def get(self, request_id: str) -> list[PendingStoreOp] | None:
        return self._by_request.get(request_id)

    def add(self, op: PendingStoreOp) -> None:
        """Add one admitted operation and index it by covered block."""
        self._by_request.setdefault(op.request_id, []).append(op)
        for block_id in op.block_hashes:
            self._requests_by_block.setdefault(block_id, set()).add(op.request_id)

    def _reindex(self, request_id: str, departed: list[PendingStoreOp]) -> None:
        """Drop the request from the blocks no surviving op still covers.

        Call after installing the surviving list: ops of one request share
        blocks across chunk boundaries, so a departed op's block is only
        forgotten when nothing left of the request covers it.
        """
        kept = {
            block_id
            for op in self._by_request.get(request_id, ())
            for block_id in op.block_hashes
        }
        for block_id in {b for op in departed for b in op.block_hashes} - kept:
            requests = self._requests_by_block[block_id]
            requests.discard(request_id)
            if not requests:
                del self._requests_by_block[block_id]

    def pop_request(self, request_id: str) -> list[PendingStoreOp]:
        """Atomically remove a request and all entries derived from its ops."""
        departed = self._by_request.pop(request_id, [])
        self._reindex(request_id, departed)
        self._requests_to_validate.discard(request_id)
        return departed

    def replace_request(
        self,
        request_id: str,
        departed: list[PendingStoreOp],
        remaining: list[PendingStoreOp],
    ) -> None:
        """Atomically remove a front/suffix and install the surviving list."""
        if remaining:
            self._by_request[request_id] = remaining
        else:
            self._by_request.pop(request_id, None)
            self._requests_to_validate.discard(request_id)
        self._reindex(request_id, departed)

    def num_ops(self) -> int:
        return sum(len(ops) for ops in self._by_request.values())

    def observe_allocations(self, allocated_block_ids: set[int]) -> None:
        """Mark requests whose snapshots require validation this step."""
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

    def admission_order(self) -> dict[str, int]:
        """Rank every pending request by admission, for drain tie-breaks."""
        return {request_id: order for order, request_id in enumerate(self._by_request)}

    def overdue_requests(self, now: float, max_age: float) -> set[str]:
        """Requests whose oldest pending operation has passed the deadline.

        Every pending request is examined: emission takes from the front, so
        a request admitted early may already have a young front, and the scan
        cannot stop at the first one still inside the deadline.

        Args:
            now: Current time in the caller's clock, in seconds.
            max_age: Deadline in seconds; the caller checks it is enabled.

        Returns:
            The ids of the requests at or past the deadline.
        """
        cutoff = now - max_age
        return {
            request_id
            for request_id, ops in self._by_request.items()
            if ops and ops[0].admitted_at_time <= cutoff
        }


class EvictionAwareStoreQueue:
    """Buffers store operations and releases them by eviction imminence.

    An operation is emitted when any of its blocks sits within the *danger
    depth* of the free queue -- the blocks the engine is expected to consume
    within ``horizon_steps`` steps -- or when its request passes
    ``max_deferral_seconds``. An idle engine never drains. Operations whose
    blocks are evicted before they come due are dropped, never stored stale.

    Not thread-safe: all methods must be called from the scheduler thread.
    """

    def __init__(self, config: LazyOffloadPolicyConfig, pool: BlockPoolReader) -> None:
        """Create an empty queue.

        Args:
            config: Policy tunables.
            pool: Read-only view of the GPU block pool.
        """
        self._config = config
        self._pool = pool
        # Primary pending storage and every derived index share one owner so
        # departure paths cannot update one without the other.
        self._pending_ops = _PendingOperations()
        # Prefix validity is a policy concern. Request phase, epochs, and
        # submitted batches are owned by the controller.
        self._broken_prefixes: set[str] = set()
        self._blocks_per_step_ema: float = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        # This step's clock, read once per drain and used to stamp
        # admissions; 0.0 until the first drain, which only matters when
        # max_deferral_seconds is enabled.
        self._now = 0.0
        self._counters = LazyOffloadCounters()
        # Periodic ledger logging: the last snapshot written and when.
        self._last_logged_decisions = LazyOffloadCounters().decisions()
        self._last_stats_log_time = 0.0

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
        epoch: int,
    ) -> None:
        """Buffer one store operation; see ``OffloadPolicy.add``."""
        existing = self._pending_ops.get(meta.request_id)
        if existing is not None and existing[0].epoch != epoch:
            raise RuntimeError(
                f"request {meta.request_id!r} mixed store epochs "
                f"{existing[0].epoch} and {epoch}"
            )
        if meta.request_id in self._broken_prefixes:
            self._counters.rejected_prefix_broken += 1
            # DEBUG, not INFO: the site that broke the chain already logged
            # the cause, and every later chunk lands here.
            logger.debug(
                "Lazy offload: skipping store for request %s tokens [%d, %d): "
                "the request's prefix chain is already broken",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return
        if any(block_hash is None for block_hash in block_hashes.values()):
            # The caller's tracker has already advanced past this range, so
            # the request's later chunks would be stored without their
            # prefix (unreachable): reject them like any other broken chain.
            self._broken_prefixes.add(meta.request_id)
            self._counters.rejected_unhashed += 1
            logger.warning(
                "Lazy offload: skipping store for request %s tokens [%d, %d) "
                "and every later chunk of it: covered blocks carry no "
                "prefix-cache hash, so their eviction could not be detected. "
                "Prefix caching is off, or a sliding-window or hybrid model "
                "left a hash-less null block in the block table. "
                "rejected_unhashed and rejected_prefix_broken count these.",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return
        self._pending_ops.add(
            PendingStoreOp(
                request_id=meta.request_id,
                store_metadata=meta,
                block_hashes=block_hashes,
                prefix_start_tokens=meta.op.start,
                prefix_end_tokens=meta.op.end,
                epoch=epoch,
                admitted_at_drain=self._counters.drain_steps,
                admitted_at_time=self._now,
            )
        )
        self._counters.admitted += 1

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
        """Break the request's prefix chain; see ``OffloadPolicy``.

        The controller calls this only for a batch from the current store
        epoch: a failure from a stale epoch cannot break the current prefix.
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

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Record the step's block pressure and release what it made due.

        Args:
            signals: The scheduler step's consumption and lifecycle inputs.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """
        self._now = time.monotonic()
        self._pending_ops.observe_allocations(signals.allocated_block_ids)
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * signals.new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(signals.new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = signals.est_next_step_blocks

        result = self._collect_due(signals.blocked_request_ids)
        self._log_drain(result)
        items: dict[str, PendingStoreItem] = {}
        for op in result.to_store:
            # add() rejects a second epoch while a request has ops buffered,
            # so every op of one request in one drain shares its epoch.
            item = items.setdefault(
                op.request_id,
                PendingStoreItem(request_id=op.request_id, epoch=op.epoch),
            )
            item.metadatas.append((op.store_metadata, op.block_hashes))
        return LazyOffloadDrain(
            items=list(items.values()),
            emptied_request_ids=result.emptied_requests,
        )

    def _collect_due(self, blocked_request_ids: set[str]) -> DrainResult:
        """Release the operations whose blocks face imminent eviction.

        Per request, the suffix from the first operation whose data is
        already lost is dropped, then the surviving front up to the last due
        operation is released (prefix closure). Emitting pins blocks out of
        the free queue and moves every block behind them toward the head, so
        each later candidate is tested against the danger depth extended by
        the blocks already emitted here, and the queue read follows the same
        threshold. A request past ``max_deferral_seconds`` is due regardless
        of the window and is decided first. See the design doc for the full
        contract.

        Args:
            blocked_request_ids: Requests that already have a store batch in
                flight. They are left pending, and any validation this step's
                allocations asked for stays pending with them.

        Returns:
            The operations to store and to drop this step.
        """
        result = DrainResult()
        if not self._pending_ops:
            return result
        self._counters.drain_steps += 1

        # Requests past the deferral deadline are due regardless of where
        # their blocks sit in the free queue: the deadline tracks when the
        # content is needed again, the danger depth when the block dies.
        overdue: set[str] = (
            self._pending_ops.overdue_requests(
                self._now, self._config.max_deferral_seconds
            )
            if self._config.max_deferral_seconds > 0.0
            else set()
        )

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
        admission_order = self._pending_ops.admission_order()

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
                in_window = [
                    rank
                    for op in surviving
                    for block_id in op.block_hashes
                    if (rank := window.ranks.get(block_id)) is not None
                ]
                if request_id in overdue:
                    rank = _OVERDUE_RANK
                elif in_window:
                    rank = min(in_window)
                else:
                    # No block inside the window: not due yet. A later
                    # widening can still bring one into view.
                    continue
                fresh.append((rank, request_id, surviving))
                candidate_ids.add(request_id)
            if not fresh:
                return
            # Most imminent first. Sorting only the undecided tail keeps the
            # order the emission loop relies on as the window widens.
            candidates.extend(fresh)
            candidates[cursor:] = sorted(
                candidates[cursor:],
                key=lambda cand: (cand[0], admission_order[cand[1]]),
            )

        # Only requests touched by this step's allocations or represented in
        # the window can have changed outcome. The reverse index avoids a
        # full pending-queue scan on every scheduler step.
        discover(self._pending_ops.requests_to_check(window.ranks) | overdue)

        ops_left = self._config.max_drain_per_step
        # Blocks this drain has pinned out of the free queue, and so the
        # distance every block behind them moves toward the head. A shared
        # block shifts the queue only on its first pin; a block that was
        # already out of the queue does not shift it at all.
        shift_blocks = 0
        held_back = 0
        pinned_free_blocks: set[int] = set()
        while ops_left > 0:
            threshold = danger_depth + shift_blocks
            if len(window.ranks) < threshold:
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
            if min_rank == _OVERDUE_RANK:
                # Past the deadline: the whole surviving front is due, with
                # no reference to the window. The drain budget still bounds
                # the burst -- an expired backlog is spread over steps, not
                # dumped in one.
                due_ops = surviving
            else:
                due_ops = self._due_front_segment(surviving, window.ranks, threshold)
                if not due_ops:
                    continue
            emitted = due_ops[:ops_left]
            ops_left -= len(emitted)
            held_back += len(due_ops) - len(emitted)
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            if min_rank == _OVERDUE_RANK:
                self._counters.emitted_overdue += len(emitted)
            self._counters.emitted_deferral_drains += self._deferral_drains(emitted)
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
        self._counters.free_queue_blocks_read += len(window.ranks)
        if held_back:
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
        return 0 if horizon_blocks < 0.5 else math.ceil(horizon_blocks)

    def _deferral_drains(self, ops: list[PendingStoreOp]) -> int:
        """Total drains these operations spent between admission and now."""
        now = self._counters.drain_steps
        return sum(now - op.admitted_at_drain for op in ops)

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
        self._counters.dropped_evicted_tokens += sum(
            op.prefix_end_tokens - op.prefix_start_tokens for op in dropped
        )
        self._counters.dropped_deferral_drains += self._deferral_drains(dropped)
        self._broken_prefixes.add(request_id)
        surviving = ops[:first_lost]
        self._replace_pending(request_id, dropped, surviving, result)
        return surviving

    def _due_front_segment(
        self,
        ops: list[PendingStoreOp],
        ranks: dict[int, int],
        threshold: int,
    ) -> list[PendingStoreOp]:
        """Find the front segment of ops to release for one request.

        An op is due when any of its blocks sits within ``threshold`` of the
        free-queue head. The segment runs from the front to the last due op,
        so a stored chunk never lacks its stored prefix.

        Returns:
            The segment, empty when no op of the request is due.
        """
        last_due = -1
        for index, op in enumerate(ops):
            if any(
                ranks.get(block_id, threshold) < threshold
                for block_id in op.block_hashes
            ):
                last_due = index
        return ops[: last_due + 1]

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        A mismatch on any block means it was evicted (or reallocated): the
        operation's data is lost and it must not be stored.
        """
        for block_id, snapshot in op.block_hashes.items():
            self._counters.blocks_validated += 1
            if self._pool.block_hash(block_id) != snapshot:
                return False
        return True

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

    def log_final_stats(self) -> None:
        """Log the cumulative counters as one INFO ``key=value`` line.

        Called at connector shutdown so the drop ledger (notably
        ``dropped_evicted``, the quality sensor) closes with an exact final
        value. Best-effort: a force-killed engine may die before reaching it,
        which is why :meth:`drain` also logs the ledger periodically.
        """
        logger.info("Lazy offload final counters: %s", self._format_ledger())

    def _log_drain(self, result: DrainResult) -> None:
        """Report one drain's evicted operations and the periodic ledger."""
        if result.dropped_evicted:
            # INFO, not DEBUG: each drop is one unit of cache-quality loss
            # and production logs rarely run at DEBUG. One aggregate line per
            # drain -- a burst that evicts a large pending queue at once must
            # not emit thousands of synchronous lines on the scheduler hot
            # path.
            logger.info(
                "Lazy offload: dropped %d store op(s): blocks evicted "
                "before drain (%s)",
                len(result.dropped_evicted),
                _format_drop_sample(result.dropped_evicted),
            )
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        """Log the counter ledger if it changed and the throttle allows.

        Runs on every drain, so the log converges to the true ledger whenever
        the engine takes a step at least ``_STATS_LOG_INTERVAL_S`` after the
        last change -- the shutdown hook alone is unreliable under a
        force-killed engine. The change test looks at the decision counters
        only: the cost sensors advance on every drain, so gating on them
        would log a line every interval for the life of the engine.
        """
        decisions = self._counters.decisions()
        if decisions == self._last_logged_decisions:
            return
        now = time.monotonic()
        if now - self._last_stats_log_time < _STATS_LOG_INTERVAL_S:
            return
        logger.info("Lazy offload counters: %s", self._format_ledger())
        self._last_logged_decisions = decisions
        self._last_stats_log_time = now

    def _format_ledger(self) -> str:
        """Render the ledger as one greppable ``key=value`` line body.

        Returns:
            Every counter followed by the pending depth, so the line closes
            as ``admitted == pending + emitted + every drop counter`` (over
            operation counts only; the token weight is not a term in it).
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        return f"{fields} pending={self._pending_ops.num_ops()}"
