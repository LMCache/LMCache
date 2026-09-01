# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered and released when the GPU blocks holding their
data approach the free queue's eviction head, or when they hit the deferral
deadline. Pure policy: it decides, ``LazyOffloadManager`` acts. See
``docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md``.
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

# Smooths the per-step block-consumption EMA; the horizon is the tunable.
_EMA_ALPHA = 0.3

# Rank of a request released by the deferral deadline. Sorts ahead of every
# real rank (free-queue positions, never negative) and below every emission
# threshold, so it is due even on a step whose danger depth is zero.
_OVERDUE_RANK = -1

_STATS_LOG_INTERVAL_S = 5.0  # Minimum seconds between ledger log lines.
_DROP_LOG_SAMPLE_OPS = 8  # Dropped ops named in the drop line; rest counted.

#: Counters that advance on every drain whatever the policy decided, so a
#: change watcher has to skip them.
_COST_SENSOR_FIELDS = frozenset(
    {"drain_steps", "free_queue_blocks_read", "requests_validated", "blocks_validated"}
)


class BlockPoolReader(Protocol):
    """Read-only, side-effect-free view of the GPU block pool.

    Production implementation is :class:`GPUBlockPoolView`; tests fake it.
    """

    def free_queue_block_ids(self) -> Iterator[int]:
        """Iterate the free queue lazily from the eviction head.

        Block ids come in eviction order, next victim first; a block's
        position is its rank. One walk per step covers the whole free pool,
        on the scheduler's critical path, so it must stay lazy.
        """
        ...

    def is_free(self, block_id: int) -> bool:
        """Whether the block currently sits in the free queue.

        O(1) where walking to its rank is O(rank); a pin shifts the queue
        whether or not the block was inside the window the step read.
        """
        ...

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the block's prefix-cache hash, None if uncached or recycled."""
        ...


class GPUBlockPoolView:
    """Production :class:`BlockPoolReader` over a vLLM ``BlockPool``."""

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a block pool obtained via ``bind_gpu_block_pool``."""
        self._block_pool = block_pool

    def free_queue_block_ids(self) -> Iterator[int]:
        """Walk the free queue's links lazily, yielding ids in eviction order.

        Avoids ``get_all_free_blocks()``, which materialises the whole queue
        every step. Raises RuntimeError if the fake head has no successor.
        """
        block = self._block_pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        while block.next_free_block is not None:  # The fake tail has none.
            yield block.block_id
            block = block.next_free_block

    def is_free(self, block_id: int) -> bool:
        """Whether the block is an eviction candidate.

        vLLM keeps exactly the unreferenced blocks in the free queue, so the
        reference count answers membership. A hybrid model's null block is
        excluded: popped out at construction, its count is not maintained.
        """
        block = self._block_pool.blocks[block_id]
        return block.ref_cnt == 0 and not block.is_null

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached."""
        return self._block_pool.blocks[block_id].block_hash


class _FreeQueueWindow:
    """The head of the free queue, materialised only as deep as it is used.

    Emissions pin blocks out of the queue and so widen the threshold a drain
    compares against, by an amount not known before the drain runs.
    """

    def __init__(self, block_ids: Iterator[int]) -> None:
        """Open an empty window over ``block_ids``, consumed on demand."""
        self._block_ids = block_ids
        self._exhausted = False
        self.ranks: dict[int, int] = {}

    def extend_to(self, depth: int) -> dict[int, int]:
        """Walk the queue until the window holds ``depth`` blocks, at most.

        Returns:
            The entries this call revealed, block id -> rank.
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

    ``block_hashes`` snapshots every covered block at admission and never
    holds a None, so a later mismatch means the block was recycled; the
    ``admitted_at_*`` stamps measure the deferral the policy achieved.
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
    """Render up to ``_DROP_LOG_SAMPLE_OPS`` dropped ops, then ``+N more``."""
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

    Each field is documented for users, with its sensor counter and the
    effect of raising it, in ``docs/source/mp/configuration.rst``.
    """

    horizon_steps: float = DEFAULT_HORIZON_STEPS
    max_drain_per_step: int = 64
    max_deferral_seconds: float = 0.0

    @classmethod
    def from_configs(cls, configs: dict[str, ConfigValue]) -> "LazyOffloadPolicyConfig":
        """Read the tunables from ``lmcache.mp.lazy_offload_<field name>``.

        Missing keys keep the field default.

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
        """Validate field ranges, raising ValueError on any out of range."""
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
    sensors beside it, ``_COST_SENSOR_FIELDS`` being the ones that measure
    the per-step decision's own cost. The design doc reads them one by one.
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
        """Every counter but the cost sensors, in declaration order."""
        return tuple(
            value
            for name, value in asdict(self).items()
            if name not in _COST_SENSOR_FIELDS
        )


@dataclass
class DrainResult:
    """Internal outcome of one drain, before it is shaped for the caller.

    ``to_store`` is ordered by eviction imminence across requests and by
    prefix within one; its blocks must be pinned before the store and
    unpinned after. ``dropped_evicted`` covers ops whose blocks were recycled
    before the drain, prefix-closure victims included.
    """

    to_store: list[PendingStoreOp] = field(default_factory=list)
    dropped_evicted: list[PendingStoreOp] = field(default_factory=list)
    emptied_requests: list[str] = field(default_factory=list)


class _PendingOperations:
    """Own pending operations and every index derived from them."""

    def __init__(self) -> None:
        # Insertion order is admission order (a re-entering request goes to
        # the back), so the dict is also the drain's tie-break order.
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

        Call after installing the survivors: ops of one request share blocks
        across chunk boundaries.
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
        """Requests whose oldest pending op is ``max_age`` seconds old or more.

        Every pending request is scanned: emission takes from the front, so
        one admitted early may already have a young front.
        """
        cutoff = now - max_age
        return {
            request_id
            for request_id, ops in self._by_request.items()
            if ops and ops[0].admitted_at_time <= cutoff
        }


class EvictionAwareStoreQueue:
    """Buffers store operations and releases them by eviction imminence.

    An operation is emitted when one of its blocks sits within the *danger
    depth* of the free queue -- what the engine is expected to consume within
    ``horizon_steps`` -- or when its request passes the deferral deadline. An
    idle engine never drains; operations whose blocks are evicted before they
    come due are dropped, never stored stale. Scheduler thread only.
    """

    def __init__(self, config: LazyOffloadPolicyConfig, pool: BlockPoolReader) -> None:
        """Create an empty queue over ``pool`` with the tunables in ``config``."""
        self._config = config
        self._pool = pool
        # Pending ops and every index derived from them share one owner, so a
        # departure path cannot update one without the other.
        self._pending_ops = _PendingOperations()
        # Prefix validity is policy; phase, epochs and batches are the
        # controller's.
        self._broken_prefixes: set[str] = set()
        self._blocks_per_step_ema: float = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        self._now = 0.0  # This step's clock, read once per drain.
        self._counters = LazyOffloadCounters()
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
            # The tracker has advanced past this range, so later chunks would
            # be stored without their prefix: treat the chain as broken.
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

        Called only for a current-epoch batch: a stale failure cannot break
        the current prefix.
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
            # add() rejects a second epoch while ops are buffered, so all of
            # one request's ops in one drain share an epoch.
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

        Per request: the suffix from the first lost op is dropped, then the
        front up to the last due op is released (prefix closure). Emitting
        pins blocks out of the free queue, so each later candidate is tested
        against a danger depth extended by what this drain already emitted.
        Requests in ``blocked_request_ids`` stay pending, validation with
        them. See the design doc for the full contract.
        """
        result = DrainResult()
        if not self._pending_ops:
            return result
        self._counters.drain_steps += 1
        # Past the deadline is due wherever the blocks sit: the deadline
        # tracks when the content is wanted, the depth when the block dies.
        overdue: set[str] = (
            self._pending_ops.overdue_requests(
                self._now, self._config.max_deferral_seconds
            )
            if self._config.max_deferral_seconds > 0.0
            else set()
        )

        danger_depth = self._danger_depth()
        # Depth zero makes nothing due, so the queue is not walked; the loss
        # check below reads hashes, not ranks, and still runs.
        window = _FreeQueueWindow(
            self._pool.free_queue_block_ids() if danger_depth > 0 else iter(())
        )
        window.extend_to(danger_depth)

        # Due now, ascending by imminence, plus the cursor of the first one
        # this drain has not decided yet.
        candidates: list[tuple[int, str, list[PendingStoreOp]]] = []
        cursor = 0
        candidate_ids: set[str] = set()
        # Loss-check survivors, so a request the window reveals later is not
        # validated twice in one step.
        surviving_by_request: dict[str, list[PendingStoreOp]] = {}
        admission_order = self._pending_ops.admission_order()

        def discover(request_ids: set[str]) -> None:
            """Validate these requests and queue the ones now due.

            Requests already queued as candidates are skipped.
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
                        # One in-flight batch per request (worker constraint).
                        # The allocation-triggered validation stays pending:
                        # after the receipt the held-back ops still need their
                        # snapshots checked.
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
                    continue  # Not in the window; a widening may reveal it.
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

        # Only requests touched by this step's allocations or shown in the
        # window can have changed outcome; the reverse index spares a full
        # pending scan per step.
        discover(self._pending_ops.requests_to_check(window.ranks) | overdue)
        ops_left = self._config.max_drain_per_step
        # Blocks pinned out of the queue by this drain, and so the distance
        # every block behind them moves headward. A shared block shifts only
        # on its first pin; one already out of the queue does not shift.
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
                # Past the deadline the whole surviving front is due, window
                # or not; the budget still spreads an expired backlog over
                # steps rather than dumping it in one.
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

        Below half a block over the whole horizon counts as idle (depth 0):
        the EMA decays asymptotically and would else pin a ceil'd 1 forever.
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

        A hash mismatch means the block was recycled; that op and every later
        op of the request go, and further admissions are rejected.
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
        """The front segment up to the last op due within ``threshold``.

        An op is due when one of its blocks sits that close to the free-queue
        head. Taking from the front keeps a stored chunk's prefix stored;
        the segment is empty when no op of the request is due.
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
        """Whether every covered block still holds its admission-time hash."""
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
        """Log the cumulative counters at shutdown as one INFO line.

        Best-effort: a force-killed engine may not reach it, which is why
        :meth:`drain` also logs the ledger periodically.
        """
        logger.info("Lazy offload final counters: %s", self._format_ledger())

    def _log_drain(self, result: DrainResult) -> None:
        """Report one drain's evicted operations and the periodic ledger."""
        if result.dropped_evicted:
            # INFO, not DEBUG: each drop is a unit of cache-quality loss and
            # production rarely runs at DEBUG. One line per drain, so a burst
            # evicting a large queue cannot flood the scheduler hot path.
            logger.info(
                "Lazy offload: dropped %d store op(s): blocks evicted "
                "before drain (%s)",
                len(result.dropped_evicted),
                _format_drop_sample(result.dropped_evicted),
            )
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        """Log the counter ledger if it changed and the throttle allows.

        The change test reads the decision counters only: the cost sensors
        advance every drain and would log a line for the engine's lifetime.
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

        Ends with the pending depth, so the operation counts close as
        ``admitted == pending + emitted + every drop counter``.
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        return f"{fields} pending={self._pending_ops.num_ops()}"
