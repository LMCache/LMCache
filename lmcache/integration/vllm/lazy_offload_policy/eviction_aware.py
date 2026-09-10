# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered and released when the GPU blocks holding their
data approach the free queue's eviction head, or when they hit the deferral
deadline. Pure policy: it decides, ``LazyOffloadManager`` acts. See
``docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md``.
"""

# Standard
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, cast
import math
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

DEFAULT_HORIZON_STEPS = 2.5

# Smooths the per-step block-consumption EMA; the horizon is the tunable.
_EMA_ALPHA = 0.3

# Rank of a request released only by the deferral deadline. Sorts behind
# every real rank (free-queue positions, bounded by the queue length), so a
# deadline that has passed still makes the request due on a step whose danger
# depth is zero, but never at the expense of a block about to be recycled:
# missing the deadline costs latency, missing an eviction costs the data.
_OVERDUE_RANK = 1 << 62

_STATS_LOG_INTERVAL_S = 5.0  # Minimum seconds between ledger log lines.
_DROP_LOG_SAMPLE_REQUESTS = 8  # Dropped requests named in the drop line.


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        store_metadata: The operation itself, handed back unchanged when it
            is emitted: the token range and the GPU blocks holding its KV.
        block_hashes: Prefix-cache hash of every block the range covers,
            snapshotted at admission. It never holds a None -- an unhashed
            operation is rejected rather than buffered -- so a later
            mismatch means the block was recycled and the data is gone.
        admitted_at_time: Monotonic clock read when the operation was
            buffered, against which ``max_deferral_seconds`` is measured.
    """

    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: BlockHashes
    admitted_at_time: float = 0.0


@dataclass(frozen=True)
class EvictionAwarePolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        horizon_steps: Scheduler steps of block consumption treated as
            imminent eviction. It sets the danger depth: the free-queue
            prefix the engine is expected to consume that soon.
        max_drain_per_step: Upper bound on the operations one drain emits,
            which caps the store work a single step can trigger.
        max_deferral_seconds: How long an operation may stay buffered before
            it is emitted regardless of eviction pressure. Zero leaves
            emission entirely to the danger window.

    The effect of raising each is documented for users in
    ``docs/source/mp/configuration.rst``.
    """

    horizon_steps: float = DEFAULT_HORIZON_STEPS
    max_drain_per_step: int = 64
    max_deferral_seconds: float = 0.0

    @classmethod
    def from_configs(
        cls, configs: dict[str, ConfigValue]
    ) -> "EvictionAwarePolicyConfig":
        """Read the tunables from ``lmcache.mp.lazy_offload_<field name>``.

        Args:
            configs: The connector's ``kv_connector_extra_config``. Only the
                ``lmcache.mp.lazy_offload_*`` keys are read; a missing key
                keeps the field default.

        Returns:
            The validated config.

        Raises:
            ValueError: If a value is outside its documented range.
        """

        def number(name: str, default: float) -> float:
            """Read one tunable as a float, falling back to ``default``."""
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

    The counts close as a ledger: ``admitted`` equals the pending depth plus
    ``emitted`` plus every ``dropped_*`` counter. ``emitted_overdue`` is a
    weight beside the equation, not a term in it.

    Attributes:
        admitted: Operations taken into the buffer.
        emitted: Operations released to the manager for submission.
        emitted_overdue: Of those, the ones released by the deferral
            deadline rather than by eviction pressure.
        dropped_evicted: Operations dropped because a covered block was
            recycled before they were emitted, plus the later operations of
            the same request that the loss made unreachable.
        rejected_unhashed: Operations refused at admission because a covered
            block carried no prefix-cache hash.
        rejected_prefix_broken: Operations refused at admission because the
            request's prefix chain was already broken.
        dropped_on_request_drop: Buffered operations discarded when vLLM
            preempted the request and freed its blocks.
        dropped_failed_store: Buffered operations discarded when a submitted
            store of the same request failed.
        dropped_id_reuse: Buffered operations of a finished request
            discarded when a new request took over its id.
    """

    admitted: int = 0
    emitted: int = 0
    emitted_overdue: int = 0
    dropped_evicted: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_failed_store: int = 0
    dropped_id_reuse: int = 0


class EvictionAwareStoreQueue(OffloadPolicy):
    """Buffers store operations and releases them by eviction imminence.

    An operation is emitted when one of its blocks sits within the *danger
    depth* of the free queue -- what the engine is expected to consume within
    ``horizon_steps`` -- or when its request passes the deferral deadline. An
    idle engine never drains; operations whose blocks are evicted before they
    come due are dropped, never stored stale. Scheduler thread only.
    """

    def __init__(self, config: EvictionAwarePolicyConfig, pool: "BlockPool") -> None:
        """Create an empty queue over ``pool`` with the tunables in ``config``.

        Args:
            config: The validated tunables of this policy.
            pool: The scheduler's GPU block pool. It is only ever read here:
                free-queue order and block hashes.
        """
        self._config = config
        self._pool = pool
        # Insertion order is admission order (a re-entering request goes to
        # the back), so the dict is also the drain's tie-break order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Prefix validity is policy; phase and batches are the manager's.
        self._broken_prefixes: set[str] = set()
        self._blocks_per_step_ema = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        self._now = 0.0  # This step's clock, set at the top of drain().
        self._counters = LazyOffloadCounters()
        self._last_logged = LazyOffloadCounters()
        self._last_stats_log_time = 0.0

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> None:
        """Buffer one store operation; see ``OffloadPolicy.add``.

        Args:
            meta: The store operation offered by the manager.
            block_hashes: Hash of every block the operation covers, read at
                admission. A None among them rejects the operation and
                breaks the request's prefix chain, since a block whose hash
                is gone cannot be checked for eviction later.
        """
        if meta.request_id in self._broken_prefixes:
            self._counters.rejected_prefix_broken += 1
            # DEBUG, not INFO: the site that broke the chain already logged
            # the cause, and every later operation lands here.
            logger.debug(
                "Lazy offload: skipping store for request %s tokens [%d, %d): "
                "the request's prefix chain is already broken",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return
        if any(block_hash is None for block_hash in block_hashes.values()):
            # The tracker has advanced past this range, so later operations
            # would be stored without their prefix: the chain is broken.
            self._broken_prefixes.add(meta.request_id)
            self._counters.rejected_unhashed += 1
            logger.warning(
                "Lazy offload: skipping store for request %s tokens [%d, %d) "
                "and every later operation of it: covered blocks carry no "
                "prefix-cache hash, so their eviction could not be detected. "
                "Prefix caching is off, or a sliding-window or hybrid model "
                "left a hash-less null block in the block table. "
                "rejected_unhashed and rejected_prefix_broken count these.",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return
        self._pending.setdefault(meta.request_id, []).append(
            PendingStoreOp(
                store_metadata=meta,
                block_hashes=block_hashes,
                # Read here rather than reusing the step clock: before
                # the first drain, and after an idle gap, that clock is
                # stale and would make a fresh op instantly overdue.
                admitted_at_time=time.monotonic(),
            )
        )
        self._counters.admitted += 1

    def has_pending_request(self, request_id: str) -> bool:
        """Whether this request currently owns buffered operations.

        Args:
            request_id: The request id to query.

        Returns:
            True while at least one of its operations is buffered.
        """
        return request_id in self._pending

    def drop_request(self, request_id: str) -> int:
        """Discard buffered operations invalidated by a tracker reset.

        The request keeps its id and restarts at token zero, so its
        broken-chain marker is cleared too.

        Args:
            request_id: The preempted request.

        Returns:
            The number of buffered operations discarded.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_on_request_drop += len(dropped)
        return len(dropped)

    def discard_for_reuse(self, request_id: str) -> None:
        """Discard the buffered state of the id's previous holder.

        Args:
            request_id: The reused request id, whose previous holder's
                buffer and broken-chain marker are dropped.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_id_reuse += len(dropped)

    def release_request(self, request_id: str) -> None:
        """Forget non-pending policy state after session teardown.

        Only the broken-chain marker is held outside the buffer, so this is
        what keeps finished request ids from accumulating.

        Args:
            request_id: The request whose session was torn down.
        """
        self._broken_prefixes.discard(request_id)

    def mark_store_failed(self, request_id: str) -> int:
        """Break the request's prefix chain; see ``OffloadPolicy``.

        The manager filters out orphaned batches first: a failure that
        predates a preemption reset or id reuse cannot break the prefix of
        the request generation now using the id.

        Args:
            request_id: The request whose submitted store failed.

        Returns:
            The number of buffered operations dropped.
        """
        dropped = self._pending.pop(request_id, [])
        self._counters.dropped_failed_store += len(dropped)
        self._broken_prefixes.add(request_id)
        return len(dropped)

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Record the step's block pressure and release what it made due.

        Per request: the suffix from the first op whose blocks were recycled
        is dropped, then the front up to the last due op is released (prefix
        closure). Requests in ``blocked_request_ids`` stay pending untouched:
        the worker holds one in-flight batch per request. Requests are served
        most imminent first until ``max_drain_per_step``; one whose deadline
        passed while no block of it sits in the window is due anyway, but it
        is served after every request holding one, so an expired backlog
        cannot spend the budget that a block about to be recycled needs.

        Args:
            signals: This step's allocation pressure, which updates the
                consumption estimate, plus the finished and blocked request
                ids. Finished ids are not read by this policy: imminence
                alone decides, whether or not the request is still running.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """
        self._now = time.monotonic()
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * signals.new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(signals.new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = signals.est_next_step_blocks
        if not self._pending:
            self._maybe_log_stats()
            return LazyOffloadDrain()

        # Depth zero makes nothing due and the queue is not walked; the loss
        # check below reads hashes, not ranks, and still runs.
        ranks = self._free_queue_ranks(self._danger_depth())
        overdue_cutoff = self._now - self._config.max_deferral_seconds
        drain = LazyOffloadDrain()
        dropped_ops = 0
        dropped_ids: list[str] = []
        # Due now, as (imminence rank, admission order, request id).
        candidates: list[tuple[int, int, str]] = []
        # Of those, the ones the deadline alone made due: they release their
        # whole buffer, not just the segment sitting in the danger window.
        overdue_ids: set[str] = set()
        for order, request_id in enumerate(list(self._pending)):
            if request_id in signals.blocked_request_ids:
                continue
            ops = self._pending[request_id]
            surviving = self._drop_evicted_suffix(request_id, ops)
            if len(surviving) != len(ops):
                dropped_ops += len(ops) - len(surviving)
                dropped_ids.append(request_id)
                if not surviving:
                    del self._pending[request_id]
                    drain.emptied_request_ids.append(request_id)
                    continue
                self._pending[request_id] = surviving
            overdue = (
                self._config.max_deferral_seconds > 0.0
                and surviving[0].admitted_at_time <= overdue_cutoff
            )
            in_window = [
                rank
                for op in surviving
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            if in_window:
                # A block the engine is about to recycle outranks a passed
                # deadline, and sizes the release too: a request past both
                # emits its due front segment, which always holds the op that
                # made it overdue. The ops behind it keep their own admission
                # clocks and come due on their own deadlines.
                candidates.append((min(in_window), order, request_id))
            elif overdue:
                candidates.append((_OVERDUE_RANK, order, request_id))
                overdue_ids.add(request_id)
        if dropped_ops:
            # INFO, not DEBUG: each drop is a unit of cache-quality loss and
            # production rarely runs at DEBUG. One line per drain, so a burst
            # evicting a large queue cannot flood the scheduler hot path.
            logger.info(
                "Lazy offload: dropped %d store op(s), blocks evicted before "
                "drain (%s)",
                dropped_ops,
                ", ".join(dropped_ids[:_DROP_LOG_SAMPLE_REQUESTS]),
            )
        candidates.sort()
        ops_left = self._config.max_drain_per_step
        for rank, _, request_id in candidates:
            if ops_left <= 0:
                break
            ops = self._pending[request_id]
            is_overdue = request_id in overdue_ids
            due = ops if is_overdue else self._due_front_segment(ops, ranks)
            emitted = due[:ops_left]
            ops_left -= len(emitted)
            self._counters.emitted += len(emitted)
            if is_overdue:
                self._counters.emitted_overdue += len(emitted)
            item = PendingStoreItem(request_id=request_id)
            item.metadatas.extend(
                (op.store_metadata, op.block_hashes) for op in emitted
            )
            drain.items.append(item)
            remaining = ops[len(emitted) :]
            if remaining:
                self._pending[request_id] = remaining
            else:
                del self._pending[request_id]
                drain.emptied_request_ids.append(request_id)
        self._maybe_log_stats()
        return drain

    def _danger_depth(self) -> int:
        """Free-queue depth considered at risk within the horizon.

        Below half a block over the whole horizon counts as idle (depth 0):
        the EMA decays asymptotically and would else pin a ceil'd 1 forever.

        Returns:
            The number of free-queue blocks the engine is expected to
            consume within ``horizon_steps``, zero on an idle engine.
        """
        per_step = max(self._blocks_per_step_ema, float(self._next_step_estimate))
        horizon_blocks = per_step * self._config.horizon_steps
        return 0 if horizon_blocks < 0.5 else math.ceil(horizon_blocks)

    def _free_queue_ranks(self, depth: int) -> dict[int, int]:
        """The first ``depth`` free-queue blocks, block id -> eviction rank.

        Walks the queue's links lazily rather than ``get_all_free_blocks()``,
        which materialises the whole queue every step.

        Args:
            depth: How many blocks to walk from the queue head. Zero or less
                returns an empty mapping without touching the queue.

        Returns:
            Block id to its position from the head, lowest evicted first.
            Shorter than ``depth`` if the queue is.

        Raises:
            RuntimeError: If the fake head has no successor.
        """
        ranks: dict[int, int] = {}
        if depth <= 0:
            return ranks
        block = self._pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        # The fake tail has no successor, which is what ends the walk.
        while block.next_free_block is not None and len(ranks) < depth:
            ranks[block.block_id] = len(ranks)
            block = block.next_free_block
        return ranks

    def _drop_evicted_suffix(
        self, request_id: str, ops: list[PendingStoreOp]
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch means the block was recycled; that op and every later
        op of the request go, and further admissions are rejected. The caller
        installs the survivors and reports the count.

        Args:
            request_id: The request the operations belong to, marked
                prefix-broken when anything is dropped.
            ops: Its buffered operations, in prefix order.

        Returns:
            The leading operations whose snapshots are still intact, which
            is ``ops`` itself when nothing was lost.
        """
        first_lost = next(
            (i for i, op in enumerate(ops) if not self._snapshot_intact(op)),
            len(ops),
        )
        if first_lost == len(ops):
            return ops
        self._counters.dropped_evicted += len(ops) - first_lost
        self._broken_prefixes.add(request_id)
        return ops[:first_lost]

    def _due_front_segment(
        self, ops: list[PendingStoreOp], ranks: dict[int, int]
    ) -> list[PendingStoreOp]:
        """The front segment up to the last op with a block in the window.

        Taking from the front keeps a stored operation's prefix stored.

        Args:
            ops: One request's buffered operations, in prefix order.
            ranks: The danger window, as returned by
                :meth:`_free_queue_ranks`. Only membership is read here.

        Returns:
            The due prefix of ``ops``, empty when none of them is due.
        """
        last_due = -1
        for index, op in enumerate(ops):
            if any(block_id in ranks for block_id in op.block_hashes):
                last_due = index
        return ops[: last_due + 1]

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        Args:
            op: The buffered operation to validate.

        Returns:
            False once any covered block was recycled, which means the
            operation's data is gone.
        """
        return all(
            self._pool.blocks[block_id].block_hash == snapshot
            for block_id, snapshot in op.block_hashes.items()
        )

    def log_final_stats(self) -> None:
        """Log the cumulative counters at shutdown as one INFO line.

        Best-effort: a force-killed engine may not reach it, which is why
        :meth:`drain` also logs the ledger periodically.
        """
        logger.info("Lazy offload final counters: %s", self._ledger_line())

    def _maybe_log_stats(self) -> None:
        """Log the counter ledger if it changed and the throttle allows."""
        if self._counters == self._last_logged:
            return
        if self._now - self._last_stats_log_time < _STATS_LOG_INTERVAL_S:
            return
        logger.info("Lazy offload counters: %s", self._ledger_line())
        self._last_logged = replace(self._counters)
        self._last_stats_log_time = self._now

    def _ledger_line(self) -> str:
        """Render the ledger as one greppable ``key=value`` line body.

        Ends with the pending depth, so the counts close as
        ``admitted == pending + emitted + every drop counter``.

        Returns:
            The counter names and values, space separated.
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        pending = sum(len(ops) for ops in self._pending.values())
        return f"{fields} pending={pending}"
