# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered and released when the GPU blocks holding their
data approach the free queue's eviction head. Pure policy: it decides,
``LazyOffloadManager`` acts. See
``docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md``.
"""

# Standard
from collections.abc import Container, Iterator
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Caps the store work one drain triggers.
MAX_DRAIN_PER_STEP = 64

_STATS_LOG_INTERVAL_S = 5.0  # Minimum seconds between ledger log lines.
_DROP_LOG_SAMPLE_REQUESTS = 8  # Dropped requests named in the drop line.


def _pending_item(request_id: str, ops: "list[PendingStoreOp]") -> PendingStoreItem:
    """Hand one request's buffered ops to the manager, in prefix order.

    Args:
        request_id: The request the ops belong to.
        ops: The ops, each paired in the item with its hash snapshot.

    Returns:
        The item carrying every op of ``ops``.
    """
    item = PendingStoreItem(request_id=request_id)
    item.metadatas.extend((op.store_metadata, op.block_hashes) for op in ops)
    return item


def _min_rank(ops: "list[PendingStoreOp]", ranks: dict[int, int]) -> int | None:
    """The most imminent rank any block of ``ops`` has.

    Args:
        ops: One request's buffered operations.
        ranks: Block id to its position from the free-queue head.

    Returns:
        The lowest rank among the blocks of ``ops`` found in ``ranks``, or
        None when none of them is there.
    """
    return min(
        (
            ranks[block_id]
            for op in ops
            for block_id in op.block_hashes
            if block_id in ranks
        ),
        default=None,
    )


def _due_front_length(ops: "list[PendingStoreOp]", window: "Container[int]") -> int:
    """Length of the front segment up to the last op with a block in ``window``.

    Taking from the front keeps a stored operation's prefix stored.

    Args:
        ops: One request's buffered operations, in prefix order.
        window: The block ids in reach. Only membership is read.

    Returns:
        How many leading ops of ``ops`` are due, zero when none of them is.
    """
    for index in range(len(ops) - 1, -1, -1):
        if any(block_id in window for block_id in ops[index].block_hashes):
            return index + 1
    return 0


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        store_metadata: The operation itself, handed back unchanged when it
            is emitted: the token range and the GPU blocks holding its KV.
        block_hashes: Prefix-cache hash of every block the range covers,
            snapshotted at admission. It never holds a None -- the manager
            replaced each hash-less block by the null block before buffering
            -- so a later mismatch means the block was recycled and the data
            is gone.
    """

    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: BlockHashes


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    The counts close as a ledger: ``admitted`` plus ``requeued`` equals the
    pending depth plus ``emitted`` plus every ``dropped_*`` counter.
    The ``bound_*`` counters stand beside the equation, not in it.

    Attributes:
        admitted: Operations taken into the buffer.
        emitted: Operations released to the manager for submission.
        dropped_evicted: Operations dropped because a covered block was
            recycled before they were emitted, plus the later operations of
            the same request that the loss made unreachable.
        rejected_prefix_broken: Operations refused at admission because the
            request's prefix chain was already broken.
        dropped_on_request_drop: Buffered operations discarded when vLLM
            preempted the request and freed its blocks.
        requeued: Emitted operations whose store failed, buffered again to
            be retried. A retried operation counts as emitted once more.
        bound_exceeded: Steps whose running requests allocated more blocks
            than the previous drain bounded. Nonzero means the danger depth
            is not a bound for this model, and buffered blocks may have
            been recycled unseen.
        bound_blocks: The previous drain's bound, summed over the steps it
            was checked against. Over ``bound_allocated_blocks`` it is the
            bound's overcount: how many blocks it reserves per one allocated.
        bound_allocated_blocks: Blocks the running requests allocated,
            summed over the same steps.
        dropped_id_reuse: Buffered operations of a finished request
            discarded when a new request took over its id.
        dropped_failed_store: Buffered operations, the last failed batch
            among them, discarded when the manager gave up the request's
            stores after repeated failures.
    """

    admitted: int = 0
    emitted: int = 0
    requeued: int = 0
    bound_exceeded: int = 0
    bound_blocks: int = 0
    bound_allocated_blocks: int = 0
    dropped_evicted: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_id_reuse: int = 0
    dropped_failed_store: int = 0


class EvictionAwareStoreQueue(OffloadPolicy):
    """Buffers store operations and releases them by eviction imminence.

    One rule decides when to store: an operation is due once one of its
    blocks sits within the *danger depth* of the free queue -- what the
    running requests and one admission can allocate before the next drain
    -- and nothing else makes it due.
    An engine with no requests never drains; operations whose blocks are
    evicted before they come due are dropped, never stored stale.
    Scheduler thread only.
    """

    def __init__(self, pool: "BlockPool") -> None:
        """Create an empty queue over ``pool``.

        Args:
            pool: The scheduler's GPU block pool. It is only ever read here:
                free-queue order and block hashes.
        """
        self._pool = pool
        # Insertion order is admission order (a re-entering request goes to
        # the back), so the dict is also the drain's tie-break order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Prefix validity is policy; phase and batches are the manager's.
        self._broken_prefixes: set[str] = set()
        # The last drain's bound, which this drain's allocations are checked
        # against.
        self._next_step_blocks = 0
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
            block_hashes: Hash of every hashed block the operation
                covers, read at admission: the manager has already replaced
                hash-less blocks, which vLLM never evicts, by the null block.
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
        self._pending.setdefault(meta.request_id, []).append(
            PendingStoreOp(
                store_metadata=meta,
                block_hashes=block_hashes,
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

    def buffered_block_ids(self) -> set[int]:
        """Every GPU block a buffered operation covers.

        Returns:
            The block ids, possibly empty.
        """
        return set().union(
            *(op.block_hashes for ops in self._pending.values() for op in ops)
        )

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
        """Drop the request's buffered ops and break its prefix chain; see
        ``OffloadPolicy.mark_store_failed``.

        Args:
            request_id: The request whose stores are given up.

        Returns:
            The number of buffered operations dropped.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.add(request_id)
        self._counters.dropped_failed_store += len(dropped)
        return len(dropped)

    def requeue_failed(self, item: PendingStoreItem) -> None:
        """Buffer a failed batch ahead of its request's later ops.

        Unpinning frees the ops' blocks to the free-queue tail, so they come
        due again when the head reaches them, rather than being resubmitted
        at once into an L1 that just rejected them.

        Args:
            item: The failed batch's operations, in token order. The
                manager filters out orphaned batches first.
        """
        request_id = item.request_id
        ops = [
            PendingStoreOp(store_metadata=meta, block_hashes=block_hashes)
            for meta, block_hashes in item.metadatas
        ]
        self._pending[request_id] = ops + self._pending.get(request_id, [])
        self._counters.requeued += len(ops)

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Record the step's block pressure and release what it made due.

        Per request: the suffix from the first op whose blocks were recycled
        is dropped, then the front up to the last op with a block in the
        danger window is released (prefix closure). Requests in
        ``blocked_request_ids`` stay pending, though their evicted suffix
        is still dropped: the worker holds one in-flight batch per request.
        Requests are served most imminent first until
        ``MAX_DRAIN_PER_STEP``; due ops past it wait for a later drain.

        Args:
            signals: The manager's bounds on the blocks the running
                requests and one admission allocate before the next
                drain, which set the danger depth, plus the
                finished and blocked request ids. Finished ids are not read
                by this policy: imminence alone decides, whether or not the
                request is still running.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """
        self._now = time.monotonic()
        self._counters.bound_blocks += self._next_step_blocks
        self._counters.bound_allocated_blocks += signals.running_blocks_allocated
        if signals.running_blocks_allocated > self._next_step_blocks:
            self._counters.bound_exceeded += 1
            logger.info(
                "Lazy offload: running requests allocated %d blocks, over the "
                "previous drain's bound of %d",
                signals.running_blocks_allocated,
                self._next_step_blocks,
            )
        self._next_step_blocks = signals.next_step_blocks
        if not self._pending:
            self._maybe_log_stats()
            return LazyOffloadDrain()

        depth = signals.next_step_blocks + signals.admission_blocks
        # Depth zero makes nothing due and the queue is not walked; the loss
        # check below reads hashes, not ranks, and still runs.
        ranks = self._free_queue_ranks(depth, self.buffered_block_ids())
        drain = LazyOffloadDrain()
        dropped_ops = 0
        dropped_ids: list[str] = []
        # Due now, as (imminence rank, admission order, request id).
        candidates: list[tuple[int, int, str]] = []
        for order, request_id in enumerate(list(self._pending)):
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
            if request_id in signals.blocked_request_ids:
                continue
            if (window_rank := _min_rank(surviving, ranks)) is not None:
                # TODO: free-queue pressure is the only retirement mechanism
                # modelled here. A sliding-window layer retires a block on a
                # token schedule instead, which is predictable but invisible
                # in the free queue, so such an op is not released when its
                # data is about to go.
                # Size the release per attention type: one due-predictor per
                # mechanism, emit up to the last index any of them marks due.
                candidates.append((window_rank, order, request_id))
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
        ops_left = MAX_DRAIN_PER_STEP
        for _, _, request_id in candidates:
            if ops_left <= 0:
                break
            ops = self._pending[request_id]
            emitted = ops[: min(_due_front_length(ops, ranks), ops_left)]
            ops_left -= len(emitted)
            self._counters.emitted += len(emitted)
            drain.items.append(_pending_item(request_id, emitted))
            remaining = ops[len(emitted) :]
            if remaining:
                self._pending[request_id] = remaining
            else:
                del self._pending[request_id]
                drain.emptied_request_ids.append(request_id)
        self._maybe_log_stats()
        return drain

    def _free_queue_ranks(
        self, depth: int, pending_blocks: "Container[int]"
    ) -> dict[int, int]:
        """The free-queue head an allocation can recycle, id -> eviction rank.

        A buffered block met in the walk is due, so it leaves the queue and
        every block behind it moves up:
        the walk runs until ``depth`` blocks outside buffered ops are
        counted. A buffered block that stays (its request is blocked, or the
        drain budget ran out) only makes the walk reach further.

        Args:
            depth: How many blocks the allocation can take from the queue
                head. Zero or less returns an empty mapping without
                touching the queue.
            pending_blocks: Every block a buffered op covers.

        Returns:
            Block id to its position from the head, lowest evicted first.
            It ends early at the queue tail.

        Raises:
            RuntimeError: If the fake head has no successor.
        """
        ranks: dict[int, int] = {}
        if depth <= 0:
            return ranks
        counted = 0
        for block in self._free_blocks():
            if counted >= depth:
                break
            ranks[block.block_id] = len(ranks)
            if block.block_id not in pending_blocks:
                counted += 1
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

    def _free_blocks(self) -> "Iterator[KVCacheBlock]":
        """Walk the free queue from its eviction head, one block at a time.

        Follows the queue's links rather than ``get_all_free_blocks()``,
        which materialises the whole queue every step.

        Yields:
            Each free block, the next to be recycled first.

        Raises:
            RuntimeError: If the fake head has no successor.
        """
        block = self._pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        # The fake tail has no successor, which is what ends the walk.
        while block.next_free_block is not None:
            yield block
            block = block.next_free_block

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        Args:
            op: The buffered operation to validate.

        Returns:
            False once any covered block was recycled, which means the
            operation's data is gone.
        """
        # TODO: an operation's blocks span every KV cache group, so on a
        # hybrid model the shortest-lived group decides for all layers: one
        # recycled sliding-window block drops the operation and the request's
        # whole tail, including full-attention layers whose data is still
        # live. Splitting the check needs per-group token ranges in
        # LoadStoreOp.
        # A plain loop over a local, not all() over a generator: every drain
        # checks every buffered block.
        blocks = self._pool.blocks
        for block_id, snapshot in op.block_hashes.items():
            if blocks[block_id].block_hash != snapshot:
                return False
        return True

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
        ``admitted + requeued == pending + emitted + every dropped_* counter``.

        Returns:
            The counter names and values, space separated.
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        pending = sum(len(ops) for ops in self._pending.values())
        return f"{fields} pending={pending}"
