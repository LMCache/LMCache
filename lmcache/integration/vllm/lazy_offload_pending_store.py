# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, cast
import enum
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import PendingStoreItem
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    DEFAULT_HORIZON_STEPS,
    AdmitResult,
    DrainResult,
    EvictionAwareStoreQueue,
    GPUBlockPoolView,
    LazyOffloadCounters,
    LazyOffloadPolicyConfig,
    PendingStoreOp,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)

ConfigValue = str | int | float | bool | list[str] | None

#: Minimum seconds between periodic counter-ledger log lines.
_STATS_LOG_INTERVAL_S = 5.0

#: Most dropped ops named in the aggregate drop line; the rest are counted.
_DROP_LOG_SAMPLE_OPS = 8


def _format_ledger(counters: LazyOffloadCounters, num_pending: int) -> str:
    """Render the ledger as one greppable ``key=value`` line body.

    Args:
        counters: The cumulative policy counters.
        num_pending: Operations still buffered at the same instant. It makes
            the line close as an equation -- ``admitted == pending +
            emitted + every drop counter`` -- so a reader can tell an
            operation still waiting for pressure from one that left the
            queue without incrementing any outcome counter. Without it the
            strongest available check is ``outcomes <= admitted``, which
            catches over-counting only.

    Returns:
        The rendered ``key=value`` body, pending depth last.
    """
    fields = " ".join(f"{name}={value}" for name, value in asdict(counters).items())
    return f"{fields} pending={num_pending}"


def _format_drop_sample(dropped: list[PendingStoreOp]) -> str:
    """Render dropped ops for the aggregate drop line, truncating the tail.

    Args:
        dropped: The operations dropped by one drain, in drop order.

    Returns:
        ``request (prefix N)`` for at most ``_DROP_LOG_SAMPLE_OPS`` ops,
        with a ``+N more`` suffix when the list was truncated.
    """
    sample = ", ".join(
        f"{op.request_id} (prefix {op.prefix_end_tokens})"
        for op in dropped[:_DROP_LOG_SAMPLE_OPS]
    )
    omitted = len(dropped) - _DROP_LOG_SAMPLE_OPS
    if omitted > 0:
        sample += f", +{omitted} more"
    return sample


class LazyOffloadMode(enum.Enum):
    """Which drain policy drives the pending store.

    - FIFO: count-triggered whole-request drain (legacy placeholder).
    - EVICTION_AWARE: pressure-triggered drain in free-queue LRU order
      (see ``lazy_offload_policy/eviction_aware.py`` and the decision-model
      design doc).
    """

    FIFO = "FIFO"
    EVICTION_AWARE = "EVICTION_AWARE"


class AddOutcome(enum.Enum):
    """Result of buffering a store operation.

    - BUFFERED: the operation is pending; it will be drained later.
    - SKIPPED_UNHASHED: a covered block has no prefix-cache hash, so its
      later eviction would be undetectable; the operation is not buffered
      and must not be stored (lazy offload requires prefix caching).
    - SKIPPED_PREFIX_BROKEN: an earlier chunk of the request was already
      dropped; storing this one would be unreachable on retrieval.
    - DEDUPLICATED: identical content is already buffered under another
      request; it will be stored (or dropped) with that operation, so
      nothing is buffered for this request.
    """

    BUFFERED = enum.auto()
    SKIPPED_UNHASHED = enum.auto()
    SKIPPED_PREFIX_BROKEN = enum.auto()
    DEDUPLICATED = enum.auto()


class LazyOffloadPendingStore:
    """
    Buffering store operations in lazy offload mode.

    Store metadata is accumulated here instead of being immediately submitted.
    When the offload policy decides it's time, a batch of items is drained
    and returned for submission.
    """

    def __init__(
        self,
        configs: dict[str, ConfigValue] | None = None,
    ) -> None:
        """
        Initialize the pending store queue.

        Args:
            configs: The kv_connector_extra_config dict. Recognized keys:
                ``lmcache.mp.lazy_offload_policy`` ("FIFO" default, or
                "EVICTION_AWARE"), ``lmcache.mp.lazy_offload_horizon_steps`` (float),
                ``lmcache.mp.lazy_offload_min_prefix_tokens`` (int),
                ``lmcache.mp.lazy_offload_max_drain_per_step`` (int), and the
                FIFO-only ``lmcache.mp.lazy_offload_threshold`` /
                ``lmcache.mp.lazy_offload_select_count``.

        Raises:
            ValueError: If the configured policy name is unknown.
        """
        configs = configs or {}
        policy = cast(str, configs.get("lmcache.mp.lazy_offload_policy", "FIFO"))
        try:
            self._mode = LazyOffloadMode(policy)
        except ValueError as e:
            raise ValueError(f"Unknown offload policy: {policy}") from e

        self._fifo_policy: FIFOOffloadPolicy | None = None
        # Built when the GPU block pool is bound (it needs the pool view).
        self._eviction_queue: EvictionAwareStoreQueue | None = None
        # Periodic counter-ledger logging state: the last snapshot written
        # to the log and when it was written (see _maybe_log_stats).
        self._last_logged_stats = LazyOffloadCounters()
        self._last_stats_log_time = 0.0
        # The throttle-versus-loss warning is emitted once per process (see
        # collect_due): it names a misconfiguration that persists for the
        # whole run, so repeating it every step would add noise without
        # adding information. The counters carry the recurrence.
        self._warned_throttled_loss = False
        self._eviction_config = LazyOffloadPolicyConfig(
            horizon_steps=float(
                cast(
                    str | int | float,
                    configs.get(
                        "lmcache.mp.lazy_offload_horizon_steps",
                        DEFAULT_HORIZON_STEPS,
                    ),
                )
            ),
            min_prefix_tokens=int(
                cast(
                    str | int | float,
                    configs.get("lmcache.mp.lazy_offload_min_prefix_tokens", 0),
                )
            ),
            max_drain_per_step=int(
                cast(
                    str | int | float,
                    configs.get("lmcache.mp.lazy_offload_max_drain_per_step", 64),
                )
            ),
        )
        if self._mode is LazyOffloadMode.FIFO:
            self._fifo_policy = FIFOOffloadPolicy(configs)
        else:
            logger.info(
                "lazy offload enabled with EVICTION_AWARE policy: %s",
                self._eviction_config,
            )

        self._select_count = int(
            cast(
                str | int | float,
                configs.get("lmcache.mp.lazy_offload_select_count", 10),
            )
        )

        # GPU block pool reference
        self._gpu_block_pool: "BlockPool | None" = None

        # save all request block ids for free
        self._request_block_ids: dict[str, list[int]] = defaultdict(list)

    @property
    def mode(self) -> LazyOffloadMode:
        """The configured drain mode."""
        return self._mode

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the GPU block pool to the pending store.

        Idempotent for the same pool. Rebinding a different pool would
        silently invalidate every buffered operation's hash snapshot and,
        in EVICTION_AWARE mode, discard the whole pending queue.

        Args:
            gpu_block_pool: The scheduler's GPU block pool.

        Raises:
            ValueError: If a different pool is already bound.
        """
        if self._gpu_block_pool is gpu_block_pool:
            return
        if self._gpu_block_pool is not None:
            raise ValueError(
                "a different GPU block pool is already bound; rebinding "
                "would discard the buffered store operations"
            )
        self._gpu_block_pool = gpu_block_pool
        if self._mode is LazyOffloadMode.EVICTION_AWARE:
            self._eviction_queue = EvictionAwareStoreQueue(
                self._eviction_config, GPUBlockPoolView(gpu_block_pool)
            )

    def add(self, meta: "LMCacheMPRequestMetadata") -> AddOutcome:
        """Buffer a store operation produced by ``GetStoreMetadata``.

        Args:
            meta: The store metadata to buffer.

        Returns:
            The buffering outcome; see :class:`AddOutcome` for the action
            the caller must take on each value.

        Raises:
            ValueError: If the GPU block pool has not been bound.
        """
        if not self._gpu_block_pool:
            raise ValueError("gpu block pool not bound")
        block_hashes = {
            bid: self._gpu_block_pool.blocks[bid].block_hash
            for bid in meta.op.flat_block_ids
        }
        if self._fifo_policy is not None:
            self._fifo_policy.add(meta, block_hashes)
            return AddOutcome.BUFFERED
        queue = self._require_eviction_queue()
        op = PendingStoreOp(
            request_id=meta.request_id,
            store_metadata=meta,
            # admit() rejects any None value in the snapshot.
            block_hashes=cast("dict[int, BlockHashWithGroupId]", block_hashes),
            prefix_start_tokens=meta.op.start,
            prefix_end_tokens=meta.op.end,
            cache_salt=meta.cache_salt,
        )
        admit = queue.admit(op)
        if admit is AdmitResult.ADMITTED:
            return AddOutcome.BUFFERED
        if admit is AdmitResult.DEDUPLICATED:
            return AddOutcome.DEDUPLICATED
        if admit is AdmitResult.REJECTED_UNHASHED_BLOCK:
            logger.warning(
                "Lazy offload: skipping store for request %s tokens [%d, %d): "
                "covered blocks carry no prefix-cache hash. Either prefix "
                "caching is off, or the model uses sliding-window or hybrid "
                "attention and these positions have fallen outside the "
                "attention window, where vLLM leaves a hash-less null block "
                "that holds no KV. Later chunks of this request are skipped "
                "too; rejected_unhashed and rejected_prefix_broken count the "
                "recurrence.",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return AddOutcome.SKIPPED_UNHASHED
        # DEBUG, not INFO: this is the tail of an already reported event.
        # Every site that breaks a request's chain logs the cause at INFO or
        # WARNING (eviction drop, gate-3 drop, unhashed blocks, failed
        # store), and one broken request rejects every later chunk it
        # produces -- at INFO that would bury its own cause.
        logger.debug(
            "Lazy offload: skipping store for request %s tokens [%d, %d): "
            "the request's prefix chain is already broken",
            meta.request_id,
            meta.op.start,
            meta.op.end,
        )
        return AddOutcome.SKIPPED_PREFIX_BROKEN

    def observe_step(
        self,
        new_blocks_allocated: int,
        est_next_step_blocks: int,
        allocated_block_ids: set[int] | None = None,
    ) -> None:
        """Forward one step's block-consumption signals to the policy.

        No-op in FIFO mode.

        Args:
            new_blocks_allocated: GPU blocks newly allocated this step.
            est_next_step_blocks: Estimated allocation of the next step.
            allocated_block_ids: Ids allocated or resurrected this step, used
                for incremental snapshot validation. None requests a full
                validation pass.
        """
        if self._eviction_queue is not None:
            self._eviction_queue.observe_step(
                new_blocks_allocated,
                est_next_step_blocks,
                allocated_block_ids,
            )

    def collect_due(self) -> DrainResult:
        """Release the operations facing imminent eviction (EVICTION_AWARE).

        Returns:
            The policy's drain decision for this step.

        Raises:
            ValueError: If called in FIFO mode or before the pool is bound.
        """
        queue = self._require_eviction_queue()
        result = queue.collect_due()
        if (
            result.ops_held_back
            and result.dropped_evicted
            and not self._warned_throttled_loss
        ):
            # The two symptoms together are what distinguishes a cap that
            # merely delays a burst from one set below the workload's
            # steady-state admission rate: the drain is capped *and* the
            # queue it could not work off is dying. WARNING, and once:
            # neither counter alone justifies telling an operator their
            # configuration is wrong.
            self._warned_throttled_loss = True
            logger.warning(
                "Lazy offload: max_drain_per_step=%d held back %d due store "
                "op(s) while %d op(s) were lost to eviction in the same "
                "step. A cap below the number of concurrently prefilling "
                "requests loses the backlog instead of delaying it; raise "
                "lmcache.mp.lazy_offload_max_drain_per_step. "
                "throttled_drains counts the recurrence.",
                self._eviction_config.max_drain_per_step,
                result.ops_held_back,
                len(result.dropped_evicted),
            )
        if result.dropped_evicted:
            # INFO, not DEBUG: each drop is one unit of cache-quality loss
            # (the gate-1 drop-rate sensor), and production logs rarely run
            # at DEBUG. One aggregate line per drain: a burst that evicts a
            # large pending queue at once must not emit thousands of
            # synchronous lines on the scheduler hot path. Per-op detail
            # stays at DEBUG.
            logger.info(
                "Lazy offload: dropped %d store op(s): blocks evicted "
                "before drain (%s)",
                len(result.dropped_evicted),
                _format_drop_sample(result.dropped_evicted),
            )
            for dropped_op in result.dropped_evicted:
                logger.debug(
                    "Lazy offload: dropped store for request %s (prefix %d): "
                    "blocks evicted before drain",
                    dropped_op.request_id,
                    dropped_op.prefix_end_tokens,
                )
        if result.dropped_short_prefix:
            # Same shape and level as the eviction path above, for the same
            # reason: a gate-3 drop is cache-quality loss the operator has
            # to be able to attribute to a request, and the counter alone
            # cannot say which one lost its prefix.
            logger.info(
                "Lazy offload: dropped %d store op(s): request prefix below "
                "the break-even length (%s)",
                len(result.dropped_short_prefix),
                _format_drop_sample(result.dropped_short_prefix),
            )
            for dropped_op in result.dropped_short_prefix:
                logger.debug(
                    "Lazy offload: dropped store for request %s (prefix %d): "
                    "request prefix below the break-even length",
                    dropped_op.request_id,
                    dropped_op.prefix_end_tokens,
                )
        self._maybe_log_stats(queue)
        return result

    def notify_store_complete(self, req_id: str) -> bool:
        """Record a completed store batch for a request.

        Args:
            req_id: The request whose store completion was reported.

        Returns:
            True if the request's session may now be torn down.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.notify_stored(req_id)
        # FIFO drains a request's buffered ops all at once, so the receipt
        # always ends the session.
        return True

    def stats(self) -> LazyOffloadCounters:
        """Return a copy of the cumulative policy counters.

        Returns:
            A snapshot of the eviction-aware policy's counters (admissions,
            emissions, drops by cause, deduplications).

        Raises:
            ValueError: If called in FIFO mode or before the pool is bound.
        """
        return self._require_eviction_queue().stats()

    def log_final_stats(self) -> None:
        """Log the cumulative policy counters as one INFO ``key=value`` line.

        Called at connector shutdown so the drop ledger (notably
        ``dropped_evicted``, the gate-1 quality sensor) closes with an
        exact final value. Best-effort: a force-killed engine process may
        die before reaching it, which is why :meth:`collect_due` also logs
        the ledger periodically. No-op in FIFO mode or when the pool was
        never bound: nothing was counted.
        """
        if self._eviction_queue is None:
            return
        logger.info(
            "Lazy offload final counters: %s",
            _format_ledger(
                self._eviction_queue.stats(),
                self._eviction_queue.num_pending_ops(),
            ),
        )

    def _maybe_log_stats(self, queue: EvictionAwareStoreQueue) -> None:
        """Log the counter ledger if it changed and the throttle allows.

        Runs on every drain (the engine calls ``collect_due`` each step),
        so the log converges to the true ledger whenever the engine takes
        a step at least ``_STATS_LOG_INTERVAL_S`` after the last change --
        the shutdown hook alone is unreliable under a force-killed engine.

        The change test looks at the counters only, not at the pending depth
        the line also carries: every mutation of the pending queue moves a
        counter with it (admission, emission, each drop cause), so a changed
        depth always shows up as changed counters.
        """
        stats = queue.stats()
        if stats == self._last_logged_stats:
            return
        now = time.monotonic()
        if now - self._last_stats_log_time < _STATS_LOG_INTERVAL_S:
            return
        logger.info(
            "Lazy offload counters: %s",
            _format_ledger(stats, queue.num_pending_ops()),
        )
        self._last_logged_stats = stats
        self._last_stats_log_time = now

    def pop_items_for_offload(self) -> list[PendingStoreItem]:
        """Pop items when the policy's trigger is satisfied (FIFO mode only).

        Returns:
            The pending store items to be submitted, or an empty list when
            no offload is due.
        """
        return self._require_fifo_policy().pop_items_for_offload(self._select_count)

    def mark_req_finished(self, req_id: str) -> bool:
        """Record that the engine finished a request.

        Args:
            req_id: The finished request.

        Returns:
            True if stores are still pending or in flight for the request
            (session teardown must wait); False otherwise.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.mark_request_finished(req_id)
        return self._require_fifo_policy().mark_req_finished(req_id)

    def drop_request(self, req_id: str) -> int:
        """Discard the request's buffered (not yet drained) operations.

        Called when the buffered state becomes stale -- e.g. the connector
        resets a preempted request's tracker, which after resume re-produces
        store metadata from token zero, overlapping anything still buffered.
        A batch already drained and submitted is unaffected: its blocks stay
        pinned until the completion receipt arrives.

        Args:
            req_id: The request whose buffered operations are discarded.

        Returns:
            The number of buffered operations discarded.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.drop_request(req_id)
        return self._require_fifo_policy().drop_request(req_id)

    def reclaim_finished_request(self, req_id: str) -> bool:
        """Release a finished predecessor's residual state on id reuse.

        In lazy mode the engine frees a finished request's id immediately,
        so a new request may arrive under an id whose previous owner still
        has buffered state (teardown deferred). Call this when a new
        request's id is first seen; see
        :meth:`EvictionAwareStoreQueue.reclaim_finished_request` for the
        conflation hazards this prevents.

        Args:
            req_id: The reused request id.

        Returns:
            True if the caller must end the predecessor's session now;
            False if there was nothing to reclaim or (EVICTION_AWARE) the
            teardown rides an outstanding completion receipt.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.reclaim_finished_request(req_id)
        return self._require_fifo_policy().reclaim_finished_request(req_id)

    def mark_store_failed(self, req_id: str) -> int:
        """Record that the request's in-flight store batch failed.

        The request's stored prefix chain is broken: its buffered
        operations are dropped and, in EVICTION_AWARE mode, further chunks
        are rejected until the request is torn down (storing them without
        the failed prefix would be unreachable).

        Args:
            req_id: The request whose store failed.

        Returns:
            The number of buffered operations dropped.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.mark_store_failed(req_id)
        # FIFO drains a request's chunks all at once after it finishes, so
        # nothing of it remains buffered by the time its store fails.
        return 0

    def _require_eviction_queue(self) -> EvictionAwareStoreQueue:
        if self._eviction_queue is None:
            raise ValueError(
                "EVICTION_AWARE queue unavailable: wrong mode or GPU block "
                "pool not bound"
            )
        return self._eviction_queue

    def _require_fifo_policy(self) -> FIFOOffloadPolicy:
        if self._fifo_policy is None:
            raise ValueError("FIFO policy unavailable in EVICTION_AWARE mode")
        return self._fifo_policy

    def has_in_flight_store(self, req_id: str) -> bool:
        """Whether a drained store batch of the request awaits its receipt.

        True from the drain that pinned and submitted the request's batch
        until the full set of worker completion receipts has been processed.
        A receipt for a request outside this window is a duplicate or stale
        resend and must be ignored (processing it would unpin blocks that
        are no longer pinned and tear the session down twice).

        Args:
            req_id: The request to check.

        Returns:
            True if a submitted store batch is awaiting completion.
        """
        return req_id in self._request_block_ids

    def update_request_gpu_block_ids(self, req_id: str, block_ids: list[int]) -> None:
        """Record blocks pinned for the request's submitted store batch.

        Opens the request's receipt window (``has_in_flight_store``).

        Args:
            req_id: The request whose batch was submitted.
            block_ids: The GPU blocks pinned for the batch.
        """
        self._request_block_ids[req_id].extend(block_ids)

    def get_request_gpu_block_ids(self, req_id: str) -> list[int]:
        """Return the blocks pinned for the request's in-flight batch.

        A read never creates state: an unknown request yields an empty
        list and leaves ``has_in_flight_store`` False.

        Args:
            req_id: The request to look up.
        """
        return self._request_block_ids.get(req_id, [])

    def remove_request_gpu_block_ids(self, req_id: str) -> None:
        """Close the request's receipt window after its receipt completes.

        Args:
            req_id: The request whose receipt was processed.
        """
        if req_id in self._request_block_ids:
            del self._request_block_ids[req_id]
