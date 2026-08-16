# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side orchestration for lazy cache offload.

This module is the integration boundary between ``LMCacheMPConnector`` and the
lazy-offload policies.  It owns policy dispatch, GPU block pinning, store-batch
coalescing, completion handling, and deferred session-release decisions.  The
connector only forwards lifecycle events and applies the returned actions.
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

# First Party
from lmcache.integration.vllm.lazy_offload_pending_store import (
    AddOutcome,
    ConfigValue,
    LazyOffloadMode,
    LazyOffloadPendingStore,
)
from lmcache.integration.vllm.lazy_offload_state import LazyOffloadRequestRegistry
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LoadStoreOp,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class StoreCompletionTracker(Protocol):
    """Aggregate per-worker completion counts for one submitted store."""

    def update_pending_store_count(self, request_id: str, count: int) -> bool:
        """Record receipts and report whether all expected workers completed.

        Args:
            request_id: Request whose store workers reported completion.
            count: Number of newly completed workers.

        Returns:
            True when the submitted batch has received every expected worker
            completion; False while more receipts remain outstanding.
        """
        ...


@dataclass
class LazyOffloadActions:
    """Explicit connector effects produced by one lazy-offload event.

    Attributes:
        stores_to_submit: Coalesced store metadata to append to the current
            connector metadata. Blocks referenced by these stores have
            already been pinned by :class:`LazyOffloadManager`.
        sessions_to_end: Request sessions whose pending and in-flight lazy
            stores have settled and can now be released by the connector.
    """

    stores_to_submit: list[LMCacheMPRequestMetadata] = field(default_factory=list)
    sessions_to_end: list[str] = field(default_factory=list)


def _count_new_blocks(scheduler_output: "SchedulerOutput") -> int:
    """Count GPU blocks allocated in one scheduler step across all groups.

    Args:
        scheduler_output: The vLLM scheduler output for the step.

    Returns:
        The gross number of block ids handed to new and cached requests.
    """
    count = 0
    for new_request in scheduler_output.scheduled_new_reqs:
        count += sum(len(group_ids) for group_ids in new_request.block_ids)
    for request_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
        if request_block_ids:
            count += sum(len(group_ids) for group_ids in request_block_ids)
    return count


def _allocated_block_ids(scheduler_output: "SchedulerOutput") -> set[int]:
    """Collect block ids allocated or resurrected in one scheduler step.

    Args:
        scheduler_output: The vLLM scheduler output for the step.

    Returns:
        Unique block ids handed to new and cached requests. The policy uses
        them to incrementally revalidate affected pending operations.
    """
    block_ids: set[int] = set()
    for new_request in scheduler_output.scheduled_new_reqs:
        for group_ids in new_request.block_ids:
            block_ids.update(group_ids)
    for request_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
        if request_block_ids:
            for group_ids in request_block_ids:
                block_ids.update(group_ids)
    return block_ids


def _coalesce_store_metadata(
    request_metas: list[LMCacheMPRequestMetadata],
) -> LMCacheMPRequestMetadata:
    """Merge one request's contiguous store operations into one operation.

    The worker tracks one in-flight store future per request, so a drained
    batch must be submitted as one operation.

    Args:
        request_metas: Non-empty STORE metadata in request-prefix order.

    Returns:
        One store metadata covering the complete contiguous input range.

    Raises:
        ValueError: If the input is empty, contains non-contiguous ranges, or
            changes the number of cache groups within the batch.
    """
    if not request_metas:
        raise ValueError("cannot coalesce an empty store batch")
    if len(request_metas) == 1:
        return request_metas[0]
    first = request_metas[0]
    last = request_metas[-1]
    merged_block_ids: list[list[int]] = [list(group) for group in first.op.block_ids]
    expected_start = first.op.end
    for meta in request_metas[1:]:
        if meta.op.start != expected_start:
            raise ValueError(
                f"non-contiguous store ops for request {first.request_id}: "
                f"expected start {expected_start}, got {meta.op.start}"
            )
        if len(meta.op.block_ids) != len(merged_block_ids):
            raise ValueError(
                f"cache-group count changed within store batch for request "
                f"{first.request_id}"
            )
        expected_start = meta.op.end
        for group_idx, group_ids in enumerate(meta.op.block_ids):
            merged_block_ids[group_idx].extend(group_ids)
    merged_op = LoadStoreOp(
        token_ids=last.op.token_ids,
        block_ids=merged_block_ids,
        start=first.op.start,
        end=last.op.end,
    )
    return LMCacheMPRequestMetadata(
        request_id=first.request_id,
        direction="STORE",
        op=merged_op,
        cache_salt=first.cache_salt,
    )


class LazyOffloadManager:
    """Own scheduler-side lazy-offload integration and side effects.

    The manager is intentionally the only lazy-offload object exposed to the
    connector. Policy implementations remain pure decision logic; this class
    translates scheduler events into policy signals, pins and unpins vLLM GPU
    blocks, and returns explicit connector actions.

    Not thread-safe. All methods must run on the vLLM scheduler thread.
    """

    def __init__(
        self,
        configs: dict[str, ConfigValue] | None,
        group_tokens_per_block: list[int],
        completion_tracker: StoreCompletionTracker,
    ) -> None:
        """Create an unbound scheduler-side manager.

        Args:
            configs: vLLM connector extra configuration.
            group_tokens_per_block: Token capacity for each KV-cache group,
                used to estimate the next scheduler step's block pressure.
            completion_tracker: Scheduler adapter view that aggregates
                per-worker completion receipt counts.
        """
        self._pending_store = LazyOffloadPendingStore(configs)
        self._group_tokens_per_block = list(group_tokens_per_block)
        self._completion_tracker = completion_tracker
        self._gpu_block_pool: "BlockPool | None" = None
        self._requests = LazyOffloadRequestRegistry()

    def bind_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the scheduler's GPU block pool.

        Args:
            gpu_block_pool: The vLLM block pool used for validation and
                pin/unpin operations.
        """
        self._pending_store.bind_gpu_block_pool(gpu_block_pool)
        self._gpu_block_pool = gpu_block_pool

    def add_store_candidate(self, metadata: LMCacheMPRequestMetadata) -> AddOutcome:
        """Buffer one store candidate produced by the request tracker.

        Args:
            metadata: STORE metadata to defer.

        Returns:
            The pending store's admission outcome.
        """
        self._requests.ensure_active(metadata.request_id)
        return self._pending_store.add(metadata)

    def on_scheduler_step(
        self, scheduler_output: "SchedulerOutput"
    ) -> LazyOffloadActions:
        """Drain stores made due by one token-producing scheduler step.

        Zero-token steps return no actions because vLLM takes its no-forward
        path and would discard connector metadata produced by that step.

        Args:
            scheduler_output: The completed scheduler decision for the step.

        Returns:
            Stores to submit and sessions made releasable by the drain.

        Raises:
            ValueError: If the GPU block pool has not been bound.
        """
        if not scheduler_output.total_num_scheduled_tokens:
            return LazyOffloadActions()
        pool = self._require_block_pool()
        if self._pending_store.mode is LazyOffloadMode.EVICTION_AWARE:
            return self._drain_eviction_aware(scheduler_output, pool)
        return self._drain_fifo(pool)

    def on_store_results(
        self,
        failed_request_ids: set[str],
        completed_store_counts: dict[str, int],
    ) -> LazyOffloadActions:
        """Apply failed stores and fully aggregated completion receipts.

        Failures are processed before completions so dropping a finished
        request's held-back suffix can make it releasable by the accompanying
        completion receipt.

        Args:
            failed_request_ids: Requests for which at least one worker
                reported the current store batch failed.
            completed_store_counts: Newly reported worker completion counts
                keyed by request. The manager filters stale receipts before
                forwarding counts to its completion tracker.

        Returns:
            Sessions made releasable by completed batches. No stores are
            produced by receipt processing.

        Raises:
            ValueError: If the GPU block pool has not been bound.
        """
        pool = self._require_block_pool()
        for request_id in failed_request_ids:
            if not self._requests.has_in_flight(request_id):
                continue
            dropped = self._pending_store.mark_store_failed(request_id)
            logger.warning(
                "Store failed for request %s; dropped %d held-back store "
                "op(s) that would lack their stored prefix",
                request_id,
                dropped,
            )

        actions = LazyOffloadActions()
        for request_id, count in completed_store_counts.items():
            if not self._requests.has_in_flight(request_id):
                logger.warning(
                    "Ignoring store-completion receipt for request %s with "
                    "no in-flight store batch",
                    request_id,
                )
                continue
            if not self._completion_tracker.update_pending_store_count(
                request_id, count
            ):
                continue
            batch = self._requests.complete_batch(request_id)
            pool.free_blocks(
                [pool.blocks[block_id] for block_id in batch.block_ids],
                prepend=True,
            )
            if self._pending_store.notify_store_complete(
                request_id
            ) and self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._requests.session_ended(request_id)
        return actions

    def on_request_finished(self, request_id: str) -> LazyOffloadActions:
        """Record request completion and decide whether its session can end.

        Args:
            request_id: The request that finished generation.

        Returns:
            An immediate session-release action only when no store is pending
            or in flight; otherwise an empty action.
        """
        self._requests.finish(request_id)
        if self._pending_store.mark_req_finished(request_id):
            return LazyOffloadActions()
        if self._requests.has_in_flight(request_id):
            return LazyOffloadActions()
        self._requests.session_ended(request_id)
        return LazyOffloadActions(sessions_to_end=[request_id])

    def on_request_reset(self, request_id: str) -> int:
        """Drop buffered operations invalidated by a preemption reset.

        Args:
            request_id: The preempted request whose tracker restarts from
                token zero.

        Returns:
            Number of buffered operations discarded.
        """
        self._requests.reset(request_id)
        dropped = self._pending_store.drop_request(request_id)
        if dropped:
            logger.info(
                "Lazy offload: dropped %d buffered store op(s) of preempted request %s",
                dropped,
                request_id,
            )
        return dropped

    def on_request_arrived(self, request_id: str) -> LazyOffloadActions:
        """Reclaim residual state if a new request reuses a finished id.

        Args:
            request_id: Identifier of the newly arrived request.

        Returns:
            A predecessor session-release action when no in-flight batch is
            carrying that release; otherwise an empty action.
        """
        self._requests.arrive(request_id)
        if not self._pending_store.reclaim_finished_request(request_id):
            return LazyOffloadActions()
        logger.info(
            "Lazy offload: request id %s reused while its predecessor's "
            "teardown was deferred; released the predecessor's session",
            request_id,
        )
        return LazyOffloadActions(sessions_to_end=[request_id])

    def log_final_stats(self) -> None:
        """Write the final eviction-aware counter ledger, when available."""
        self._pending_store.log_final_stats()

    def _drain_eviction_aware(
        self,
        scheduler_output: "SchedulerOutput",
        pool: "BlockPool",
    ) -> LazyOffloadActions:
        """Run a pressure-triggered drain and pin every emitted operation."""
        gross_new_blocks = _count_new_blocks(scheduler_output)
        est_next_step_blocks = sum(
            -(-scheduler_output.total_num_scheduled_tokens // tokens_per_block)
            for tokens_per_block in self._group_tokens_per_block
        )
        self._pending_store.observe_step(
            gross_new_blocks,
            est_next_step_blocks,
            _allocated_block_ids(scheduler_output),
        )
        result = self._pending_store.collect_due()

        ops_by_request: dict[str, list[LMCacheMPRequestMetadata]] = {}
        blocks_by_request: dict[str, list[int]] = {}
        for pending_op in result.to_store:
            ops_by_request.setdefault(pending_op.request_id, []).append(
                pending_op.store_metadata
            )
            blocks_by_request.setdefault(pending_op.request_id, []).extend(
                pending_op.block_hashes
            )

        stores_to_submit = [
            _coalesce_store_metadata(request_metas)
            for request_metas in ops_by_request.values()
        ]
        for request_id in blocks_by_request:
            if self._requests.has_in_flight(request_id):
                raise RuntimeError(
                    f"request {request_id!r} emitted while a store batch "
                    "is still in flight"
                )
        for request_id, block_ids in blocks_by_request.items():
            pool.touch([pool.blocks[block_id] for block_id in block_ids])
            self._requests.register_batch(request_id, block_ids)

        for request_id in result.released_requests:
            self._requests.session_ended(request_id)
        return LazyOffloadActions(
            stores_to_submit=stores_to_submit,
            sessions_to_end=result.released_requests,
        )

    def _drain_fifo(self, pool: "BlockPool") -> LazyOffloadActions:
        """Run the legacy FIFO drain with ex-post hash validation."""
        actions = LazyOffloadActions()
        for item in self._pending_store.pop_items_for_offload(
            self._requests.in_flight_request_ids()
        ):
            valid_metas: list[LMCacheMPRequestMetadata] = []
            valid_block_ids: list[int] = []
            for metadata, old_block_hashes in item.metadatas:
                gpu_block_ids = list(old_block_hashes)
                blocks = [pool.blocks[block_id] for block_id in gpu_block_ids]
                pool.touch(blocks)
                new_block_hashes = {
                    block_id: pool.blocks[block_id].block_hash
                    for block_id in gpu_block_ids
                }
                if (
                    any(block_hash is None for block_hash in new_block_hashes.values())
                    or old_block_hashes != new_block_hashes
                ):
                    logger.warning(
                        "Block hashes missing or mismatched for request %s, "
                        "dropping its remaining chunks",
                        item.request_id,
                    )
                    pool.free_blocks(blocks)
                    break
                valid_metas.append(metadata)
                valid_block_ids.extend(gpu_block_ids)
            if not valid_metas:
                actions.sessions_to_end.append(item.request_id)
                self._requests.session_ended(item.request_id)
                continue
            actions.stores_to_submit.append(_coalesce_store_metadata(valid_metas))
            self._requests.register_batch(item.request_id, valid_block_ids)
        return actions

    def _require_block_pool(self) -> "BlockPool":
        """Return the bound block pool or reject an invalid lifecycle call."""
        if self._gpu_block_pool is None:
            raise ValueError("lazy offload GPU block pool is not bound")
        return self._gpu_block_pool
