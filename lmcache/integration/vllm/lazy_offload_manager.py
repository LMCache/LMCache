# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side orchestration for lazy cache offload.

The boundary between ``LMCacheMPConnector`` and the lazy-offload policies: it
owns policy dispatch, GPU block pinning, store-batch coalescing, completion
handling and deferred session release. The connector only forwards events.

Terms used here and in ``lazy_offload_state``, on top of the policy-facing
ones defined in ``lazy_offload_policy.base``:

- **Request generation**: one use of a request id. vLLM recreates a tracker
  under the same id when it resumes a preempted request, and a later,
  unrelated request may reuse the id of a finished one, so one id can
  outlive several generations.
- **Receipt**: a worker's report that a submitted store batch ended, in
  success or in failure. The batch settles once every worker has reported.
- **Pin**: the reference this class takes on a GPU block so that vLLM cannot
  recycle it while the worker is still reading it. The receipt releases it.
- **Orphaned**: said of a batch whose generation ended before its receipt
  arrived. Its pins are still released, but its failure is not charged to
  the generation now holding the id.
- **Token ledger**: the one token-id list a request's buffered operations
  share. The tracker hands out a fresh copy of the whole sequence with every
  operation, which deferral would otherwise retain once per operation.
"""

# Standard
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

# First Party
from lmcache.integration.vllm.lazy_offload_policy import create_offload_policy
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    OffloadPolicy,
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

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Record ``count`` new worker receipts for ``request_id``.

        The parameters are positional-only: the implementer names the first
        one ``req_id``, so a keyword call would not bind.

        Args:
            request_id: The request whose in-flight store batch the receipts
                belong to.
            count: Worker completions newly reported for that batch, added
                to the ones already counted.

        Returns:
            True once the submitted batch has every expected completion.
        """
        ...


@dataclass
class LazyOffloadActions:
    """Explicit connector effects produced by one lazy-offload event.

    Attributes:
        stores_to_submit: What to store now, one coalesced operation per
            request, whose GPU blocks this class has already pinned. The
            caller submits each and the receipt unpins them.
        sessions_to_end: Requests whose LMCache session the caller may
            release: either they have settled (finished, nothing buffered,
            no batch in flight), or a new request is taking over the id and
            the previous holder's deferred teardown has to happen first.
            The caller must apply these before opening a session of its
            own for the same id.
    """

    stores_to_submit: list[LMCacheMPRequestMetadata] = field(default_factory=list)
    sessions_to_end: list[str] = field(default_factory=list)


def _new_blocks(scheduler_output: "SchedulerOutput") -> int:
    """Count the GPU blocks one scheduler step handed out.

    Args:
        scheduler_output: The step's schedule, whose newly scheduled and
            cached requests carry the block ids allocated to them.

    Returns:
        The gross count over all requests and cache groups, which
        drives the policy's estimate of free-queue consumption.
    """
    groups = [request.block_ids for request in scheduler_output.scheduled_new_reqs]
    groups.extend(
        request_block_ids
        for request_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids
        if request_block_ids
    )
    return sum(
        len(group_ids) for request_groups in groups for group_ids in request_groups
    )


def _coalesce_store_metadata(
    request_metas: list[LMCacheMPRequestMetadata],
) -> LMCacheMPRequestMetadata:
    """Merge one request's contiguous STORE metadata, in prefix order, into one.

    The worker tracks one in-flight store future per request, so a drained
    batch must be submitted as a single operation.

    Args:
        request_metas: One request's buffered STORE metadata, in prefix
            order, each covering the token range that follows the one
            before it.

    Returns:
        A single STORE metadata spanning the whole range, carrying the
        concatenated block ids of every cache group.

    Raises:
        ValueError: If the input is empty, non-contiguous, or changes cache
            group count mid-batch.
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

    The only lazy-offload object exposed to the connector: it turns scheduler
    events into policy signals, pins and unpins vLLM GPU blocks, and returns
    connector actions. Every method other than :meth:`bind_block_pool` and
    :meth:`log_final_stats` raises ``ValueError`` until the pool is bound.
    Scheduler thread only.
    """

    def __init__(
        self,
        configs: dict[str, ConfigValue] | None,
        group_tokens_per_block: list[int],
        completion_tracker: StoreCompletionTracker,
    ) -> None:
        """Create an unbound scheduler-side manager.

        Args:
            configs: Connector extra config, read by the policy at bind time.
            group_tokens_per_block: Token capacity per KV-cache group, used
                to estimate the next step's block pressure.
            completion_tracker: Adapter view aggregating per-worker receipts.
        """
        self._configs = dict(configs or {})
        self._group_tokens_per_block = list(group_tokens_per_block)
        self._completion_tracker = completion_tracker
        # Both are set by bind_block_pool: the policy reads the pool.
        self._gpu_block_pool: "BlockPool | None" = None
        self._policy: OffloadPolicy | None = None
        self._requests = LazyOffloadRequestRegistry()
        # One token ledger per request whose operations are buffered.
        self._token_ledgers: dict[str, list[int]] = {}

    def bind_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the scheduler's GPU block pool and build the policy.

        Idempotent for the same pool; rebinding a different one would
        silently invalidate every buffered hash snapshot.

        Args:
            gpu_block_pool: The scheduler's block pool, from which the
                manager reads block hashes and which it pins into.

        Raises:
            ValueError: If a different pool is bound, or the policy name or
                tunables are invalid.
        """
        if self._gpu_block_pool is gpu_block_pool:
            return
        if self._gpu_block_pool is not None:
            raise ValueError(
                "a different GPU block pool is already bound; rebinding "
                "would discard the buffered store operations"
            )
        self._gpu_block_pool = gpu_block_pool
        self._policy = create_offload_policy(self._configs, gpu_block_pool)

    def add_store_candidate(self, metadata: LMCacheMPRequestMetadata) -> None:
        """Buffer one STORE metadata produced by the request tracker.

        The block hashes are read here, at admission, so that the policy
        holds the snapshot the data was stored under.

        One request's operations must be offered in token order, each
        starting where the previous one ended: they are submitted as one
        coalesced store, which a gap makes impossible.

        The operation is rebound to the request's token ledger before it is
        buffered, so deferring N operations of one request costs one copy of
        the token sequence rather than N.

        Args:
            metadata: The store operation offered by the tracker: the token
                range and the GPU blocks holding its KV. Its ``op`` is not
                mutated; a rebound copy is buffered instead.
        """
        pool = self._require_block_pool()
        block_hashes: BlockHashes = {
            block_id: pool.blocks[block_id].block_hash
            for block_id in metadata.op.flat_block_ids
        }
        self._require_policy().add(self._rebind_tokens(metadata), block_hashes)

    def on_scheduler_step(
        self, scheduler_output: "SchedulerOutput"
    ) -> LazyOffloadActions:
        """Drain stores made due by one token-producing scheduler step.

        A zero-token step returns no actions: vLLM takes its no-forward path
        and would discard metadata produced by that step.

        Args:
            scheduler_output: The step's schedule, supplying the block
                pressure the drain decision reads.

        Returns:
            Stores to submit and sessions made releasable by the drain.

        Raises:
            ValueError: If no GPU block pool has been bound, or if a
                request's buffered operations are not contiguous.
            RuntimeError: If the policy emitted for a request that
                already has a store batch in flight.
        """
        if not scheduler_output.total_num_scheduled_tokens:
            return LazyOffloadActions()
        return self._drain(scheduler_output, self._require_block_pool())

    def on_store_results(
        self,
        failed_request_ids: set[str],
        completed_store_counts: dict[str, int],
    ) -> LazyOffloadActions:
        """Apply failed stores and fully aggregated completion receipts.

        Failures go first, so dropping a finished request's held-back suffix
        can make it releasable by the accompanying receipt. Stale receipts in
        ``completed_store_counts`` are filtered here.

        Args:
            failed_request_ids: Requests whose submitted store failed on at
                least one worker.
            completed_store_counts: Worker completions newly reported this
                round, keyed by request id. A batch settles once its count
                reaches the number of workers.

        Returns:
            Sessions made releasable by completed batches.
        """
        pool = self._require_block_pool()
        for request_id in failed_request_ids:
            if not self._requests.has_in_flight(request_id):
                continue
            if self._requests.in_flight_is_orphaned(request_id):
                # A reset or id reuse detached the batch: it still owns its
                # pins, but cannot break the current prefix chain.
                continue
            dropped = self._require_policy().mark_store_failed(request_id)
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
            pool.free_blocks([pool.blocks[block_id] for block_id in batch.block_ids])
            if not self._require_policy().has_pending_request(
                request_id
            ) and self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._release_session(request_id)
        return actions

    def on_request_finished(self, request_id: str) -> LazyOffloadActions:
        """Record request completion and decide whether its session can end.

        Args:
            request_id: The request vLLM reported as finished.

        Returns:
            A session-release action only when no store is pending or in
            flight; otherwise an empty action.
        """
        self._requests.finish(request_id)
        if self._require_policy().has_pending_request(request_id):
            return LazyOffloadActions()
        if self._requests.has_in_flight(request_id):
            return LazyOffloadActions()
        self._release_session(request_id)
        return LazyOffloadActions(sessions_to_end=[request_id])

    def on_request_reset(self, request_id: str) -> None:
        """Drop the operations a preemption reset invalidated.

        The resumed request restarts at token zero and no longer owns the
        blocks its buffered operations point at, so any batch it already
        submitted is detached from it here.

        Args:
            request_id: The preempted request.
        """
        self._requests.reset(request_id)
        self._token_ledgers.pop(request_id, None)
        dropped = self._require_policy().drop_request(request_id)
        if dropped:
            logger.info(
                "Lazy offload: dropped %d buffered store op(s) of preempted request %s",
                dropped,
                request_id,
            )

    def on_request_arrived(self, request_id: str) -> LazyOffloadActions:
        """Reclaim residual state if a new request reuses a finished id.

        Args:
            request_id: The id whose tracker vLLM just created, which may be
                a first arrival, a preempted request coming back, or a new
                request taking over a finished id.

        Returns:
            A session-release action for the predecessor when the id is
            reused and nothing of the predecessor is in flight. With a batch
            still in flight the release is skipped and never happens on its
            own: a session is keyed by request id, so ending it now would
            end the successor's too. The successor's own teardown covers it.
        """
        reused_finished_id = self._requests.is_finished(request_id)
        predecessor_in_flight = self._requests.has_in_flight(request_id)
        self._requests.arrive(request_id)
        if not reused_finished_id:
            return LazyOffloadActions()
        self._require_policy().discard_for_reuse(request_id)
        self._token_ledgers.pop(request_id, None)
        if predecessor_in_flight:
            return LazyOffloadActions()
        logger.info(
            "Lazy offload: request id %s reused while its predecessor's "
            "teardown was deferred; released the predecessor's session",
            request_id,
        )
        return LazyOffloadActions(sessions_to_end=[request_id])

    def has_inflight_store_work(self) -> bool:
        """Whether submitted stores are still holding GPU blocks.

        Buffered operations are deliberately excluded. They are emitted only
        by a step that schedules model tokens, so a connector-only step
        cannot advance them and keeping the engine awake for them would spin.
        A submitted batch is different: its receipt arrives from the worker
        without any further scheduling, and the blocks it pins stay pinned
        until it does.

        Returns:
            True while at least one submitted batch awaits its receipt.
        """
        return bool(self._requests.in_flight_request_ids())

    def log_final_stats(self) -> None:
        """Write the policy's final counter ledger, when one was built."""
        if self._policy is not None:
            self._policy.log_final_stats()

    def _drain(
        self,
        scheduler_output: "SchedulerOutput",
        pool: "BlockPool",
    ) -> LazyOffloadActions:
        """Apply one policy-neutral drain plan and its GPU side effects.

        Each emitted operation is re-validated against the hash snapshot the
        policy kept, pinned, and coalesced into one submission per request.
        A request whose snapshot no longer matches loses that operation and
        every later one of the same drain.

        Args:
            scheduler_output: The step's schedule, converted into the
                block-pressure signals the policy reads.
            pool: The bound GPU block pool, touched and pinned here.

        Returns:
            The stores to submit and the sessions the drain made releasable.

        Raises:
            RuntimeError: If the policy emitted for a request that already
                has a store batch in flight, which would make the worker's
                receipts ambiguous.
        """
        drain = self._require_policy().drain(
            DrainSignals(
                new_blocks_allocated=_new_blocks(scheduler_output),
                est_next_step_blocks=sum(
                    -(-scheduler_output.total_num_scheduled_tokens // tokens_per_block)
                    for tokens_per_block in self._group_tokens_per_block
                ),
                finished_request_ids=self._requests.finished_request_ids(),
                blocked_request_ids=self._requests.in_flight_request_ids(),
            )
        )
        actions = LazyOffloadActions()
        for item in drain.items:
            if self._requests.has_in_flight(item.request_id):
                raise RuntimeError(
                    f"request {item.request_id!r} emitted while a store batch "
                    "is still in flight"
                )
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
                        "dropping its remaining store operations",
                        item.request_id,
                    )
                    pool.free_blocks(blocks)
                    break
                valid_metas.append(metadata)
                valid_block_ids.extend(gpu_block_ids)
            if not valid_metas:
                continue
            actions.stores_to_submit.append(_coalesce_store_metadata(valid_metas))
            self._requests.register_batch(item.request_id, valid_block_ids)
        for request_id in drain.emptied_request_ids:
            if self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._release_session(request_id)
        return actions

    def _rebind_tokens(
        self, metadata: LMCacheMPRequestMetadata
    ) -> LMCacheMPRequestMetadata:
        """Point one operation at its request's token ledger.

        The tracker builds a fresh list of the request's whole token
        sequence for every operation it produces. Under lazy offload those
        lists are retained until the operation is submitted, so a long
        request would hold one copy per buffered operation. Each list is a
        prefix of the next -- vLLM only appends -- so the ledger absorbs the
        new tail and every buffered operation of the request shares it.

        Args:
            metadata: The store operation as the tracker produced it.

        Returns:
            A copy whose ``op.token_ids`` is the shared ledger. The caller's
            metadata is left untouched.
        """
        tokens = metadata.op.token_ids
        ledger = self._token_ledgers.get(metadata.request_id)
        if ledger is None:
            ledger = list(tokens)
            self._token_ledgers[metadata.request_id] = ledger
        elif len(tokens) > len(ledger):
            ledger.extend(tokens[len(ledger) :])
        return replace(metadata, op=replace(metadata.op, token_ids=ledger))

    def _release_session(self, request_id: str) -> None:
        """Clear policy and registry state for a settled request.

        Args:
            request_id: The request whose session the caller is ending.
        """
        self._require_policy().release_request(request_id)
        self._requests.session_ended(request_id)
        self._token_ledgers.pop(request_id, None)

    def _require_policy(self) -> OffloadPolicy:
        """Return the bound policy or reject an invalid lifecycle call.

        Returns:
            The policy built at bind time.

        Raises:
            ValueError: If no GPU block pool has been bound yet.
        """
        if self._policy is None:
            raise ValueError("lazy offload GPU block pool is not bound")
        return self._policy

    def _require_block_pool(self) -> "BlockPool":
        """Return the bound block pool or reject an invalid lifecycle call.

        Returns:
            The scheduler's GPU block pool.

        Raises:
            ValueError: If no GPU block pool has been bound yet.
        """
        if self._gpu_block_pool is None:
            raise ValueError("lazy offload GPU block pool is not bound")
        return self._gpu_block_pool
