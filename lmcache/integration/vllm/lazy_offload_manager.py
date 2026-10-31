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
  arrived. Its pins are still released, but its failure is not retried for
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
    PendingStoreItem,
)
from lmcache.integration.vllm.lazy_offload_state import LazyOffloadRequestRegistry
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LoadStoreOp,
)
from lmcache.utils import cdiv, init_logger, round_up

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Failed stores in a row after which a request's stores are given up: a
# failure that outlasts the retries would otherwise be retried forever and
# keep the request's session open.
MAX_STORE_ATTEMPTS = 3


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
        token_bytes=last.op.token_bytes,
        block_ids=merged_block_ids,
        start=first.op.start,
        end=last.op.end,
    )
    return LMCacheMPRequestMetadata(
        request_id=first.request_id,
        direction="STORE",
        op=merged_op,
        cache_salt=first.cache_salt,
        request_configs=first.request_configs,
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
        max_step_tokens: int,
        num_lookahead_tokens: int,
        max_model_len: int,
        mamba_groups: dict[int, int] | None = None,
        prefill_alignment: int = 0,
    ) -> None:
        """Create an unbound scheduler-side manager.

        Args:
            configs: Connector extra config, read by the policy at bind time.
            group_tokens_per_block: Token capacity per KV-cache group, used
                to bound how many blocks an allocation can take. Scratch
                groups report ``0`` and are left out of every bound.
            completion_tracker: Adapter view aggregating per-worker receipts.
            max_step_tokens: vLLM's per-step token budget
                (``max_num_scheduled_tokens``, else ``max_num_batched_tokens``):
                the most tokens one step schedules over all requests together,
                and so for any one of them.
            num_lookahead_tokens: Slots vLLM allocates past the scheduled
                tokens for speculative decoding, which is also the number of
                draft tokens a decode step schedules.
            max_model_len: vLLM's ``max_model_len``: the most tokens one
                request holds, which bounds what one admission allocates.
            mamba_groups: Align-mode Mamba groups -> the blocks a running
                request's step can take beyond its one state block: a
                prefill checkpoint block and a partial-hit copy, where the
                model has them. Such a group sizes its state by the
                scheduled tokens alone, without lookahead slots, and nulls
                the positions a step skips rather than allocating them.
            prefill_alignment: Tokens a prefill chunk that stops short of
                the prompt's end ends on a multiple of, as vLLM splits
                chunks of Mamba-hybrid models; 0 when a chunk may end
                anywhere.
        """
        self._configs = dict(configs or {})
        self._group_tokens_per_block = list(group_tokens_per_block)
        self._max_step_tokens = max_step_tokens
        self._num_lookahead_tokens = num_lookahead_tokens
        self._mamba_groups = dict(mamba_groups or {})
        self._prefill_alignment = prefill_alignment
        self._completion_tracker = completion_tracker
        # Both are set by bind_block_pool: the policy reads the pool.
        self._gpu_block_pool: "BlockPool | None" = None
        self._policy: OffloadPolicy | None = None
        self._requests = LazyOffloadRequestRegistry()
        # One token ledger per request whose operations are buffered.
        self._token_ledgers: dict[str, bytearray] = {}
        # Request id -> its store batches that failed in a row.
        self._failed_store_attempts: dict[str, int] = {}
        # Token counts of admitted requests and, once scheduled, the tokens
        # they will have computed after their last scheduled step: what
        # bounds the blocks they can allocate before the next drain.
        self._num_tokens: dict[str, int] = {}
        self._num_computed: dict[str, int] = {}
        self._admission_blocks = self._admission_bound(max_model_len)
        # Per group, block ids each scheduled request has been handed, to
        # tell what a step allocated from blocks vLLM only moved within its
        # table, and its table length, which counts the lookahead slots the
        # last allocation covered too.
        self._handed_blocks: dict[str, list[set[int]]] = {}
        self._table_lengths: dict[str, list[int]] = {}

    @property
    def admission_blocks(self) -> int:
        """Free-queue head each drain keeps clear of buffered blocks for one
        admission: the most blocks one admission can allocate, from
        ``max_model_len``. It deepens the danger window beyond the running
        requests' bound, so a step's first admission recycles no buffered
        block.

        Returns:
            The bound, in blocks summed over the cacheable groups.
        """
        return self._admission_blocks

    def bind_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the scheduler's GPU block pool and build the policy.

        Idempotent for the same pool; rebinding a different one would
        silently invalidate every buffered hash snapshot.

        Args:
            gpu_block_pool: The scheduler's block pool, from which the
                manager reads block hashes and which it pins into.

        Raises:
            ValueError: If a different pool is bound, or the policy name is
                unknown.
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

        Lazy offload stores a block at the latest moment before vLLM reuses
        it. vLLM frees hash-less blocks to the head of the free queue,
        reused first, and hashed blocks to the tail in LRU order
        (``BlockPool.free_blocks``); an allocation takes blocks from the
        head (``get_new_blocks``). A hashed block is reused only after the
        blocks ahead of it, so lazy offload can track it as it moves toward
        the head and store it before reuse: it has slack. A hash-less block
        is taken by the next allocation: it has no slack. Pinning it to
        store it later would make vLLM take the next free block instead,
        evicting a hashed, still reusable block early, which trades a
        reusable block for one vLLM does not intend to reuse. So lazy
        offload covers only hashed blocks, which are exactly the ones vLLM
        retains past the step: it hashes them in ``allocate_slots``, before
        this call. A hash-less block (an align-mode recurrent state sparse
        retention does not keep) is treated as absent: replaced by the null
        block in the buffered copy, like the null block hybrid models put
        at positions holding no data, and it never breaks the prefix chain,
        which is defined over hashed blocks only. The server skips a chunk
        that is null in every group. Full attention blocks in a store op
        are always hashed, since an op covers whole computed blocks and
        lazy offload requires prefix caching. The hashes of the remaining
        blocks are read here, at admission, so that the policy holds the
        snapshot the data was stored under.

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
        null_block_id = pool.null_block.block_id
        protected_block_ids = [
            [
                block_id
                if pool.blocks[block_id].block_hash is not None
                else null_block_id
                for block_id in group_block_ids
            ]
            for group_block_ids in metadata.op.block_ids
        ]
        block_hashes: BlockHashes = {
            block_id: pool.blocks[block_id].block_hash
            for group_block_ids in protected_block_ids
            for block_id in group_block_ids
            if block_id != null_block_id
        }
        rebound = self._rebind_tokens(metadata)
        rebound = replace(
            rebound, op=replace(rebound.op, block_ids=protected_block_ids)
        )
        self._require_policy().add(rebound, block_hashes)

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

    def on_admission(self, request_id: str, num_tokens: int) -> None:
        """Record a request's token count when its lookup resolves.

        Called from ``get_num_new_matched_tokens``, possibly more than once
        before vLLM admits the request; the latest count wins.

        The count bounds what the request allocates once it runs: a prefill
        chunk stops at its last token, and a decode step takes one token
        and its drafts.

        Args:
            request_id: The request being admitted.
            num_tokens: Its tokens, prompt and any output so far.
        """
        self._num_tokens[request_id] = num_tokens

    def on_store_results(
        self,
        failed_request_ids: set[str],
        completed_store_counts: dict[str, int],
    ) -> LazyOffloadActions:
        """Apply failed stores and fully aggregated completion receipts.

        A failure only marks the in-flight batch: other workers may still be
        reading its blocks. Once every worker has reported, a failed batch
        that is not orphaned goes back to the policy to be retried; unpinned,
        its blocks join the free-queue tail. Stale receipts in
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
            if self._requests.has_in_flight(request_id):
                self._requests.fail_in_flight(request_id)
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
            if not batch.failed:
                self._failed_store_attempts.pop(request_id, None)
            elif not batch.orphaned:
                # The data is still on the GPU; a reset or id reuse would
                # have detached the batch from the generation it belongs to.
                self._retry_or_give_up(batch.operations)
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
        self._forget_progress(request_id)
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
        self._failed_store_attempts.pop(request_id, None)
        self._forget_progress(request_id)
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
        self._failed_store_attempts.pop(request_id, None)
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
        running_blocks_allocated = self._running_blocks_allocated(scheduler_output)
        drain = self._require_policy().drain(
            DrainSignals(
                next_step_blocks=self._running_blocks_bound(scheduler_output),
                admission_blocks=self._admission_blocks,
                finished_request_ids=self._requests.finished_request_ids(),
                blocked_request_ids=self._requests.in_flight_request_ids(),
                running_blocks_allocated=running_blocks_allocated,
            )
        )
        actions = LazyOffloadActions()
        for item in drain.items:
            if self._requests.has_in_flight(item.request_id):
                raise RuntimeError(
                    f"request {item.request_id!r} emitted while a store batch "
                    "is still in flight"
                )
            valid = PendingStoreItem(request_id=item.request_id)
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
                valid.metadatas.append((metadata, old_block_hashes))
                valid_block_ids.extend(gpu_block_ids)
            if not valid.metadatas:
                continue
            actions.stores_to_submit.append(
                _coalesce_store_metadata([meta for meta, _ in valid.metadatas])
            )
            self._requests.register_batch(item.request_id, valid_block_ids, valid)
        for request_id in drain.emptied_request_ids:
            if self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._release_session(request_id)
        return actions

    def _rebind_tokens(
        self, metadata: LMCacheMPRequestMetadata
    ) -> LMCacheMPRequestMetadata:
        """Point one operation at its request's token ledger.

        The tracker builds a fresh buffer of the request's whole token
        sequence for every operation it produces. Under lazy offload those
        buffers are retained until the operation is submitted, so a long
        request would hold one copy per buffered operation. Each buffer is a
        prefix of the next -- vLLM only appends -- so the ledger absorbs the
        new tail and every buffered operation of the request shares it.

        The ledger is a ``bytearray`` so it can grow in place and the
        already-buffered operations see the growth.

        Args:
            metadata: The store operation as the tracker produced it.

        Returns:
            A copy whose ``op.token_bytes`` is the shared ledger. The
            caller's metadata is left untouched.
        """
        tokens = metadata.op.token_bytes
        ledger = self._token_ledgers.get(metadata.request_id)
        if ledger is None:
            ledger = bytearray(tokens)
            self._token_ledgers[metadata.request_id] = ledger
        elif len(tokens) > len(ledger):
            ledger += tokens[len(ledger) :]
        return replace(metadata, op=replace(metadata.op, token_bytes=ledger))

    def _running_blocks_allocated(self, scheduler_output: "SchedulerOutput") -> int:
        """Count the blocks this step allocated to requests already running.

        Each scheduled request's table is tracked per group on the way: a
        new or resumed request's is replaced, a running one's grows by the
        block ids this step handed it. Admissions and preempted requests
        resuming are left out of the count: the bound covers neither. A
        step's new block ids also carry null blocks and Mamba speculative
        blocks vLLM moved to the table's end, which come from no free queue.
        A block the request freed and was handed back later is missed, which
        can only hide an excess, never invent one.

        Args:
            scheduler_output: This step's schedule.

        Returns:
            The blocks allocated, over every KV-cache group.
        """
        null_block = {self._require_block_pool().null_block.block_id}
        num_groups = len(self._group_tokens_per_block)
        for new_request in scheduler_output.scheduled_new_reqs:
            self._handed_blocks[new_request.req_id] = [
                set(group) for group in new_request.block_ids
            ]
            self._table_lengths[new_request.req_id] = [
                len(group) for group in new_request.block_ids
            ]
        cached = scheduler_output.scheduled_cached_reqs
        allocated = 0
        for request_id, new_block_ids in zip(
            cached.req_ids, cached.new_block_ids, strict=True
        ):
            groups: tuple[list[int], ...] = new_block_ids or ([],) * num_groups
            if request_id in cached.resumed_req_ids:
                self._handed_blocks[request_id] = [set(group) for group in groups]
                self._table_lengths[request_id] = [len(group) for group in groups]
                continue
            # vLLM moves a request into its running queue only by scheduling
            # it new or resumed, so its table is already recorded.
            known = self._handed_blocks[request_id]
            lengths = self._table_lengths[request_id]
            for group_idx, group in enumerate(groups):
                # Most steps hand a decoding request nothing new in a group.
                if not group:
                    continue
                handed = set(group)
                allocated += len(handed - known[group_idx] - null_block)
                known[group_idx] |= handed
                lengths[group_idx] += len(group)
        return allocated

    def _running_blocks_bound(self, scheduler_output: "SchedulerOutput") -> int:
        """Bound the blocks running requests can allocate in the next step.

        Each request scheduled this step has computed its scheduled tokens
        by the next one (spec-decode rejections only lower that), and vLLM
        allocated their slots already. The next step schedules each request
        at most one chunk, or a decode token and its drafts, and all of
        them together at most ``max_step_tokens``; slots run ``lookahead``
        past what a request schedules. Per group, the most blocks those
        slots can cross bounds what vLLM allocates.

        Args:
            scheduler_output: This step's schedule.

        Returns:
            The bound, over every running request and KV-cache group.
        """
        scheduled = scheduler_output.num_scheduled_tokens
        for request_id in scheduler_output.preempted_req_ids or ():
            # reset_prefix_cache(reset_running_requests=True) preempts and
            # resumes a request in one step: its progress is already current.
            if request_id not in scheduled:
                self._forget_progress(request_id)
        for new_request in scheduler_output.scheduled_new_reqs:
            self._num_computed[new_request.req_id] = (
                new_request.num_computed_tokens + scheduled[new_request.req_id]
            )
        cached = scheduler_output.scheduled_cached_reqs
        for request_id, num_computed in zip(
            cached.req_ids, cached.num_computed_tokens, strict=True
        ):
            self._num_computed[request_id] = num_computed + scheduled[request_id]
        running: list[tuple[int, int, int, list[int]]] = []
        for request_id, num_computed in self._num_computed.items():
            remaining = max(self._num_tokens[request_id] - num_computed, 0)
            max_scheduled = (
                min(remaining, self._max_step_tokens)
                if remaining
                else 1 + self._num_lookahead_tokens
            )
            running.append(
                (
                    num_computed,
                    max_scheduled,
                    remaining,
                    self._table_lengths[request_id],
                )
            )
        return sum(
            self._budgeted_blocks(running, group_idx)
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )

    def _budgeted_blocks(
        self, running: list[tuple[int, int, int, list[int]]], group_idx: int
    ) -> int:
        """Bound one group's new blocks within the next step's token budget.

        A request's first new block costs the tokens that fill the blocks
        its table holds, less the lookahead slots; each further one costs a
        whole block. The table already holds the blocks the last step's
        lookahead slots reached. A prefill chunk stopping short of the
        prompt's end ends on an aligned position, which can make the first
        crossing cost up to a whole block too. An align-mode Mamba group
        takes one state block per step, plus its extra blocks.

        The bound is the fractional knapsack over every crossing with the
        step's token budget: the cheapest blocks per token are taken first,
        and the crossing the budget ends in counts in full. Dropping the
        rule that a request's further crossings need its first one only
        raises the optimum, so the bound is at least what any schedule
        allocates; the crossing taken in part costs at most one crossing
        more, and only on a step the budget binds.

        Args:
            running: Per running request: its computed tokens, the most
                tokens the next step can schedule for it, its prompt tokens
                not yet computed (zero once decoding), and its table length
                per group.
            group_idx: The KV-cache group.

        Returns:
            The bound for this group.
        """
        tokens_per_block = self._group_tokens_per_block[group_idx]
        mamba_extra = self._mamba_groups.get(group_idx)
        lookahead = 0 if mamba_extra is not None else self._num_lookahead_tokens
        alignment = self._prefill_alignment
        # (token cost, blocks yielded, count) per kind of crossing.
        crossings: list[tuple[int, int, int]] = []
        num_further = 0
        for num_computed, max_scheduled, remaining, lengths in running:
            # Mamba tables also hold the state blocks reserved per draft
            # token, so their length says nothing about the tokens covered.
            num_allocated = (
                cdiv(num_computed, tokens_per_block)
                if mamba_extra is not None
                else lengths[group_idx]
            )
            reachable = (
                cdiv(num_computed + max_scheduled + lookahead, tokens_per_block)
                - num_allocated
            )
            if reachable <= 0:
                continue
            room = num_allocated * tokens_per_block - num_computed
            cost = max(1, room + 1 - lookahead)
            if remaining and alignment:
                aligned_end = round_up(num_computed + cost, alignment)
                cost = min(aligned_end - num_computed, remaining)
                if cost > max_scheduled:
                    continue
            if mamba_extra is not None:
                crossings.append((cost, 1 + mamba_extra, 1))
                continue
            blocks = (
                cdiv(num_computed + cost + lookahead, tokens_per_block) - num_allocated
            )
            crossings.append((cost, blocks, 1))
            num_further += reachable - blocks
        crossings.append((tokens_per_block, 1, num_further))
        budget = self._max_step_tokens
        bound = 0
        for cost, blocks, count in sorted(
            crossings, key=lambda item: item[0] / item[1]
        ):
            taken = min(count, budget // cost)
            bound += taken * blocks
            budget -= taken * cost
            if taken < count:
                bound += cdiv(blocks * budget, cost)
                break
        return bound

    def _span_blocks(self, num_tokens: int) -> int:
        """Bound the blocks a request's first allocation of ``num_tokens`` takes.

        Args:
            num_tokens: The slots allocated, starting anywhere in a block.

        Returns:
            Per group, summed over the groups: the blocks the slots fill, a
            partial one at the start, one for a partial hit's copy-on-write
            block or a Mamba checkpoint, and one per lookahead token. An
            align-mode Mamba group nulls the blocks the slots fill but its
            state block, so its allocation is capped whatever their number:
            the state block, its extra blocks, and the state blocks a
            request's first allocation reserves, one per lookahead token.
        """
        extra = 2 + self._num_lookahead_tokens
        return sum(
            1 + self._mamba_groups[group_idx] + self._num_lookahead_tokens
            if group_idx in self._mamba_groups
            else cdiv(num_tokens, tokens_per_block) + extra
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )

    def _admission_bound(self, max_model_len: int) -> int:
        """Bound the blocks any one admission allocates.

        An admission's local hit plus the slots it allocates hold at most
        ``max_model_len`` tokens, plus the recomputed last token of a full
        hit and the lookahead slots. Splitting them between the hit and the
        allocation costs a group at most one block more than allocating them
        all: two partial blocks instead of one. An align-mode Mamba group's
        hit is at most its local and its external state block.

        Args:
            max_model_len: The most tokens one request holds.

        Returns:
            The bound, in blocks summed over the cacheable groups.
        """
        split_blocks = sum(
            2 if group_idx in self._mamba_groups else 1
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )
        return (
            self._span_blocks(max_model_len + 1 + self._num_lookahead_tokens)
            + split_blocks
        )

    def _forget_progress(self, request_id: str) -> None:
        """Stop counting a request toward the running requests' bound.

        Args:
            request_id: A request that finished or lost its blocks.
        """
        self._num_tokens.pop(request_id, None)
        self._num_computed.pop(request_id, None)
        self._handed_blocks.pop(request_id, None)
        self._table_lengths.pop(request_id, None)

    def _retry_or_give_up(self, item: PendingStoreItem) -> None:
        """Buffer a failed batch again, or give up its request's stores.

        After ``MAX_STORE_ATTEMPTS`` failed batches in a row, the batch and
        the request's buffered operations are dropped
        (``OffloadPolicy.mark_store_failed``).

        Args:
            item: The failed batch's operations, not orphaned.
        """
        request_id = item.request_id
        policy = self._require_policy()
        attempts = self._failed_store_attempts.pop(request_id, 0) + 1
        # Requeued first even when given up, so the policy's ledger counts
        # the batch among the dropped operations.
        policy.requeue_failed(item)
        if attempts < MAX_STORE_ATTEMPTS:
            self._failed_store_attempts[request_id] = attempts
            logger.warning(
                "Store failed for request %s; retrying its %d store op(s)",
                request_id,
                len(item.metadatas),
            )
            return
        dropped = policy.mark_store_failed(request_id)
        logger.warning(
            "Store failed for request %s %d times in a row; dropped its %d "
            "store op(s) and stopped storing the request",
            request_id,
            attempts,
            dropped,
        )

    def _release_session(self, request_id: str) -> None:
        """Clear policy and registry state for a settled request.

        Args:
            request_id: The request whose session the caller is ending.
        """
        self._failed_store_attempts.pop(request_id, None)
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
