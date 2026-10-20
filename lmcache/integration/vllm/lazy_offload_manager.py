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
  A held operation (see ``lazy_offload_policy.base``) is pinned before its
  submission; the submitted batch takes that pin over.
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
import math

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
        max_step_tokens: int,
        num_lookahead_tokens: int,
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
            max_step_tokens: vLLM's ``max_num_batched_tokens``: the most
                tokens one request can be scheduled in one step.
            num_lookahead_tokens: Slots vLLM allocates past the scheduled
                tokens for speculative decoding, which is also the number of
                draft tokens a decode step schedules.
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
        self._token_ledgers: dict[str, list[int]] = {}
        # Blocks pinned for held operations, per request, until submission.
        self._held_block_ids: dict[str, set[int]] = {}
        # Pinned blocks that sat in the free queue when pinned: unpinning
        # returns them to its head, where eviction order had them.
        self._pinned_while_free: set[int] = set()
        # Token counts of admitted requests and, once scheduled, the tokens
        # they will have computed after their last scheduled step: what
        # bounds the blocks they can allocate before the next drain.
        self._num_tokens: dict[str, int] = {}
        self._num_computed: dict[str, int] = {}
        # Requests loading asynchronously -> tokens left after the load.
        self._load_remaining: dict[str, int] = {}
        # Requests whose load finished but that are not yet scheduled -> the
        # blocks their first chunk allocates. vLLM schedules them only when
        # their chunk fits, which can be steps later.
        self._resuming: dict[str, int] = {}
        # The last drain's bound, which the next schedule's running
        # requests allocate before a resumed load does.
        self._next_step_blocks = 0
        # Per group, block ids each scheduled request has been handed, to
        # tell what a step allocated from blocks vLLM only moved within its
        # table, and its table length, which counts the lookahead slots the
        # last allocation covered too.
        self._handed_blocks: dict[str, list[set[int]]] = {}
        self._table_lengths: dict[str, list[int]] = {}

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
        holds the snapshot the data was stored under. The null block is left
        out: hybrid models (mamba align mode, sliding window) put it into
        block tables at positions holding no data, it is never recycled, and
        its hash is always None, which the policy would take for an evicted
        block and reject the request's whole remaining prefix.

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
            if not pool.blocks[block_id].is_null
        }
        self._require_policy().add(self._rebind_tokens(metadata), block_hashes)

    def on_scheduler_step(
        self, scheduler_output: "SchedulerOutput"
    ) -> LazyOffloadActions:
        """Drain stores made due by one scheduler step.

        A step that schedules no tokens drains too: vLLM's no-forward path
        still calls ``wait_for_save``, which submits the stores. It happens
        when blocks run out for every waiting request, and the pins a
        squeezed admission left are then what must be stored to free them.

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
        return self._drain(scheduler_output, self._require_block_pool())

    def on_admission(
        self,
        request_id: str,
        num_tokens: int,
        num_computed_tokens: int,
        num_external_tokens: int,
        load_async: bool,
    ) -> None:
        """Protect buffered blocks an admission is about to recycle.

        vLLM allocates an admitted request's blocks right after asking the
        connector for its external hit, evicting from the free-queue head
        before this step's drain can emit anything there. A long hit
        allocates its whole prefix at once, which no danger depth predicts.
        Pinning those blocks would shrink the free queue the allocation
        draws from and could refuse it, so they move to the tail instead,
        still allocatable and still buffered. Only when the allocation would
        reach the tail too are they pinned now.

        Args:
            request_id: The request being admitted.
            num_tokens: Its tokens, prompt and any output so far.
            num_computed_tokens: Its local prefix hit, in tokens.
            num_external_tokens: Its external hit, in tokens.
            load_async: Whether the hit loads asynchronously. vLLM then
                allocates for the hit alone and schedules the rest once the
                load finishes, which :meth:`on_loads_finished` covers.
        """
        self._num_tokens[request_id] = num_tokens
        remaining = num_tokens - num_computed_tokens - num_external_tokens
        if load_async:
            self._load_remaining[request_id] = remaining
            num_new_tokens = num_external_tokens
        else:
            num_new_tokens = num_external_tokens + self._chunk_tokens(remaining)
        # vLLM touches the local hit before allocating, which takes its free
        # blocks out of the queue, so the allocation reaches that much further.
        # An align-mode Mamba hit is null blocks but its last state block, and
        # an external hit gets one state block of its own before the
        # allocation, outside the allocation's cap.
        num_hit_blocks = sum(
            min(1, num_computed_tokens) + min(1, num_external_tokens)
            if group_idx in self._mamba_groups
            else -(-num_computed_tokens // tokens_per_block)
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )
        self._protect(self._span_blocks(num_new_tokens) + num_hit_blocks)

    def on_loads_finished(self, request_ids: set[str]) -> None:
        """Protect buffered blocks that resuming finished loads will recycle.

        A request whose asynchronous load finished is scheduled again
        without asking the connector for a hit, so this is the last call
        before it allocates for the rest of its tokens. The next schedule's
        running requests allocate first, so their bound counts too. A full
        cache can keep the request waiting for several steps, while the
        free-queue head moves past the protected blocks; each drain until it
        is scheduled counts its allocation in the danger depth.

        Args:
            request_ids: Requests whose load the workers reported finished.
                Ids without an asynchronous load are ignored.
        """
        resumed = {
            request_id: self._span_blocks(
                self._chunk_tokens(self._load_remaining.pop(request_id)),
                first_allocation=False,
            )
            for request_id in request_ids
            if request_id in self._load_remaining
        }
        if not resumed:
            return
        self._resuming.update(resumed)
        self._protect(self._next_step_blocks + sum(resumed.values()))

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
        for request_id in failed_request_ids:
            if not self._requests.has_in_flight(request_id):
                continue
            if self._requests.in_flight_is_orphaned(request_id):
                # A reset or id reuse detached the batch: it still owns its
                # pins, but cannot break the current prefix chain.
                continue
            dropped = self._require_policy().mark_store_failed(request_id)
            self._release_holds(request_id)
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
            self._unpin(list(batch.block_ids))
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
        self._forget_progress(request_id)
        dropped = self._require_policy().drop_request(request_id)
        self._release_holds(request_id)
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
        self._release_holds(request_id)
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

        Held operations are pinned first. Each emitted operation is
        re-validated against the hash snapshot the policy kept, pinned unless
        it was held, and coalesced into one submission per request. A request
        whose snapshot no longer matches loses that operation and every later
        one of the same drain.

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
        self._next_step_blocks = self._running_blocks_bound(scheduler_output)
        for request_id in scheduler_output.num_scheduled_tokens:
            self._resuming.pop(request_id, None)
        drain = self._require_policy().drain(
            DrainSignals(
                next_step_blocks=self._next_step_blocks,
                resuming_blocks=sum(self._resuming.values()),
                finished_request_ids=self._requests.finished_request_ids(),
                blocked_request_ids=self._requests.in_flight_request_ids(),
                preempted_request_ids=set(scheduler_output.preempted_req_ids or ()),
                running_blocks_allocated=running_blocks_allocated,
            )
        )
        self._pin_holds(drain.holds)
        self._move_to_tail(drain.to_tail)
        actions = LazyOffloadActions()
        for item in drain.items:
            if self._requests.has_in_flight(item.request_id):
                raise RuntimeError(
                    f"request {item.request_id!r} emitted while a store batch "
                    "is still in flight"
                )
            held = self._held_block_ids.pop(item.request_id, set())
            valid_metas: list[LMCacheMPRequestMetadata] = []
            valid_block_ids: list[int] = []
            for metadata, old_block_hashes in item.metadatas:
                gpu_block_ids = list(old_block_hashes)
                self._pin(
                    [block_id for block_id in gpu_block_ids if block_id not in held]
                )
                held.difference_update(gpu_block_ids)
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
                    self._unpin(gpu_block_ids)
                    break
                valid_metas.append(metadata)
                valid_block_ids.extend(gpu_block_ids)
            if held:
                # Held ops the drain budget left buffered keep their pins.
                self._held_block_ids[item.request_id] = held
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

    def _running_blocks_allocated(self, scheduler_output: "SchedulerOutput") -> int:
        """Count the blocks this step allocated to requests already running.

        Admissions and preempted requests resuming are left out: the bound
        covers neither. A step's new block ids also carry null blocks and
        Mamba speculative blocks vLLM moved to the table's end, which come
        from no free queue. A block the request freed and was handed back
        later is missed, which can only hide an excess, never invent one.

        Args:
            scheduler_output: This step's schedule.

        Returns:
            The blocks allocated, over every KV-cache group.
        """
        null_block_id = self._require_block_pool().null_block.block_id
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
            groups = new_block_ids or ([],) * num_groups
            if request_id in cached.resumed_req_ids:
                self._handed_blocks[request_id] = [set(group) for group in groups]
                self._table_lengths[request_id] = [len(group) for group in groups]
                continue
            known = self._handed_blocks.setdefault(
                request_id, [set() for _ in range(num_groups)]
            )
            lengths = self._table_lengths.setdefault(request_id, [0] * num_groups)
            for group_idx, group in enumerate(groups):
                handed = set(group)
                allocated += len(handed - known[group_idx] - {null_block_id})
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
        for request_id in scheduler_output.finished_req_ids:
            self._forget_progress(request_id)
        for request_id in scheduler_output.preempted_req_ids or ():
            self._forget_progress(request_id)
        scheduled = scheduler_output.num_scheduled_tokens
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
            remaining = self._num_tokens[request_id] - num_computed
            lengths = self._table_lengths[request_id]
            if remaining > 0:
                running.append(
                    (
                        num_computed,
                        min(remaining, self._max_step_tokens),
                        remaining,
                        lengths,
                    )
                )
            else:
                running.append(
                    (num_computed, 1 + self._num_lookahead_tokens, 0, lengths)
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

        Every crossing yields as many blocks as any other in the common
        case, so the cheapest crossings that fit the budget whole are the
        most any schedule allocates. Otherwise the cheapest blocks per
        token are taken first, fractionally, which bounds it too.

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
        first_blocks: list[tuple[int, int]] = []  # (token cost, blocks)
        num_further = 0
        for num_computed, max_scheduled, remaining, lengths in running:
            # Mamba tables also hold the state blocks reserved per draft
            # token, so their length says nothing about the tokens covered.
            num_allocated = (
                -(-num_computed // tokens_per_block)
                if mamba_extra is not None
                else lengths[group_idx]
            )
            reachable = (
                -(-(num_computed + max_scheduled + lookahead) // tokens_per_block)
                - num_allocated
            )
            if reachable <= 0:
                continue
            room = num_allocated * tokens_per_block - num_computed
            cost = max(1, room + 1 - lookahead)
            if remaining and alignment:
                aligned_end = -(-(num_computed + cost) // alignment) * alignment
                cost = min(aligned_end - num_computed, remaining)
                if cost > max_scheduled:
                    continue
            if mamba_extra is not None:
                first_blocks.append((cost, 1 + mamba_extra))
                continue
            blocks = (
                -(-(num_computed + cost + lookahead) // tokens_per_block)
                - num_allocated
            )
            first_blocks.append((cost, blocks))
            num_further += reachable - blocks
        if len({blocks for _, blocks in first_blocks}) <= 1 and (
            not num_further or all(blocks == 1 for _, blocks in first_blocks)
        ):
            return self._whole_crossings(first_blocks, num_further, tokens_per_block)
        budget = float(self._max_step_tokens)
        bound = 0.0
        for cost, blocks in sorted(first_blocks, key=lambda item: item[0] / item[1]):
            if cost > budget:
                bound += blocks * budget / cost
                break
            budget -= cost
            bound += blocks
        else:
            bound += min(num_further, budget / tokens_per_block)
        return math.ceil(bound)

    def _whole_crossings(
        self,
        first_blocks: list[tuple[int, int]],
        num_further: int,
        tokens_per_block: int,
    ) -> int:
        """Bound a group's new blocks when every crossing yields as many.

        Each crossing is paid in full, so with equal yields the most blocks
        come from the cheapest crossings that fit the budget. That makes a
        further block a crossing at a whole block's cost too, ignoring
        that it needs its request's first one, which can only overcount.

        Args:
            first_blocks: Per request that can cross: the tokens its first
                crossing costs and the blocks it yields, the same for all.
            num_further: Further blocks the requests can reach, one each.
            tokens_per_block: Tokens per block in the group.

        Returns:
            The bound for the group.
        """
        if not first_blocks:
            return 0
        costs = sorted(cost for cost, _ in first_blocks)
        budget = self._max_step_tokens
        crossings = 0
        index = 0
        while True:
            if index < len(costs) and (
                not num_further or costs[index] <= tokens_per_block
            ):
                cost = costs[index]
                index += 1
            elif num_further:
                cost = tokens_per_block
                num_further -= 1
            else:
                break
            if cost > budget:
                break
            budget -= cost
            crossings += 1
        return crossings * first_blocks[0][1]

    def _chunk_tokens(self, remaining: int) -> int:
        """Bound the slots one scheduled chunk of a request allocates.

        Args:
            remaining: The request's tokens not yet computed. A full hit
                leaves none, and vLLM then recomputes the last token.

        Returns:
            One step's worth of those tokens, plus the lookahead slots.
        """
        chunk = min(max(remaining, 1), self._max_step_tokens)
        return chunk + self._num_lookahead_tokens

    def _span_blocks(self, num_tokens: int, first_allocation: bool = True) -> int:
        """Bound the blocks an allocation of ``num_tokens`` slots takes.

        Args:
            num_tokens: The slots allocated, starting anywhere in a block.
            first_allocation: Whether the request holds no blocks yet. An
                asynchronous load's admission allocated for its hit, so its
                resumed allocation is not the first.

        Returns:
            Per group, summed over the groups: the blocks the slots fill, a
            partial one at the start, one for a partial hit's copy-on-write
            block or a Mamba checkpoint, and the Mamba state blocks a
            request's first allocation reserves, one per draft token. An
            align-mode Mamba group nulls the blocks the slots fill but its
            state block, so its allocation is capped whatever their number,
            and only a first allocation's cap adds the draft-token blocks.
        """
        extra = 2 + self._num_lookahead_tokens
        mamba_draft_blocks = self._num_lookahead_tokens if first_allocation else 0
        return sum(
            1 + self._mamba_groups[group_idx] + mamba_draft_blocks
            if group_idx in self._mamba_groups
            else -(-num_tokens // tokens_per_block) + extra
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )

    def _protect(self, num_blocks: int) -> None:
        """Move or pin the buffered blocks an allocation would recycle.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.
        """
        protection = self._require_policy().protect(num_blocks)
        self._move_to_tail(protection.to_tail)
        self._pin_holds(protection.holds)

    def _move_to_tail(self, items: list[PendingStoreItem]) -> None:
        """Move buffered blocks to the free-queue tail, unpinned.

        A cached block is freed to the tail with its hash intact; a block a
        request still owns stays out of the queue.

        Args:
            items: The operations whose blocks move, in prefix order.
        """
        pool = self._require_block_pool()
        for item in items:
            for _, block_hashes in item.metadatas:
                blocks = [pool.blocks[block_id] for block_id in block_hashes]
                pool.touch(blocks)
                pool.free_blocks(blocks)

    def _forget_progress(self, request_id: str) -> None:
        """Stop counting a request toward the running requests' bound.

        Args:
            request_id: A request that finished or lost its blocks.
        """
        self._num_tokens.pop(request_id, None)
        self._num_computed.pop(request_id, None)
        self._load_remaining.pop(request_id, None)
        self._resuming.pop(request_id, None)
        self._handed_blocks.pop(request_id, None)
        self._table_lengths.pop(request_id, None)

    def _pin_holds(self, holds: list[PendingStoreItem]) -> None:
        """Pin the blocks of operations the policy just marked held.

        The policy checked these snapshots in the same call, and nothing
        allocates between its check and this pin.

        Args:
            holds: The policy's new holds, one entry per request.
        """
        for hold in holds:
            held = self._held_block_ids.setdefault(hold.request_id, set())
            for _, block_hashes in hold.metadatas:
                self._pin(list(block_hashes))
                held.update(block_hashes)

    def _release_holds(self, request_id: str) -> None:
        """Unpin the blocks of a request's held operations the policy dropped.

        Args:
            request_id: The request whose held operations were discarded.
        """
        self._unpin(list(self._held_block_ids.pop(request_id, set())))

    def _pin(self, block_ids: list[int]) -> None:
        """Pin blocks, remembering which ones were waiting for eviction.

        Args:
            block_ids: The blocks to pin, in eviction order.
        """
        pool = self._require_block_pool()
        blocks = [pool.blocks[block_id] for block_id in block_ids]
        self._pinned_while_free.update(
            block_id
            for block_id, block in zip(block_ids, blocks, strict=True)
            if block.ref_cnt == 0
        )
        pool.touch(blocks)

    def _unpin(self, block_ids: list[int]) -> None:
        """Release pins, returning formerly free blocks to the queue head.

        ``BlockPool.free_blocks`` appends a cached block to the tail as if
        just used; a block this manager pinned out of the free queue was
        near the head, so the head is closer to its place in eviction order.
        A block another request still owns stays wherever vLLM puts it.

        Args:
            block_ids: The pinned blocks to release, in eviction order.
        """
        pool = self._require_block_pool()
        blocks = [pool.blocks[block_id] for block_id in block_ids]
        pool.free_blocks(blocks)
        returned = []
        for block_id, block in zip(block_ids, blocks, strict=True):
            if block_id not in self._pinned_while_free:
                continue
            self._pinned_while_free.discard(block_id)
            if block.ref_cnt == 0:
                pool.free_block_queue.remove(block)
                returned.append(block)
        pool.free_block_queue.prepend_n(returned)

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
