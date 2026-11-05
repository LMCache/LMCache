# SPDX-License-Identifier: Apache-2.0
"""Drain behaviour of the eviction-aware lazy-offload policy.

Pure policy logic: no vLLM, no torch, no GPU. The block pool is the shared
``FakeBlockPool``, which offers the only two surfaces this policy reads: the
free queue as a linked list, where position is eviction rank, and each block's
current hash. The danger depth is the sum of the two bounds the manager passes in
``DrainSignals``; deriving them from the schedule is tested with the manager.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
import logging

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy import eviction_aware
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    DrainSignals,
    LazyOffloadDrain,
)
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    _STATS_LOG_INTERVAL_S,
    EvictionAwareStoreQueue,
    LazyOffloadCounters,
)
from tests.v1.lazy_offload.block_pool_fake import FakeBlockPool

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


@dataclass(frozen=True)
class FakeStoreOp:
    """The token range of one store operation."""

    start: int
    end: int


@dataclass(frozen=True)
class FakeStoreMetadata:
    """Opaque payload standing in for ``LMCacheMPRequestMetadata``."""

    request_id: str
    op: FakeStoreOp


def add_op(
    queue: EvictionAwareStoreQueue,
    pool: FakeBlockPool,
    request_id: str,
    block_ids: list[int],
    end_tokens: int,
    start_tokens: int = -1,
) -> None:
    """Admit one op whose hash snapshot is the pool's current state."""
    if start_tokens < 0:
        start_tokens = max(0, end_tokens - 256)
    meta = cast(
        "LMCacheMPRequestMetadata",
        FakeStoreMetadata(request_id, FakeStoreOp(start_tokens, end_tokens)),
    )
    hashes = cast(
        BlockHashes,
        {block_id: pool.blocks[block_id].block_hash for block_id in block_ids},
    )
    queue.add(meta, hashes)


def add_unhashed_op(
    queue: EvictionAwareStoreQueue, request_id: str, end_tokens: int
) -> None:
    """Admit one op whose single covered block carries no hash."""
    meta = cast(
        "LMCacheMPRequestMetadata",
        FakeStoreMetadata(
            request_id, FakeStoreOp(max(0, end_tokens - 256), end_tokens)
        ),
    )
    queue.add(meta, cast(BlockHashes, {999: None}))


def drain(
    queue: EvictionAwareStoreQueue,
    next_step: int = 0,
    admission: int = 0,
    finished: frozenset[str] = frozenset(),
    blocked: frozenset[str] = frozenset(),
) -> LazyOffloadDrain:
    """Run one drain step whose danger depth is ``next_step + admission``."""
    return queue.drain(
        DrainSignals(
            next_step_blocks=next_step,
            admission_blocks=admission,
            finished_request_ids=set(finished),
            blocked_request_ids=set(blocked),
            running_blocks_allocated=0,
        )
    )


def emitted_ends(result: LazyOffloadDrain) -> list[int]:
    """The end-token of every emitted op, in emission order."""
    return [meta.op.end for item in result.items for meta, _ in item.metadatas]


def emitted_requests(result: LazyOffloadDrain) -> list[str]:
    """The request id of every emitted item, in emission order."""
    return [item.request_id for item in result.items]


def counters(queue: EvictionAwareStoreQueue) -> LazyOffloadCounters:
    """The policy's cumulative counters."""
    return queue._counters


def pending_ops(queue: EvictionAwareStoreQueue) -> int:
    """Pending depth derived from the documented counter ledger."""
    ledger = counters(queue)
    return (
        ledger.admitted
        - ledger.emitted
        - ledger.dropped_evicted
        - ledger.dropped_on_request_drop
        - ledger.dropped_failed_store
        - ledger.dropped_id_reuse
    )


def make_queue(pool: FakeBlockPool) -> EvictionAwareStoreQueue:
    """Build a queue over ``pool``."""
    return EvictionAwareStoreQueue(cast("eviction_aware.BlockPool", pool))


class FakeClock:
    """Deterministic stand-in for the ``time`` module (monotonic only)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        """The current fake time."""
        return self.now


@pytest.fixture
def drain_budget_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap every drain at one emitted operation."""
    monkeypatch.setattr(eviction_aware, "MAX_DRAIN_PER_STEP", 1)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace the policy module's clock with a settable one."""
    fake = FakeClock()
    monkeypatch.setattr(eviction_aware, "time", fake)
    return fake


class TestAdmission:
    def test_admits_fully_hashed_op(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        assert queue.has_pending_request("req")
        assert counters(queue).admitted == 1
        assert pending_ops(queue) == 1

    def test_rejects_op_with_unhashed_block(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        meta = cast(
            "LMCacheMPRequestMetadata",
            FakeStoreMetadata("req", FakeStoreOp(0, 256)),
        )
        queue.add(meta, cast(BlockHashes, {1: pool.blocks[1].block_hash, 2: None}))
        assert not queue.has_pending_request("req")
        assert counters(queue).rejected_unhashed == 1
        assert counters(queue).admitted == 0

    def test_unhashed_rejection_breaks_prefix_chain(self) -> None:
        """The caller's tracker has already advanced past the skipped range, so a later
        op would be stored without its prefix -- it must be rejected.
        """
        pool = FakeBlockPool()
        pool.seed_held([2])
        queue = make_queue(pool)
        add_unhashed_op(queue, "req", end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        assert not queue.has_pending_request("req")
        assert counters(queue).rejected_prefix_broken == 1

    def test_ops_admitted_before_unhashed_rejection_stay_storable(self) -> None:
        """Only ops past the skipped range are unreachable; the prefix buffered before
        the rejection is intact and still emits.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_unhashed_op(queue, "req", end_tokens=512)
        result = drain(queue, next_step=1)
        assert emitted_ends(result) == [256]


class TestPressureTrigger:
    def test_idle_engine_never_drains(self) -> None:
        """Free-queue position alone is never a trigger; pressure is required."""
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        result = drain(queue)
        assert result.items == []
        assert pending_ops(queue) == 1

    def test_depth_is_not_carried_over_from_an_earlier_drain(self) -> None:
        """The danger depth is this drain's bounds alone: a burst on an earlier
        step leaves no depth behind once the bounds return to zero.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        drain(queue, next_step=4, admission=4)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue)
        assert result.items == []
        assert pending_ops(queue) == 1

    def test_pressure_drains_blocks_within_danger_depth(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 5, 6, 3, 4])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        add_op(queue, pool, "req", [3, 4], end_tokens=512)
        result = drain(queue, next_step=2)
        # Danger depth 2: the first op's blocks leave the queue as due, so the
        # allocation reaches the unbuffered blocks 5 and 6, and no further.
        assert emitted_ends(result) == [256]
        assert pending_ops(queue) == 1

    @pytest.mark.parametrize(
        "next_step,admission", [(3, 0), (0, 3)], ids=["running", "admission"]
    )
    def test_either_bound_alone_triggers_drain(
        self, next_step: int, admission: int
    ) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue, next_step=next_step, admission=admission)
        assert emitted_ends(result) == [256]

    def test_in_use_blocks_are_not_at_risk(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1, 2])  # hashed but not in the free queue
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        result = drain(queue, next_step=8, admission=8)
        assert result.items == []
        assert pending_ops(queue) == 1

    def test_finished_request_with_pending_ops_is_untouched(self) -> None:
        """Finishing ends prefix growth, not the eviction clock: pending ops of a
        finished request wait for their blocks to come due.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=2048)
        result = drain(queue, finished=frozenset({"req"}))
        assert result.items == []
        assert result.emptied_request_ids == []
        assert pending_ops(queue) == 1


class TestPrefixClosure:
    def test_due_later_op_flushes_earlier_ops_first(self) -> None:
        """A due op pulls its whole stored prefix out with it, in order."""
        pool = FakeBlockPool()
        pool.seed_held([1, 2])  # first op's blocks in use
        pool.seed_free([3, 4])  # second op's blocks at risk
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        add_op(queue, pool, "req", [3, 4], end_tokens=512)
        result = drain(queue, next_step=2)
        assert emitted_ends(result) == [256, 512]

    def test_eviction_drops_suffix_and_keeps_prefix(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 3])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        add_op(queue, pool, "req", [3], end_tokens=768)
        pool.evict(2)  # middle op's data is lost
        result = drain(queue)
        assert result.items == []
        assert result.emptied_request_ids == []
        assert counters(queue).dropped_evicted == 2
        assert pending_ops(queue) == 1  # the intact prefix stays pending

    def test_admission_rejected_after_prefix_break(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        pool.evict(1)
        drain(queue)
        add_op(queue, pool, "req", [2], end_tokens=512)
        assert not queue.has_pending_request("req")
        assert counters(queue).rejected_prefix_broken == 1


class TestStoreFailure:
    def test_store_failure_breaks_prefix_and_drops_held_back_ops(self) -> None:
        """A failed in-flight store leaves the request without its stored prefix: held-
        back operations must be dropped and later ops rejected, or they would be stored
        unreachable.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 3, 2])  # block 3 is unbuffered and ends the window
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        result = drain(queue, next_step=1)
        assert emitted_ends(result) == [256]  # first op in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        assert counters(queue).dropped_failed_store == 1

        add_op(queue, pool, "req", [2], end_tokens=512)
        assert counters(queue).rejected_prefix_broken == 1

    def test_failure_of_fresh_ops_after_reset_is_honored(self) -> None:
        """After drop_request the id starts a clean chain; a failure on the fresh
        generation breaks it as usual.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        assert len(drain(queue, next_step=2).items) == 1
        assert queue.drop_request("req") == 0

        pool.seed_free([2])
        pool.seed_held([3])
        add_op(queue, pool, "req", [2], end_tokens=256)
        add_op(queue, pool, "req", [3], end_tokens=512)
        result = drain(queue, next_step=2)
        assert emitted_ends(result) == [256]  # fresh op in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        add_op(queue, pool, "req", [3], end_tokens=768)
        assert counters(queue).rejected_prefix_broken == 1


class TestDrainOrderingAndBudget:
    def test_most_imminent_request_drains_first(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 3, 4])  # ranks 0..3
        queue = make_queue(pool)
        add_op(queue, pool, "req-late", [3, 4], end_tokens=256)
        add_op(queue, pool, "req-soon", [1, 2], end_tokens=256)
        result = drain(queue, next_step=4)
        assert emitted_requests(result) == ["req-soon", "req-late"]

    def test_equal_rank_preserves_request_admission_order(self) -> None:
        """The tie break under sustained pressure is admission order, not arbitrary
        request-id ordering.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req-z-first", [1], end_tokens=256)
        add_op(queue, pool, "req-a-second", [1, 2], end_tokens=256)
        result = drain(queue, next_step=1)
        assert emitted_requests(result) == ["req-z-first", "req-a-second"]

    @pytest.mark.usefixtures("drain_budget_of_one")
    def test_drain_budget_spreads_emission_over_steps(self) -> None:
        """The cap cuts a due segment from the tail, emitting only a front slice, and
        the remainder drains on a later step.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        first = drain(queue, next_step=2)
        assert emitted_ends(first) == [256]
        assert first.emptied_request_ids == []
        assert pending_ops(queue) == 1
        second = drain(queue, next_step=2)
        assert emitted_ends(second) == [512]
        assert second.emptied_request_ids == ["req"]


class TestEligibilityInputs:
    @pytest.mark.usefixtures("drain_budget_of_one")
    def test_blocked_request_is_held_until_the_manager_unblocks_it(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)

        assert len(emitted_ends(drain(queue, next_step=2))) == 1
        assert drain(queue, next_step=2, blocked=frozenset({"req"})).items == []
        assert len(emitted_ends(drain(queue, next_step=2))) == 1

    def test_blocked_request_still_drops_its_evicted_suffix(self) -> None:
        """A blocked request is not released, but its loss is still detected: the
        evicted ops are dropped while its batch is in flight.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        pool.evict(1)
        result = drain(queue, blocked=frozenset({"req"}))
        assert counters(queue).dropped_evicted == 1
        assert result.items == []
        assert result.emptied_request_ids == ["req"]
        assert not queue.has_pending_request("req")


class TestRequestLifecycleDiscards:
    """What the manager's lifecycle calls discard, and what they leave."""

    def test_drop_request_discards_ops_and_counts_them(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        assert queue.drop_request("req") == 1
        assert not queue.has_pending_request("req")
        assert counters(queue).dropped_on_request_drop == 1

    def test_drop_request_clears_prefix_state(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        add_unhashed_op(queue, "req", end_tokens=256)  # breaks the chain
        assert queue.drop_request("req") == 0
        add_op(queue, pool, "req", [1], end_tokens=256)
        assert queue.has_pending_request("req")

    def test_discard_for_reuse_counts_dropped_ops(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        queue.discard_for_reuse("req")
        assert not queue.has_pending_request("req")
        assert counters(queue).dropped_id_reuse == 1

    def test_discard_for_reuse_clears_prefix_state(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        queue.mark_store_failed("req")

        queue.discard_for_reuse("req")
        add_op(queue, pool, "req", [1], end_tokens=256)
        assert queue.has_pending_request("req")

    def test_release_request_clears_non_pending_prefix_state(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1])
        queue = make_queue(pool)
        queue.mark_store_failed("req")
        queue.release_request("req")

        add_op(queue, pool, "req", [1], end_tokens=256)
        assert queue.has_pending_request("req")


class TestDrainReporting:
    def test_emptied_request_ids_on_full_emission(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue, next_step=1)
        assert emitted_ends(result) == [256]
        assert result.emptied_request_ids == ["req"]
        assert not queue.has_pending_request("req")


class TestFreeQueueWalkBound:
    """The per-step free-queue read is bounded by the danger depth."""

    def test_idle_step_reads_no_free_queue_links(self) -> None:
        """No expected consumption means no rank can be below the danger depth, so the
        whole walk is dead work and must not run.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        assert drain(queue).items == []
        assert pool.link_reads == 0
        assert pending_ops(queue) == 1

    def test_idle_step_still_drops_ops_whose_blocks_were_evicted(self) -> None:
        """Skipping the walk must not skip the loss check: data loss is read from block
        hashes, not from ranks.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        pool.evict(1)
        result = drain(queue)
        assert counters(queue).dropped_evicted == 1
        assert result.emptied_request_ids == ["req"]
        assert pool.link_reads == 0

    def test_walk_stops_at_danger_depth(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool)
        add_op(queue, pool, "req", [90, 91], end_tokens=256)
        assert drain(queue, next_step=4).items == []
        # Danger depth 4: the walk reads the entry link and at most two links
        # per ranked block (look-ahead and advance) -- 2 * 4 + 2 -- and never
        # reaches the other 95 blocks.
        assert 0 < pool.link_reads <= 10


@contextmanager
def capture_log_lines() -> Iterator[list[str]]:
    """Collect the policy logger's INFO lines, without pytest's ``caplog``.

    ``init_logger`` sets ``propagate = False``, so these records never reach
    the root logger that ``caplog`` installs its handler on. ``caplog`` sees
    them only because pytest also walks the loggers that are already
    non-propagating when it starts capturing and attaches there too -- which
    pytest documents as best-effort, and which does not hold when the whole
    repo suite runs in one process. Attaching here depends on none of that,
    and on nothing an earlier test left behind on the logger.
    """
    lines: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    logger = eviction_aware.logger
    handler = _Collector(logging.INFO)
    previous_level, previously_disabled = logger.level, logger.disabled
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previously_disabled


class TestCounterLedger:
    """The ledger log line, which is the counters' only public surface."""

    def test_final_stats_report_the_whole_ledger(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "r1", [1], end_tokens=256)
        add_op(queue, pool, "r2", [2], end_tokens=256)
        drain(queue, next_step=1)

        with capture_log_lines() as lines:
            queue.log_final_stats()

        (line,) = [line for line in lines if "final counters:" in line]
        assert "admitted=2" in line
        assert f"emitted={counters(queue).emitted}" in line
        assert f"pending={pending_ops(queue)}" in line

    def test_periodic_ledger_lines_are_throttled(self, clock: FakeClock) -> None:
        """One line per interval, not one per drain: the drain runs on the scheduler's
        hot path.
        """
        pool = FakeBlockPool()
        pool.seed_held([1, 2, 3])
        queue = make_queue(pool)
        clock.now = 1000.0

        with capture_log_lines() as lines:

            def ledger_lines() -> list[str]:
                return [line for line in lines if "Lazy offload counters:" in line]

            add_op(queue, pool, "r1", [1], end_tokens=256)
            drain(queue, next_step=1)
            assert len(ledger_lines()) == 1

            # Same interval, new counters: the line is held back.
            add_op(queue, pool, "r2", [2], end_tokens=256)
            drain(queue, next_step=1)
            assert len(ledger_lines()) == 1

            clock.now += 2 * _STATS_LOG_INTERVAL_S
            add_op(queue, pool, "r3", [3], end_tokens=256)
            drain(queue, next_step=1)
            assert len(ledger_lines()) == 2
