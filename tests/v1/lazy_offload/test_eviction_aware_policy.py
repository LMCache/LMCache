# SPDX-License-Identifier: Apache-2.0
"""Drain behaviour of the eviction-aware lazy-offload policy.

Pure policy logic: no vLLM, no torch, no GPU. The block pool is the shared
``FakeBlockPool``, which offers the only two surfaces this policy reads: the
free queue as a linked list, where position is eviction rank, and each block's
current hash. Config validation lives in ``test_policy_selection.py``.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
    EvictionAwarePolicyConfig,
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
    new_blocks: int = 0,
    est_next: int = 0,
    finished: frozenset[str] = frozenset(),
    blocked: frozenset[str] = frozenset(),
) -> LazyOffloadDrain:
    """Run one drain step with the given signals."""
    return queue.drain(
        DrainSignals(
            new_blocks_allocated=new_blocks,
            est_next_step_blocks=est_next,
            finished_request_ids=set(finished),
            blocked_request_ids=set(blocked),
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


def make_queue(
    pool: FakeBlockPool,
    horizon_steps: float = 1.0,
    max_drain_per_step: int = 64,
    max_deferral_seconds: float = 0.0,
) -> EvictionAwareStoreQueue:
    """Build a queue over ``pool`` with test-friendly defaults."""
    config = EvictionAwarePolicyConfig(
        horizon_steps=horizon_steps,
        max_drain_per_step=max_drain_per_step,
        max_deferral_seconds=max_deferral_seconds,
    )
    return EvictionAwareStoreQueue(config, cast("eviction_aware.BlockPool", pool))


class FakeClock:
    """Deterministic stand-in for the ``time`` module (monotonic only)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        """The current fake time."""
        return self.now


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
        result = drain(queue, new_blocks=1)
        assert emitted_ends(result) == [256]


class TestPressureTrigger:
    def test_idle_engine_never_drains(self) -> None:
        """Free-queue position alone is never a trigger; pressure is required."""
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        result = drain(queue, new_blocks=0, est_next=0)
        assert result.items == []
        assert pending_ops(queue) == 1

    def test_depth_returns_to_zero_after_burst(self) -> None:
        """The EMA decays below half a block over the horizon and the depth goes back to
        zero rather than pinning a ceil'd 1 forever.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        drain(queue, new_blocks=4)
        for _ in range(10):
            drain(queue, new_blocks=0)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue, new_blocks=0)
        assert result.items == []
        assert pending_ops(queue) == 1

    def test_pressure_drains_blocks_within_danger_depth(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 3, 4])
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        add_op(queue, pool, "req", [3, 4], end_tokens=512)
        result = drain(queue, new_blocks=2)
        # danger depth 2: only blocks at ranks 0-1 (the first op) are at risk.
        assert emitted_ends(result) == [256]
        assert pending_ops(queue) == 1

    def test_feedforward_alone_triggers_drain(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue, new_blocks=0, est_next=3)
        assert emitted_ends(result) == [256]

    def test_in_use_blocks_are_not_at_risk(self) -> None:
        pool = FakeBlockPool()
        pool.seed_held([1, 2])  # hashed but not in the free queue
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        result = drain(queue, new_blocks=8, est_next=8)
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
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        add_op(queue, pool, "req", [3, 4], end_tokens=512)
        result = drain(queue, new_blocks=2)
        assert emitted_ends(result) == [256, 512]

    def test_eviction_drops_suffix_and_keeps_prefix(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 3])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        add_op(queue, pool, "req", [3], end_tokens=768)
        pool.evict(2)  # middle op's data is lost
        result = drain(queue, new_blocks=0)
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
        drain(queue, new_blocks=0)
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
        pool.seed_free([1, 2])
        queue = make_queue(pool, horizon_steps=2.0, max_drain_per_step=1)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        result = drain(queue, new_blocks=1)
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
        queue = make_queue(pool, horizon_steps=2.0)
        add_op(queue, pool, "req", [1], end_tokens=256)
        assert len(drain(queue, new_blocks=1).items) == 1
        assert queue.drop_request("req") == 0

        pool.seed_free([2])
        pool.seed_held([3])
        add_op(queue, pool, "req", [2], end_tokens=256)
        add_op(queue, pool, "req", [3], end_tokens=512)
        result = drain(queue, new_blocks=2)
        assert emitted_ends(result) == [256]  # fresh op in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        add_op(queue, pool, "req", [3], end_tokens=768)
        assert counters(queue).rejected_prefix_broken == 1


class TestDrainOrderingAndBudget:
    def test_most_imminent_request_drains_first(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2, 3, 4])  # ranks 0..3
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req-late", [3, 4], end_tokens=256)
        add_op(queue, pool, "req-soon", [1, 2], end_tokens=256)
        result = drain(queue, new_blocks=4)
        assert emitted_requests(result) == ["req-soon", "req-late"]

    def test_equal_rank_preserves_request_admission_order(self) -> None:
        """The tie break under sustained pressure is admission order, not arbitrary
        request-id ordering.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req-z-first", [1], end_tokens=256)
        add_op(queue, pool, "req-a-second", [1, 2], end_tokens=256)
        result = drain(queue, new_blocks=1)
        assert emitted_requests(result) == ["req-z-first", "req-a-second"]

    def test_drain_budget_spreads_emission_over_steps(self) -> None:
        """The cap cuts a due segment from the tail, emitting only a front slice, and
        the remainder drains on a later step.
        """
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)
        first = drain(queue, new_blocks=2)
        assert emitted_ends(first) == [256]
        assert first.emptied_request_ids == []
        assert pending_ops(queue) == 1
        second = drain(queue, new_blocks=2)
        assert emitted_ends(second) == [512]
        assert second.emptied_request_ids == ["req"]


class TestEligibilityInputs:
    def test_blocked_request_is_held_until_the_manager_unblocks_it(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        add_op(queue, pool, "req", [1], end_tokens=256)
        add_op(queue, pool, "req", [2], end_tokens=512)

        assert len(emitted_ends(drain(queue, new_blocks=2))) == 1
        assert drain(queue, new_blocks=2, blocked=frozenset({"req"})).items == []
        assert len(emitted_ends(drain(queue, new_blocks=2))) == 1

    def test_blocked_request_skips_validation_too(self) -> None:
        """A blocked request is skipped whole, validation included; the loss is only
        discovered once its receipt unblocks it.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool)
        add_op(queue, pool, "req", [1], end_tokens=256)
        pool.evict(1)
        result = drain(queue, blocked=frozenset({"req"}))
        assert counters(queue).dropped_evicted == 0
        assert queue.has_pending_request("req")
        assert result.emptied_request_ids == []

        result = drain(queue)
        assert counters(queue).dropped_evicted == 1
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
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1], end_tokens=256)
        result = drain(queue, new_blocks=1)
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
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1, 2], end_tokens=256)
        assert drain(queue, new_blocks=0, est_next=0).items == []
        assert pool.link_reads == 0
        assert pending_ops(queue) == 1

    def test_idle_step_still_drops_ops_whose_blocks_were_evicted(self) -> None:
        """Skipping the walk must not skip the loss check: data loss is read from block
        hashes, not from ranks.
        """
        pool = FakeBlockPool()
        pool.seed_free([1])
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [1], end_tokens=256)
        pool.evict(1)
        result = drain(queue, new_blocks=0, est_next=0)
        assert counters(queue).dropped_evicted == 1
        assert result.emptied_request_ids == ["req"]
        assert pool.link_reads == 0

    def test_walk_stops_at_danger_depth(self) -> None:
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool, horizon_steps=1.0)
        add_op(queue, pool, "req", [90, 91], end_tokens=256)
        assert drain(queue, new_blocks=0, est_next=4).items == []
        # Danger depth 4: the walk reads the entry link and at most two links
        # per ranked block (look-ahead and advance) -- 2 * 4 + 2 -- and never
        # reaches the other 95 blocks.
        assert 0 < pool.link_reads <= 10


class TestConfigKeys:
    """The ``lmcache.mp.lazy_offload_*`` tunables, as a user sets them."""

    @pytest.mark.parametrize(
        "horizon,cap,deferral",
        [(4.0, 8, 1.5), ("4", "8", "1.5")],
        ids=["native", "strings-from-json"],
    )
    def test_tunables_are_read_from_their_documented_keys(
        self, horizon: str | float, cap: str | int, deferral: str | float
    ) -> None:
        config = EvictionAwarePolicyConfig.from_configs(
            {
                "lmcache.mp.lazy_offload_horizon_steps": horizon,
                "lmcache.mp.lazy_offload_max_drain_per_step": cap,
                "lmcache.mp.lazy_offload_max_deferral_seconds": deferral,
            }
        )

        assert config == EvictionAwarePolicyConfig(
            horizon_steps=4.0, max_drain_per_step=8, max_deferral_seconds=1.5
        )

    def test_an_unset_key_keeps_the_field_default(self) -> None:
        assert EvictionAwarePolicyConfig.from_configs({}) == (
            EvictionAwarePolicyConfig()
        )

    def test_an_out_of_range_tunable_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_steps"):
            EvictionAwarePolicyConfig.from_configs(
                {"lmcache.mp.lazy_offload_horizon_steps": "0"}
            )


class TestCounterLedger:
    """The ledger log line, which is the counters' only public surface."""

    def test_final_stats_report_the_whole_ledger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = FakeBlockPool()
        pool.seed_free([1, 2])
        queue = make_queue(pool)
        add_op(queue, pool, "r1", [1], end_tokens=256)
        add_op(queue, pool, "r2", [2], end_tokens=256)
        drain(queue, new_blocks=1)

        with caplog.at_level("INFO", logger=eviction_aware.logger.name):
            queue.log_final_stats()

        (line,) = [
            r.getMessage()
            for r in caplog.records
            if "final counters:" in r.getMessage()
        ]
        assert "admitted=2" in line
        assert f"emitted={counters(queue).emitted}" in line
        assert f"pending={pending_ops(queue)}" in line

    def test_periodic_ledger_lines_are_throttled(
        self, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One line per interval, not one per drain: the drain runs on the scheduler's
        hot path.
        """
        pool = FakeBlockPool()
        pool.seed_held([1, 2, 3])
        queue = make_queue(pool)
        clock.now = 1000.0

        def ledger_lines() -> list[str]:
            return [
                r.getMessage()
                for r in caplog.records
                if "Lazy offload counters:" in r.getMessage()
            ]

        with caplog.at_level("INFO", logger=eviction_aware.logger.name):
            add_op(queue, pool, "r1", [1], end_tokens=256)
            drain(queue, new_blocks=1)
            assert len(ledger_lines()) == 1

            # Same interval, new counters: the line is held back.
            add_op(queue, pool, "r2", [2], end_tokens=256)
            drain(queue, new_blocks=1)
            assert len(ledger_lines()) == 1

            clock.now += 2 * _STATS_LOG_INTERVAL_S
            add_op(queue, pool, "r3", [3], end_tokens=256)
            drain(queue, new_blocks=1)
            assert len(ledger_lines()) == 2


class TestDeferralDeadline:
    """The wall-clock bound on how long an operation may wait."""

    def test_disabled_by_default_leaves_a_far_block_pending(
        self, clock: FakeClock
    ) -> None:
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool, horizon_steps=1.0)
        clock.now = 100.0
        drain(queue, new_blocks=1, est_next=1)
        add_op(queue, pool, "r1", [99], end_tokens=256)
        clock.now = 10_100.0
        result = drain(queue, new_blocks=1, est_next=1)
        assert result.items == []
        assert counters(queue).emitted_overdue == 0

    def test_emits_when_the_bound_is_passed(self, clock: FakeClock) -> None:
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue, new_blocks=1, est_next=1)
        add_op(queue, pool, "r1", [99], end_tokens=256)
        # Deep in the free queue: the danger window cannot make it due.
        clock.now = 125.0
        assert drain(queue, new_blocks=1, est_next=1).items == []
        clock.now = 131.0
        result = drain(queue, new_blocks=1, est_next=1)
        assert emitted_requests(result) == ["r1"]
        assert counters(queue).emitted_overdue == 1
        assert counters(queue).emitted == 1

    def test_window_emission_is_not_counted_as_overdue(self, clock: FakeClock) -> None:
        pool = FakeBlockPool()
        pool.seed_free([7])
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue, new_blocks=4, est_next=4)
        add_op(queue, pool, "r1", [7], end_tokens=256)
        clock.now = 101.0
        result = drain(queue, new_blocks=4, est_next=4)
        assert emitted_requests(result) == ["r1"]
        assert counters(queue).emitted_overdue == 0

    def test_overdue_emits_with_a_zero_danger_depth(self, clock: FakeClock) -> None:
        """An idle engine never opens the window; the deadline still fires."""
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue)
        add_op(queue, pool, "r1", [99], end_tokens=256)
        clock.now = 200.0
        result = drain(queue)
        assert emitted_requests(result) == ["r1"]
        assert counters(queue).emitted_overdue == 1
        assert pool.link_reads == 0

    def test_overdue_releases_the_whole_surviving_front(self, clock: FakeClock) -> None:
        """Past the deadline the whole surviving front is due, wherever its blocks sit
        -- minus any evicted suffix, which is dropped first.
        """
        pool = FakeBlockPool()
        pool.seed_free([97, 98, 99])
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue)
        add_op(queue, pool, "r1", [97], end_tokens=256)
        add_op(queue, pool, "r1", [98], end_tokens=512)
        add_op(queue, pool, "r1", [99], end_tokens=768)
        pool.evict(99)
        clock.now = 200.0
        result = drain(queue)
        assert emitted_ends(result) == [256, 512]
        assert counters(queue).emitted_overdue == 2
        assert counters(queue).dropped_evicted == 1
        assert result.emptied_request_ids == ["r1"]

    def test_a_block_at_risk_outranks_a_passed_deadline(self, clock: FakeClock) -> None:
        """Missing a deadline costs latency; missing an eviction costs the data."""
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(
            pool,
            horizon_steps=1.0,
            max_drain_per_step=1,
            max_deferral_seconds=30.0,
        )
        clock.now = 100.0
        drain(queue, new_blocks=1, est_next=1)
        add_op(queue, pool, "overdue", [99], end_tokens=256)
        clock.now = 200.0  # past the deadline for "overdue" only
        add_op(queue, pool, "at-risk", [1], end_tokens=256)

        result = drain(queue, new_blocks=1, est_next=1)

        assert emitted_requests(result) == ["at-risk"]
        assert queue.has_pending_request("overdue")

    def test_a_request_past_both_emits_only_its_due_front(
        self, clock: FakeClock
    ) -> None:
        """Holding a block in the window sizes the release too: the request emits its
        due front segment, and the ops behind it keep their own admission clocks instead
        of being dumped with it.
        """
        pool = FakeBlockPool()
        pool.seed_free(list(range(1, 101)))
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue, new_blocks=1, est_next=1)
        add_op(queue, pool, "r1", [1], end_tokens=256)  # at the queue head
        add_op(queue, pool, "r1", [99], end_tokens=512)  # deep in the queue
        clock.now = 200.0  # both ops are past the deadline

        result = drain(queue, new_blocks=1, est_next=1)

        assert emitted_ends(result) == [256]
        assert counters(queue).emitted_overdue == 0

    def test_overdue_respects_the_drain_budget(self, clock: FakeClock) -> None:
        pool = FakeBlockPool()
        pool.seed_free([97, 98])
        queue = make_queue(
            pool, horizon_steps=1.0, max_deferral_seconds=30.0, max_drain_per_step=1
        )
        clock.now = 100.0
        drain(queue)
        add_op(queue, pool, "r1", [97], end_tokens=256)
        add_op(queue, pool, "r1", [98], end_tokens=512)
        clock.now = 200.0
        first = drain(queue)
        assert emitted_ends(first) == [256]
        assert pending_ops(queue) == 1
        clock.now = 201.0
        second = drain(queue)
        assert emitted_ends(second) == [512]

    def test_overdue_still_drops_an_evicted_chain(self, clock: FakeClock) -> None:
        """The deadline releases survivors, never data the pool has lost."""
        pool = FakeBlockPool()
        pool.seed_free([97, 98])
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue)
        add_op(queue, pool, "r1", [97], end_tokens=256)
        add_op(queue, pool, "r1", [98], end_tokens=512)
        pool.evict(97)
        clock.now = 200.0
        result = drain(queue)
        assert result.items == []
        assert counters(queue).dropped_evicted == 2
        assert result.emptied_request_ids == ["r1"]

    def test_deadline_measures_the_front_op_not_the_request(
        self, clock: FakeClock
    ) -> None:
        """A request keeps its urgency from its oldest surviving op."""
        pool = FakeBlockPool()
        pool.seed_free([98, 99])
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        clock.now = 100.0
        drain(queue)
        add_op(queue, pool, "r1", [99], end_tokens=256)
        clock.now = 200.0
        assert len(emitted_ends(drain(queue))) == 1
        # Fresh op on the same request: the deadline restarts from its own
        # admission, so the next drain leaves it pending.
        clock.now = 201.0
        drain(queue)
        add_op(queue, pool, "r1", [98], end_tokens=512)
        clock.now = 210.0
        assert drain(queue).items == []
        assert pending_ops(queue) == 1
