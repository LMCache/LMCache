# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the eviction-aware lazy offload policy.

Pure policy tests: no vLLM, no torch, no GPU. The block pool is faked
through the ``BlockPoolReader`` protocol.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, cast

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    AdmitResult,
    EvictionAwareStoreQueue,
    LazyOffloadPolicyConfig,
    PendingStoreOp,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


class FakePoolView:
    """In-memory BlockPoolReader: a free queue (head first) and a hash map.

    ``free_queue_block_ids`` is a generator that counts the blocks the
    policy actually consumes into ``blocks_walked``. The production view
    walks a linked list on the scheduler's critical path, so a fake that
    handed over the whole queue at once would hide how deep a step reads --
    the quantity every token's decode latency pays for.
    """

    def __init__(self) -> None:
        self.free_queue: list[int] = []
        self.hashes: dict[int, bytes] = {}
        self.blocks_walked = 0
        self.hash_requests: list[int] = []

    def free_queue_block_ids(self) -> Iterator[int]:
        for block_id in self.free_queue:
            self.blocks_walked += 1
            yield block_id

    def is_free(self, block_id: int) -> bool:
        return block_id in self.free_queue

    def block_hash(self, block_id: int) -> bytes | None:
        self.hash_requests.append(block_id)
        return self.hashes.get(block_id)

    def evict(self, block_id: int) -> None:
        """Simulate eviction + reallocation: hash reset, out of the queue."""
        self.free_queue.remove(block_id)
        del self.hashes[block_id]


@dataclass
class FakeStoreMetadata:
    """Opaque payload standing in for LMCacheMPRequestMetadata."""

    label: str


def make_op(
    request_id: str,
    block_ids: list[int],
    pool: FakePoolView,
    prefix_end_tokens: int,
    cache_salt: str = "",
    prefix_start_tokens: int = -1,
    epoch: int = 0,
) -> PendingStoreOp:
    """Build a pending op whose hash snapshot matches the pool's state.

    ``prefix_start_tokens`` defaults to one 256-token chunk before the end,
    so consecutive ops built with 256-spaced ends form a contiguous chain;
    pass it explicitly to model a deduplication hole.
    """
    if prefix_start_tokens < 0:
        prefix_start_tokens = max(0, prefix_end_tokens - 256)
    return PendingStoreOp(
        request_id=request_id,
        store_metadata=cast(
            "LMCacheMPRequestMetadata",
            FakeStoreMetadata(label=f"{request_id}:{prefix_end_tokens}"),
        ),
        block_hashes={block_id: pool.hashes[block_id] for block_id in block_ids},
        prefix_start_tokens=prefix_start_tokens,
        prefix_end_tokens=prefix_end_tokens,
        epoch=epoch,
        cache_salt=cache_salt,
    )


def seed_blocks(pool: FakePoolView, block_ids: list[int], free: bool) -> None:
    """Give each block a distinct hash; optionally append to the free queue."""
    for block_id in block_ids:
        pool.hashes[block_id] = f"hash-{block_id}".encode()
        if free:
            pool.free_queue.append(block_id)


def make_queue(
    pool: FakePoolView,
    horizon_steps: float = 1.0,
    min_prefix_tokens: int = 0,
    max_drain_per_step: int = 64,
) -> EvictionAwareStoreQueue:
    config = LazyOffloadPolicyConfig(
        horizon_steps=horizon_steps,
        min_prefix_tokens=min_prefix_tokens,
        max_drain_per_step=max_drain_per_step,
    )
    return EvictionAwareStoreQueue(config, pool)


class TestConfigValidation:
    def test_default_horizon_uses_calibrated_value(self) -> None:
        assert LazyOffloadPolicyConfig().horizon_steps == 2.5

    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(horizon_steps=0)

    def test_rejects_negative_min_prefix(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(min_prefix_tokens=-1)

    def test_rejects_zero_drain_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_drain_per_step=0)


class TestAdmission:
    def test_admits_fully_hashed_op(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        result = queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 1

    def test_rejects_mixed_epochs_for_one_request(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        assert (
            queue.admit(make_op("req", [1], pool, 256, epoch=3)) is AdmitResult.ADMITTED
        )

        with pytest.raises(RuntimeError, match="mixed store epochs 3 and 4"):
            queue.admit(make_op("req", [2], pool, 512, epoch=4))

    def test_rejects_op_with_unhashed_block(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        op = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={1: pool.hashes[1], 2: None},  # type: ignore[dict-item]
            prefix_start_tokens=0,
            prefix_end_tokens=256,
        )
        assert queue.admit(op) is AdmitResult.REJECTED_UNHASHED_BLOCK
        assert queue.num_pending_ops() == 0
        assert queue.stats().rejected_unhashed == 1

    def test_unhashed_rejection_breaks_prefix_chain(self) -> None:
        """The caller's tracker has already advanced past the skipped range,
        so a later chunk would be stored without its prefix (the retrieval
        prefix lookup stops at the hole) -- it must be rejected."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        unhashed = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={1: None},  # type: ignore[dict-item]
            prefix_start_tokens=0,
            prefix_end_tokens=256,
        )
        assert queue.admit(unhashed) is AdmitResult.REJECTED_UNHASHED_BLOCK
        later = make_op("req", [2], pool, prefix_end_tokens=512)
        assert queue.admit(later) is AdmitResult.REJECTED_PREFIX_BROKEN
        assert queue.num_pending_ops() == 0

    def test_chunks_admitted_before_unhashed_rejection_stay_storable(self) -> None:
        """Only chunks past the skipped range are unreachable; the prefix
        buffered before the rejection is intact and still emits."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        seed_blocks(pool, [2], free=False)
        queue = make_queue(pool)
        first = make_op("req", [1], pool, prefix_end_tokens=256)
        assert queue.admit(first) is AdmitResult.ADMITTED
        unhashed = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={2: None},  # type: ignore[dict-item]
            prefix_start_tokens=256,
            prefix_end_tokens=512,
        )
        assert queue.admit(unhashed) is AdmitResult.REJECTED_UNHASHED_BLOCK
        assert queue.num_pending_ops() == 1
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256]


class TestPressureTrigger:
    def test_idle_engine_never_drains(self) -> None:
        """Free-queue position alone is never a trigger; pressure is required."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.num_pending_ops() == 1

    def test_depth_returns_to_zero_after_burst(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        for _ in range(10):
            queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []

    def test_pressure_drains_blocks_within_danger_depth(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3, 4], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        # danger depth 2: only blocks at ranks 0-1 (the first op) are at risk.
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_feedforward_alone_triggers_drain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=3)
        assert len(queue.collect_due().to_store) == 1

    def test_in_use_blocks_are_not_at_risk(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # hashed but not in free queue
        queue = make_queue(pool)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=8)
        assert queue.collect_due().to_store == []
        assert queue.num_pending_ops() == 1


class TestPrefixClosure:
    def test_due_later_op_flushes_earlier_ops_first(self) -> None:
        """A due chunk pulls its whole stored prefix out with it, in order."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # first chunk's blocks in use
        seed_blocks(pool, [3, 4], free=True)  # second chunk's blocks at risk
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3, 4], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256, 512]

    def test_eviction_drops_suffix_and_keeps_prefix(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=768))
        pool.evict(2)  # middle chunk's data is lost
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.dropped_evicted] == [512, 768]
        assert queue.num_pending_ops() == 1  # the intact prefix stays pending
        assert queue.stats().dropped_evicted == 2

    def test_admission_rejected_after_prefix_break(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        queue.collect_due()
        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN


class TestEconomyGate:
    def test_short_prefix_dropped_not_stored(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert len(result.dropped_short_prefix) == 2
        assert queue.num_pending_ops() == 0
        assert queue.stats().rejected_short_prefix == 2

    def test_long_prefix_passes_gate(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=2048))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1

    def test_gate_disabled_by_default(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=16))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1


class TestStoreFailure:
    def test_store_failure_breaks_prefix_and_drops_held_back_ops(self) -> None:
        """A failed in-flight store leaves the request without its stored
        prefix: held-back operations must be dropped and later chunks
        rejected, or they would be stored unreachable."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=2.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1  # first op in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        assert queue.stats().dropped_failed_store == 1

        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN

    def test_failure_of_fresh_batch_after_reset_is_honored(self) -> None:
        """A current-epoch failure after reset breaks the chain as usual."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=2.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        queue.drop_request("req")

        seed_blocks(pool, [2], free=True)
        seed_blocks(pool, [3], free=False)
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=512))
        assert len(queue.collect_due().to_store) == 1  # fresh batch in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        result = queue.admit(make_op("req", [3], pool, prefix_end_tokens=768))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN


class TestContentDeduplication:
    """One pending op per unique content: requests sharing a hot prefix
    must not each buffer their own copy (the unbounded-growth case), and
    a deduplicated request must not defer its session teardown."""

    def test_identical_content_from_other_request_is_deduplicated(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        assert (
            queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        assert (
            queue.admit(make_op("req-b", [1, 2], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )
        assert queue.num_pending_ops() == 1
        assert queue.stats().deduplicated == 1
        assert not queue.has_pending_request("req-b")

    def test_hot_prefix_requests_keep_one_pending_op(self) -> None:
        """The round-2 repro: a hot shared prefix must not grow the queue
        by one op per request."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # hot: never in the free queue
        queue = make_queue(pool)
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=8)
        for i in range(100):
            request_id = f"req-{i}"
            queue.admit(make_op(request_id, [1, 2], pool, prefix_end_tokens=256))
            queue.collect_due()
            assert queue.has_pending_request(request_id) is (i == 0)
        assert queue.num_pending_ops() == 1

    def test_different_salt_is_not_deduplicated(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
        result = queue.admit(
            make_op("req-b", [1, 2], pool, prefix_end_tokens=256, cache_salt="s")
        )
        assert result is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 2

    def test_content_admittable_again_after_emission(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED

    def test_content_admittable_again_after_eviction_drop(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        op_b = make_op("req-b", [1], pool, prefix_end_tokens=256)
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_evicted) == 1
        # req-b snapshotted before the eviction; only req-a's chain broke.
        assert queue.admit(op_b) is AdmitResult.ADMITTED

    def test_content_admittable_again_after_short_prefix_drop(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_short_prefix) == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED

    def test_content_admittable_again_after_drop_request(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        assert queue.drop_request("req-a") == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED

    def test_cover_with_recycled_blocks_does_not_absorb_live_copy(self) -> None:
        """A dedup hit must verify the covering op's snapshot is still live.
        The covering op can already be a corpse while it waits in the pending
        list (its blocks recycled by this step's allocation, or its cleanup
        skipped while its request holds an in-flight batch); deduplicating
        against it would discard the only live copy of the content."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        corpse = make_op("req-a", [1], pool, prefix_end_tokens=256)
        assert queue.admit(corpse) is AdmitResult.ADMITTED
        # Block 1 evicted; req-b recomputed the same content into block 2
        # (block hashes are content-derived, so the chains are equal).
        content_hash = pool.hashes[1]
        pool.evict(1)
        pool.hashes[2] = content_hash
        live = make_op("req-b", [2], pool, prefix_end_tokens=256)
        assert queue.admit(live) is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 2

    def test_content_key_follows_live_copy_after_corpse_drop(self) -> None:
        """Dropping the corpse must not release the live copy's content key:
        a third identical admission still deduplicates against the live op."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        content_hash = pool.hashes[1]
        pool.evict(1)
        pool.hashes[2] = content_hash
        assert (
            queue.admit(make_op("req-b", [2], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.dropped_evicted) == 1  # req-a's corpse
        assert (
            queue.admit(make_op("req-c", [2], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )
        assert queue.num_pending_ops() == 1

    def test_cover_with_corpse_earlier_sibling_does_not_absorb_live_copy(self) -> None:
        """The dedup liveness check must cover the whole prefix chain, not
        just the covering op: a cover whose earlier sibling is already a
        corpse is deterministically dropped by prefix closure on the next
        drain, so it must not absorb a live copy either. Requires the front
        block to die before the tail block, which hybrid/sliding-window
        block freeing can produce."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a", [2], pool, prefix_end_tokens=512))
        # Front block recycled (corpse), tail block intact. req-b recomputed
        # the same content into fresh blocks (hashes are content-derived,
        # so the chains are equal).
        front_hash, tail_hash = pool.hashes[1], pool.hashes[2]
        pool.evict(1)
        pool.hashes[11] = front_hash
        pool.hashes[12] = tail_hash
        assert (
            queue.admit(make_op("req-b", [11], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        assert (
            queue.admit(make_op("req-b", [12], pool, prefix_end_tokens=512))
            is AdmitResult.ADMITTED
        )
        # req-a's doomed chain is swept; req-b now owns both content keys,
        # so a third identical request deduplicates against the live copy.
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_evicted) == 2
        assert (
            queue.admit(make_op("req-c", [12], pool, prefix_end_tokens=512))
            is AdmitResult.DEDUPLICATED
        )

    def test_cover_with_corpse_later_sibling_still_absorbs_duplicates(self) -> None:
        """Only corpses before the cover doom it: a later sibling's loss
        prefix-closes from its own position, leaving the cover storable, so
        the cover remains a valid deduplication target."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a", [2], pool, prefix_end_tokens=512))
        front_hash = pool.hashes[1]
        pool.evict(2)  # tail corpse; the front op survives prefix closure
        pool.hashes[11] = front_hash
        assert (
            queue.admit(make_op("req-b", [11], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )


class TestEmissionContiguity:
    """An emitted batch is coalesced into one contiguous store operation, so
    emission must never span a deduplication hole in the pending list."""

    def _queue_with_hole(self) -> "tuple[FakePoolView, EvictionAwareStoreQueue]":
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        seed_blocks(pool, [3], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        # [256, 512) was deduplicated under another request: the pending
        # list has a hole between the two admitted ops.
        queue.admit(
            make_op("req", [3], pool, prefix_end_tokens=768, prefix_start_tokens=512)
        )
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=4)
        return pool, queue

    def test_emission_stops_at_dedup_hole(self) -> None:
        _, queue = self._queue_with_hole()
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_post_hole_op_emitted_in_next_batch_after_receipt(self) -> None:
        _, queue = self._queue_with_hole()
        queue.collect_due()
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [768]
        assert queue.num_pending_ops() == 0


class TestDrainOrderingAndCap:
    def test_most_imminent_request_drains_first(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)  # ranks 0..3
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-late", [3, 4], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-soon", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-soon", "req-late"]

    def test_equal_rank_preserves_request_admission_order(self) -> None:
        """Incremental set discovery must not change the historical tie break.

        This matters under sustained pressure: arbitrary request-id ordering
        changes which shared hot prefixes remain pending long enough to dedup.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-z-first", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a-second", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids=set(),
        )
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == [
            "req-z-first",
            "req-a-second",
        ]

    def test_drain_cap_cuts_from_the_tail(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        first = queue.collect_due()
        assert [op.prefix_end_tokens for op in first.to_store] == [256]
        assert queue.num_pending_ops() == 1
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        second = queue.collect_due()
        assert [op.prefix_end_tokens for op in second.to_store] == [512]

    def test_cap_reports_what_it_held_back(self) -> None:
        """The sizing sensor: a drain the cap cut reports the ops it did not
        emit, and counts itself once regardless of how many it held."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        for index, block in enumerate([1, 2, 3]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=3, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.to_store) == 1
        assert result.ops_held_back == 2
        assert queue.stats().throttled_drains == 1

    def test_uncapped_drain_holds_nothing_back(self) -> None:
        """The sensor must stay silent on the default cap, or it would read
        as a misconfiguration on every healthy deployment."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.to_store) == 2
        assert result.ops_held_back == 0
        assert queue.stats().throttled_drains == 0


class TestPinCascadeShift:
    """Emitting a segment pins its blocks out of the free queue, shifting
    every block behind them toward the head before the next allocation runs.
    collect_due extends the due threshold by the blocks already emitted in
    the same call so shifted candidates drain now instead of losing the race.
    """

    def test_emission_shift_pulls_next_candidate_into_the_window(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=True)  # ranks 0..4
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4, 5], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1: req-a is due (rank 0). Pinning its 3 blocks
        # will move req-b (min rank 3) to rank 0 before the next step's
        # allocation, so req-b must drain in the same call.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]

    def test_in_use_blocks_do_not_expand_the_shift(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        seed_blocks(pool, [9], free=False)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 9], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-a pins two blocks, but only block 1 leaves the free queue.
        # Block 3 moves from rank 2 to rank 1, exactly outside depth 1.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1

    def test_shared_blocks_expand_the_shift_only_once(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-c", [5], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-a removes blocks 1 and 2; req-b then removes only block 3,
        # because shared block 2 is already pinned. Block 5 moves from rank
        # 4 to rank 1, which is exactly outside depth 1.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]
        assert queue.num_pending_ops() == 1

    def test_shift_never_opens_the_gate_by_itself(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1 but no candidate reaches it (min ranks 1 and 3):
        # with no first emission there is no shift, and nothing drains.
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.num_pending_ops() == 2

    def test_candidate_beyond_the_shifted_window_stays_pending(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3, 4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1, req-a due at rank 0 and pins one block; req-b's
        # min rank 2 is exactly at the shifted threshold (1 + 1), not below.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1

    def test_drain_cap_stops_the_cascade(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req-a", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-b sits inside the shifted window (3 < 1 + 3) but the per-step
        # cap is exhausted by req-a's op, so req-b waits for the next step.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1


class TestControllerEligibilityInputs:
    def test_blocked_request_is_held_until_controller_unblocks_it(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)

        assert len(queue.collect_due().to_store) == 1
        assert queue.collect_due({"req"}).to_store == []
        assert len(queue.collect_due().to_store) == 1

    def test_discard_for_reuse_clears_buffer_and_prefix_state(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.mark_store_failed("req")

        assert queue.discard_for_reuse("req") == 0
        assert (
            queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )

    def test_release_request_clears_non_pending_prefix_state(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.mark_store_failed("req")
        queue.release_request("req")

        assert (
            queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )


class TestFreeQueueSnapshotBound:
    """The per-step free-queue read is bounded, and the bound decides nothing.

    ``collect_due`` runs once per scheduler step on the critical path, so
    what it reads is paid by every request's TTFT and every token's TPOT.
    Reading the whole free queue is O(free blocks) -- tens of thousands on a
    pool with room to spare -- while the only ranks any decision compares
    are those within the danger depth, extended by the blocks this same call
    *does* pin out of the queue. Everything deeper is indistinguishable from
    a block that is not in the queue at all, which the policy already treats
    as not at risk.

    "Does", not "can": the depth a full-budget drain could reach is not what
    a step should pay for, because a step almost never drains a full budget.
    The read therefore follows the emissions rather than anticipating them,
    which leaves ``max_drain_per_step`` bounding the D2H burst and nothing
    else.
    """

    def test_snapshot_stops_at_danger_depth_plus_pending_blocks(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 101)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [90, 91], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=4)
        assert queue.collect_due().to_store == []
        # 4 blocks per step over a 1-step horizon, plus the 2 pending blocks
        # that an emission in this call could shift the queue by.
        assert pool.blocks_walked == 4

    def test_incremental_step_checks_only_bounded_candidate_requests(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 1001)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        for block_id in range(1, 1001):
            queue.admit(
                make_op(
                    f"req-{block_id}",
                    [block_id],
                    pool,
                    prefix_end_tokens=256,
                )
            )
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids=set(),
        )
        queue.collect_due()

        # danger depth 1 plus at most 64 one-block emissions: neither the
        # free-list walk nor hash validation reaches the other 935 requests.
        assert pool.blocks_walked == 64
        assert set(pool.hash_requests) <= set(range(1, 66))

    def test_allocated_block_signal_revalidates_a_nonfree_op(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=256))
        pool.evict(3)
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids={3},
        )
        result = queue.collect_due()

        assert [op.request_id for op in result.dropped_evicted] == ["req"]
        assert queue.num_pending_ops() == 0

    def test_idle_step_reads_no_ranks_at_all(self) -> None:
        """No expected consumption means no rank can be below the danger
        depth, so the whole snapshot is dead work."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []
        assert pool.blocks_walked == 0
        assert queue.num_pending_ops() == 1

    def test_idle_step_still_drops_ops_whose_blocks_were_evicted(self) -> None:
        """Skipping the snapshot must not skip the loss check: whether an
        op's data is still there is read from block hashes, not from ranks,
        and a request that lost its blocks while the engine idled has to be
        dropped on that step like any other."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.dropped_evicted) == 1
        assert queue.stats().dropped_evicted == 1
        assert pool.blocks_walked == 0

    def test_bound_covers_the_candidate_that_only_an_emission_shifts_into_reach(
        self,
    ) -> None:
        """The bound has to include the shift the call itself causes.

        `b` sits one rank past the danger depth, so it is not due on the
        depth alone; emitting `a` pins one block out of the queue ahead of
        it, which brings it inside. A snapshot cut at the danger depth would
        have shown `b`'s block as absent -- read as "not in the free queue,
        not at risk" -- and lost it to the next allocation.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=2)
        result = queue.collect_due()
        assert {op.request_id for op in result.to_store} == {"a", "b"}
        # Danger depth 2, widened by the one block emitting `a` pinned.
        assert pool.blocks_walked == 3

    def test_read_depth_does_not_scale_with_the_drain_budget(self) -> None:
        """The budget bounds the emissions, not the read.

        Both settings face the same queue, the same backlog and the same
        danger depth, and neither has anything due; the ranks either one
        compares are the same ranks, so the walk has to be the same length.
        """
        walked = []
        for budget in (1, 64):
            pool = FakePoolView()
            seed_blocks(pool, list(range(1, 501)), free=True)
            queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=budget)
            for block_id in range(400, 500):
                queue.admit(
                    make_op(f"req-{block_id}", [block_id], pool, prefix_end_tokens=256)
                )
            queue.observe_step(new_blocks_allocated=3, est_next_step_blocks=0)
            assert queue.collect_due().to_store == []
            walked.append(pool.blocks_walked)
        assert walked == [3, 3]

    def test_read_depth_follows_the_shift_an_emission_causes(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 501)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=64)
        queue.admit(make_op("req", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        # Danger depth 1 plus the three blocks the emission actually pinned
        # out of the queue -- not 64 times the largest pending operation.
        assert pool.blocks_walked == 4

    def test_a_pin_deeper_than_the_window_still_counts_as_a_shift(self) -> None:
        """Queue membership is read from the pool, not from the window.

        `req-a` is due on its head block, but pinning it removes its deep
        block from the queue as well, and both moves `req-b` toward the
        head. Counting only the pins the window happened to cover would
        widen the window by one instead of two, leave `req-b`'s block
        unread, and lose it to the next allocation -- and the deeper the
        pin, the less likely the window is to have covered it.
        """
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 21)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 15], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]
        # Danger depth 1, widened to 3 by `req-a`'s two pins and to 4 by
        # `req-b`'s one -- the walk stops where the emissions stop.
        assert pool.blocks_walked == 4

    def test_counters_report_what_each_step_read_and_validated(self) -> None:
        """The decision loop's own cost is observable, not inferred.

        Nothing here is due on either step, which is the case that has to be
        cheap: the backlog sits far from the eviction head and the step
        still pays a walk and a validation pass for it.
        """
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 101)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [90, 91], pool, prefix_end_tokens=256))
        for _ in range(2):
            queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
            assert queue.collect_due().to_store == []
        stats = queue.stats()
        assert stats.drain_steps == 2
        assert stats.free_queue_blocks_read == 4
        assert stats.requests_validated == 2
        assert stats.blocks_validated == 4
