# SPDX-License-Identifier: Apache-2.0
"""Drain behaviour of the FIFO lazy-offload policy.

FIFO reads no GPU state: it buffers by request and releases whole requests in
admission order once enough eligible ones accumulate. Everything goes through
the ``OffloadPolicy`` interface, so the store metadata is an opaque mock.
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy

THRESHOLD_KEY = "lmcache.mp.lazy_offload_threshold"
SELECT_COUNT_KEY = "lmcache.mp.lazy_offload_select_count"


def make_meta(request_id: str = "req-0") -> MagicMock:
    """Build a mock ``LMCacheMPRequestMetadata`` for one store operation."""
    meta = MagicMock()
    meta.request_id = request_id
    return meta


def make_block_hashes(block_ids: list[int]) -> BlockHashes:
    """Build a hash snapshot for the given block ids."""
    return {block_id: f"hash-{block_id}".encode() for block_id in block_ids}


def drain(
    policy: OffloadPolicy, finished: set[str], blocked: set[str] | None = None
) -> LazyOffloadDrain:
    """Drain an idle step with the given eligibility sets."""
    return policy.drain(
        DrainSignals(
            new_blocks_allocated=0,
            est_next_step_blocks=0,
            finished_request_ids=finished,
            blocked_request_ids=blocked or set(),
        )
    )


class TestDrainThreshold:
    def test_default_threshold_holds_a_small_batch(self) -> None:
        """The default of 100 eligible requests is not met by 99 of them."""
        policy = FIFOOffloadPolicy({})
        finished = {f"req-{index}" for index in range(99)}
        for index in range(99):
            policy.add(make_meta(f"req-{index}"), make_block_hashes([index]))

        assert drain(policy, finished).items == []

    def test_threshold_counts_eligible_requests_only(self) -> None:
        policy = FIFOOffloadPolicy({THRESHOLD_KEY: 2})
        for index in range(3):
            policy.add(make_meta(f"req-{index}"), make_block_hashes([index]))

        assert drain(policy, {"req-0"}).items == []
        drained = drain(policy, {"req-0", "req-2"})
        assert [item.request_id for item in drained.items] == ["req-0", "req-2"]
        assert drained.emptied_request_ids == ["req-0", "req-2"]
        assert policy.has_pending_request("req-1")

    def test_blocked_request_is_neither_counted_nor_released(self) -> None:
        policy = FIFOOffloadPolicy({THRESHOLD_KEY: 1})
        policy.add(make_meta("blocked"), make_block_hashes([0]))
        policy.add(make_meta("ready"), make_block_hashes([1]))

        drained = drain(policy, {"blocked", "ready"}, blocked={"blocked"})

        assert [item.request_id for item in drained.items] == ["ready"]
        assert policy.has_pending_request("blocked")

    def test_select_count_caps_and_orders_the_drain(self) -> None:
        policy = FIFOOffloadPolicy({THRESHOLD_KEY: 3, SELECT_COUNT_KEY: 2})
        finished = {f"req-{index}" for index in range(5)}
        for index in range(5):
            policy.add(make_meta(f"req-{index}"), make_block_hashes([index]))

        first = drain(policy, finished).items
        second = drain(policy, finished).items

        assert [item.request_id for item in first] == ["req-0", "req-1"]
        assert [item.request_id for item in second] == ["req-2", "req-3"]
        # One request left, below the threshold of three.
        assert drain(policy, finished).items == []


class TestCoalescingAndSnapshots:
    def test_one_request_drains_as_one_item(self) -> None:
        policy = FIFOOffloadPolicy({THRESHOLD_KEY: 1})
        policy.add(make_meta("req"), make_block_hashes([0]))
        policy.add(make_meta("req"), make_block_hashes([0, 1]))

        (item,) = drain(policy, {"req"}).items

        assert len(item.metadatas) == 2

    def test_add_keeps_the_hash_snapshot_untouched(self) -> None:
        policy = FIFOOffloadPolicy({THRESHOLD_KEY: 1})
        meta = make_meta("req")
        policy.add(meta, make_block_hashes([0, 1]))

        (item,) = drain(policy, {"req"}).items

        assert item.metadatas == [(meta, {0: b"hash-0", 1: b"hash-1"})]


class TestDiscard:
    def test_drop_request_reports_the_discarded_operations(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(make_meta("req"), make_block_hashes([0]))
        policy.add(make_meta("req"), make_block_hashes([0, 1]))

        assert policy.drop_request("req") == 2
        assert policy.drop_request("req") == 0

    def test_discard_for_reuse_empties_the_buffer(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(make_meta("req"), make_block_hashes([0]))

        policy.discard_for_reuse("req")

        assert not policy.has_pending_request("req")

    def test_failed_store_reports_nothing_dropped(self) -> None:
        """FIFO keeps no prefix-chain state, so a failure drops nothing and the
        request's buffer is left as it was.
        """
        policy = FIFOOffloadPolicy({})
        policy.add(make_meta("req"), make_block_hashes([0]))

        assert policy.mark_store_failed("req") == 0
        assert policy.has_pending_request("req")
