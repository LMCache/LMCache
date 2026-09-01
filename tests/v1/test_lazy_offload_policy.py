# SPDX-License-Identifier: Apache-2.0
"""Tests for lazy-offload policy selection and the FIFO policy.

Carried over from the pre-existing pending-store suite and adapted to the
drain interface the eviction-aware policy introduces: the controller now
supplies the finished and blocked request-id sets, so the FIFO policy no
longer tracks request lifecycle itself.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy import (
    LazyOffloadMode,
    create_offload_policy,
)
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
)
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    EvictionAwareStoreQueue,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy

FIFO_CONFIG: dict[str, ConfigValue] = {
    "lmcache.mp.lazy_offload_policy": LazyOffloadMode.FIFO.value
}


def _make_meta(request_id: str = "req-0", num_blocks: int = 1) -> MagicMock:
    """Build a mock ``LMCacheMPRequestMetadata`` covering ``num_blocks``."""
    meta = MagicMock()
    meta.request_id = request_id
    meta.op.flat_block_ids = list(range(num_blocks))
    return meta


def _make_block_hashes(block_ids: list[int]) -> BlockHashes:
    """Build a hash snapshot for the given block ids."""
    return {block_id: f"hash-{block_id}".encode() for block_id in block_ids}


def _drain(policy: OffloadPolicy, finished: set[str]) -> LazyOffloadDrain:
    """Drain an idle step in which ``finished`` requests are eligible."""
    return policy.drain(
        DrainSignals(
            new_blocks_allocated=0,
            est_next_step_blocks=0,
            allocated_block_ids=set(),
            finished_request_ids=finished,
            blocked_request_ids=set(),
        )
    )


class TestPolicySelection:
    def test_default_is_eviction_aware(self) -> None:
        policy = create_offload_policy({}, MagicMock())
        assert isinstance(policy, EvictionAwareStoreQueue)

    def test_fifo_is_selectable(self) -> None:
        policy = create_offload_policy(dict(FIFO_CONFIG), MagicMock())
        assert isinstance(policy, FIFOOffloadPolicy)

    def test_unknown_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown offload policy"):
            create_offload_policy(
                {"lmcache.mp.lazy_offload_policy": "UNKNOWN"}, MagicMock()
            )


class TestFIFOOffloadPolicy:
    def test_configures_threshold_and_select_count(self) -> None:
        policy = FIFOOffloadPolicy({})
        assert policy._threshold == 100
        assert policy._select_count == 10
        policy = FIFOOffloadPolicy(
            {
                "lmcache.mp.lazy_offload_threshold": 3,
                "lmcache.mp.lazy_offload_select_count": 2,
            }
        )
        assert policy._threshold == 3
        assert policy._select_count == 2

    def test_add_aggregates_one_request_epoch(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(_make_meta("req", 1), _make_block_hashes([0]), epoch=2)
        policy.add(_make_meta("req", 2), _make_block_hashes([0, 1]), epoch=2)
        assert len(policy._pending_items["req"].metadatas) == 2

    def test_add_rejects_mixed_epochs_for_one_request(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(_make_meta("req"), _make_block_hashes([0]), epoch=2)
        with pytest.raises(RuntimeError, match="mixed store epochs 2 and 3"):
            policy.add(_make_meta("req"), _make_block_hashes([1]), epoch=3)

    def test_threshold_counts_controller_eligible_requests(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for index in range(3):
            policy.add(_make_meta(f"req-{index}"), _make_block_hashes([index]), 0)

        assert _drain(policy, {"req-0"}).items == []
        drained = _drain(policy, {"req-0", "req-2"})
        assert [item.request_id for item in drained.items] == ["req-0", "req-2"]
        assert drained.emptied_request_ids == ["req-0", "req-2"]
        assert policy.has_pending_request("req-1")

    def test_blocked_request_is_not_eligible_or_popped(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for request_id, block_id in (("blocked", 0), ("ready", 1)):
            policy.add(_make_meta(request_id), _make_block_hashes([block_id]), 0)

        drained = policy.drain(
            DrainSignals(
                new_blocks_allocated=0,
                est_next_step_blocks=0,
                allocated_block_ids=set(),
                finished_request_ids={"blocked", "ready"},
                blocked_request_ids={"blocked"},
            )
        )
        assert [item.request_id for item in drained.items] == ["ready"]
        assert policy.has_pending_request("blocked")

    def test_select_count_caps_and_orders_the_drain(self) -> None:
        policy = FIFOOffloadPolicy(
            {
                "lmcache.mp.lazy_offload_threshold": 3,
                "lmcache.mp.lazy_offload_select_count": 2,
            }
        )
        finished = {f"req-{index}" for index in range(5)}
        for index in range(5):
            meta = _make_meta(f"req-{index}")
            policy.add(meta, _make_block_hashes([index]), 0)

        first = _drain(policy, finished).items
        second = _drain(policy, finished).items
        assert [item.request_id for item in first] == ["req-0", "req-1"]
        assert [item.request_id for item in second] == ["req-2", "req-3"]
        # One request left, below the threshold of three.
        assert _drain(policy, finished).items == []

    def test_add_keeps_the_hash_snapshot_for_the_controller(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        meta = _make_meta("req", num_blocks=2)
        policy.add(meta, _make_block_hashes([0, 1]), 0)
        (item,) = _drain(policy, {"req"}).items
        assert item.metadatas == [(meta, {0: b"hash-0", 1: b"hash-1"})]

    def test_discard_operations_report_chunk_count(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(_make_meta("req", 1), _make_block_hashes([0]), 0)
        policy.add(_make_meta("req", 2), _make_block_hashes([0, 1]), 0)
        assert policy.drop_request("req") == 2
        assert policy.drop_request("req") == 0

        policy.add(_make_meta("req"), _make_block_hashes([0]), 0)
        assert policy.discard_for_reuse("req") == 1
        assert not policy.has_pending_request("req")

    def test_failed_store_leaves_nothing_buffered(self) -> None:
        policy = FIFOOffloadPolicy({})
        policy.add(_make_meta("req"), _make_block_hashes([0]), 0)
        assert policy.mark_store_failed("req") == 0
