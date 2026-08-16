# SPDX-License-Identifier: Apache-2.0
"""Request-epoch state-machine contracts for lazy offload."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_state import (
    LazyOffloadRequestRegistry,
    RequestPhase,
)


def test_preemption_reset_advances_epoch_once_across_rearrival() -> None:
    registry = LazyOffloadRequestRegistry()
    assert registry.arrive("req") == 0

    assert registry.reset("req") == 1
    assert registry.arrive("req") == 1


def test_reset_makes_existing_batch_stale() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.register_batch("req", [1])
    assert registry.in_flight_is_current("req")

    registry.reset("req")
    assert not registry.in_flight_is_current("req")


def test_finished_id_reuse_advances_epoch_and_keeps_old_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1, 2])
    registry.finish("req")

    assert registry.arrive("req") == 1
    assert registry.is_active("req")
    assert not registry.in_flight_is_current("req")
    batch = registry.complete_batch("req")
    assert batch.epoch == 0
    assert batch.block_ids == (1, 2)
    assert not registry.can_end_session("req")


def test_session_release_requires_finished_request_and_no_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1])
    registry.finish("req")
    assert registry.is_finished("req")
    assert registry.finished_request_ids() == {"req"}
    assert not registry.can_end_session("req")

    registry.complete_batch("req")
    assert registry.can_end_session("req")
    registry.session_ended("req")
    assert not registry.can_end_session("req")


def test_overlapping_batches_are_rejected() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.register_batch("req", [1])

    with pytest.raises(RuntimeError, match="already has an in-flight"):
        registry.register_batch("req", [2])


def test_finish_clears_reset_rearrival_marker() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.reset("req")
    registry.finish("req")

    assert registry.arrive("req") == 2


def test_initial_candidate_state_is_active() -> None:
    registry = LazyOffloadRequestRegistry()
    assert registry.ensure_active("req") == 0
    assert registry.is_active("req")
    assert registry.is_current_epoch("req", 0)
    assert not registry.is_current_epoch("req", 1)
    # Keep the enum in this public contract: slots use explicit phases rather
    # than booleans with call-site-specific meanings.
    assert RequestPhase.ACTIVE is not RequestPhase.FINISHED
