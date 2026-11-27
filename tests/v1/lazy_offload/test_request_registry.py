# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle and submitted-batch contracts for lazy offload.

``LazyOffloadRequestRegistry`` answers two questions for the manager: has this
request finished, and does it still have a store batch in flight. The hard
cases are the ones where a request id outlives its owner -- a preemption reset,
or a new request reusing a finished id -- where the outstanding batch must be
detached from the generation now holding the id.
"""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_state import LazyOffloadRequestRegistry


def test_arrival_opens_an_active_slot_with_nothing_in_flight() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")

    assert not registry.is_finished("req")
    assert not registry.has_in_flight("req")
    assert registry.finished_request_ids() == set()


def test_unknown_request_is_neither_finished_nor_in_flight() -> None:
    """The manager queries ids it has never seen; that must not create state."""
    registry = LazyOffloadRequestRegistry()

    assert not registry.is_finished("never-seen")
    assert not registry.has_in_flight("never-seen")
    assert not registry.can_end_session("never-seen")


def test_fresh_batch_is_not_orphaned() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1, 2])

    assert registry.has_in_flight("req")
    assert registry.in_flight_request_ids() == {"req"}
    assert not registry.in_flight_is_orphaned("req")


def test_reset_orphans_the_in_flight_batch() -> None:
    """A preemption reset detaches the batch but never drops its pins."""
    registry = LazyOffloadRequestRegistry()
    registry.register_batch("req", [1])
    assert not registry.in_flight_is_orphaned("req")

    registry.reset("req")

    assert registry.has_in_flight("req")
    assert registry.in_flight_is_orphaned("req")


def test_rearrival_after_reset_keeps_a_new_batch_fresh() -> None:
    """The recreated tracker's arrival must not orphan the successor's own batch: only
    the pre-reset batch is detached, and its completion returns it with the flag set.
    """
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1])
    registry.reset("req")
    registry.arrive("req")

    stale = registry.complete_batch("req")
    assert stale.block_ids == (1,)
    assert stale.orphaned

    registry.register_batch("req", [2])
    assert not registry.in_flight_is_orphaned("req")


def test_finished_id_reuse_orphans_the_outstanding_batch() -> None:
    """A distinct request reusing a finished id detaches the predecessor's batch; the
    batch still closes normally and keeps its block ids.
    """
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1, 2])
    registry.finish("req")
    assert not registry.in_flight_is_orphaned("req")

    registry.arrive("req")

    assert not registry.is_finished("req")
    assert registry.in_flight_is_orphaned("req")
    batch = registry.complete_batch("req")
    assert batch.block_ids == (1, 2)
    assert batch.orphaned
    # The successor is active, so its session must not be releasable.
    assert not registry.can_end_session("req")


def test_session_release_requires_finished_request_and_no_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.register_batch("req", [1])
    registry.finish("req")
    assert registry.is_finished("req")
    assert registry.finished_request_ids() == {"req"}
    assert not registry.can_end_session("req")
    # session_ended on a non-releasable slot is a no-op.
    registry.session_ended("req")
    assert registry.has_in_flight("req")

    registry.complete_batch("req")
    assert registry.can_end_session("req")
    registry.session_ended("req")
    assert not registry.can_end_session("req")
    assert registry.finished_request_ids() == set()


def test_overlapping_batches_are_rejected() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.register_batch("req", [1])

    with pytest.raises(RuntimeError, match="already has an in-flight"):
        registry.register_batch("req", [2])


def test_complete_batch_requires_an_open_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    with pytest.raises(KeyError):
        registry.complete_batch("unknown")
    registry.arrive("req")
    with pytest.raises(KeyError):
        registry.complete_batch("req")


def test_finish_clears_the_rearrival_marker_for_later_id_reuse() -> None:
    """After reset + finish, a rearrival is an id reuse, not the preempted tracker
    coming back: a batch submitted between them (a finished request's op draining late)
    must be orphaned by that arrival.
    """
    registry = LazyOffloadRequestRegistry()
    registry.reset("req")
    registry.finish("req")
    registry.register_batch("req", [1])

    registry.arrive("req")

    assert registry.in_flight_is_orphaned("req")
