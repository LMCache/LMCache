# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`StreamPosition`."""

# First Party
from lmcache.v1.mp_coordinator.ingest.stream_position import StreamPosition


def test_next_offset_is_none_for_a_partition_never_recorded() -> None:
    position = StreamPosition()

    assert position.next_offset("cache-events", 0) is None


def test_next_offset_is_one_past_the_last_recorded_offset() -> None:
    position = StreamPosition()

    position.record("cache-events", 0, 41)

    assert position.next_offset("cache-events", 0) == 42


def test_a_later_record_overwrites_an_earlier_one() -> None:
    position = StreamPosition()

    position.record("cache-events", 0, 10)
    position.record("cache-events", 0, 20)

    assert position.next_offset("cache-events", 0) == 21


def test_partitions_are_tracked_independently() -> None:
    position = StreamPosition()

    position.record("cache-events", 0, 5)
    position.record("cache-events", 1, 99)

    assert position.next_offset("cache-events", 0) == 6
    assert position.next_offset("cache-events", 1) == 100
    assert position.next_offset("cache-events", 2) is None


def test_capture_and_restore_round_trip() -> None:
    original = StreamPosition()
    original.record("cache-events", 0, 5)
    original.record("cache-events", 1, 99)

    restored = StreamPosition()
    restored.restore(original.capture())

    assert restored.next_offset("cache-events", 0) == 6
    assert restored.next_offset("cache-events", 1) == 100


def test_restore_replaces_rather_than_merges() -> None:
    position = StreamPosition()
    position.record("cache-events", 0, 5)

    position.restore({"offsets": [["cache-events", 1, 99]]})

    assert position.next_offset("cache-events", 0) is None
    assert position.next_offset("cache-events", 1) == 100


def test_capture_is_plain_data() -> None:
    """The artifact writer only ever sees dicts, lists and scalars."""
    position = StreamPosition()
    position.record("cache-events", 0, 5)

    captured = position.capture()

    assert captured == {"offsets": [["cache-events", 0, 5]]}
