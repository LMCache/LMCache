# SPDX-License-Identifier: Apache-2.0
"""Tests for the cache-event broadcaster (consumer registration plus
batch and fence fan-out)."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import (
    CacheEventBroadcaster,
    ConsumerStats,
)
import lmcache.v1.mp_coordinator.ingest.event_broadcaster as event_broadcaster


class _RecordingConsumer:
    def __init__(self, name: str, log: list[tuple[str, object]]) -> None:
        self._name = name
        self._log = log

    def consume(self, batch: CacheEventBatch) -> None:
        self._log.append((self._name, batch))

    def fence_instance(self, instance_id: str) -> None:
        self._log.append((self._name, instance_id))


class _RaisingConsumer:
    def consume(self, batch: CacheEventBatch) -> None:
        raise RuntimeError("consume bug")

    def fence_instance(self, instance_id: str) -> None:
        raise RuntimeError("fence bug")


def _batch(seq: int = 1) -> CacheEventBatch:
    key = ObjectKey(chunk_hash=b"\xaa" * 4, model_name="m", kv_rank=0)
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=1)],
    )


def test_broadcast_fans_out_to_consumers_in_registration_order():
    log: list[tuple[str, object]] = []
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RecordingConsumer("first", log))
    broadcaster.register_consumer(_RecordingConsumer("second", log))

    batch = _batch()
    broadcaster.broadcast(batch)

    assert log == [("first", batch), ("second", batch)]


def test_broadcast_with_no_consumers_is_a_noop():
    CacheEventBroadcaster().broadcast(_batch())


def test_fence_with_no_consumers_is_a_noop():
    CacheEventBroadcaster().fence_instance("node-a")


def test_fence_reaches_every_consumer_in_registration_order():
    log: list[tuple[str, object]] = []
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RecordingConsumer("first", log))
    broadcaster.register_consumer(_RecordingConsumer("second", log))

    broadcaster.fence_instance("node-a")

    assert log == [("first", "node-a"), ("second", "node-a")]


def test_consumer_registered_later_sees_only_later_batches():
    log: list[tuple[str, object]] = []
    broadcaster = CacheEventBroadcaster()
    first, second = _batch(seq=1), _batch(seq=2)
    broadcaster.broadcast(first)
    broadcaster.register_consumer(_RecordingConsumer("late", log))
    broadcaster.broadcast(second)

    assert log == [("late", second)]


def _capture_exceptions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the module logger's exception lines; it does not propagate."""
    lines: list[str] = []
    monkeypatch.setattr(
        event_broadcaster.logger,
        "exception",
        lambda msg, *args: lines.append(msg % args),
    )
    return lines


def test_a_consumer_that_raises_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[tuple[str, object]] = []
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RaisingConsumer())
    broadcaster.register_consumer(_RecordingConsumer("after", log))
    errors = _capture_exceptions(monkeypatch)

    batch = _batch()
    broadcaster.broadcast(batch)

    assert log == [("after", batch)]
    assert errors == [
        "Cache-event consumer _RaisingConsumer failed on batch node-a/1/1"
    ]


def test_a_fence_that_raises_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[tuple[str, object]] = []
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RaisingConsumer())
    broadcaster.register_consumer(_RecordingConsumer("after", log))
    errors = _capture_exceptions(monkeypatch)

    broadcaster.fence_instance("node-a")

    assert log == [("after", "node-a")]
    assert errors == ["Cache-event consumer _RaisingConsumer failed to fence node-a"]


def test_stats_tally_each_consumer_s_batches_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RaisingConsumer())
    broadcaster.register_consumer(_RecordingConsumer("ok", []))
    _capture_exceptions(monkeypatch)

    broadcaster.broadcast(_batch(seq=1))
    broadcaster.broadcast(_batch(seq=2))
    broadcaster.fence_instance("node-a")

    stats = broadcaster.stats()
    raising, recording = stats["_RaisingConsumer"], stats["_RecordingConsumer"]
    assert (
        raising.batches_delivered,
        raising.consume_failures,
        raising.fence_failures,
    ) == (2, 2, 1)
    assert (
        recording.batches_delivered,
        recording.consume_failures,
        recording.fence_failures,
    ) == (2, 0, 0)
    assert raising.apply_seconds >= 0 and recording.apply_seconds >= 0


def test_stats_list_every_consumer_before_any_batch() -> None:
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RecordingConsumer("ok", []))

    assert broadcaster.stats() == {"_RecordingConsumer": ConsumerStats()}
