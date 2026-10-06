# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's cache-event ingest gate: seq dedup, gap
detection, incarnation fencing, and what each of those hands to the
registered consumers."""

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate, IngestResult
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
import lmcache.v1.mp_coordinator.ingest.event_gate as event_gate


class _RecordingConsumer:
    """Consumer that records the calls the gate makes on it."""

    def __init__(self) -> None:
        self.batches: list[CacheEventBatch] = []
        self.fenced: list[str] = []

    def consume(self, batch: CacheEventBatch) -> None:
        self.batches.append(batch)

    def fence_instance(self, instance_id: str) -> None:
        self.fenced.append(instance_id)


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: CacheEventType = CacheEventType.STORE,
    tier: Tier = Tier.L1,
    backend: str = "dram",
    keys: list[ObjectKey] | None = None,
    size_bytes: int = 1024,
    shared: bool = False,
    dropped_events: int = 0,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        entries=[
            CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=size_bytes)
            for k in (keys or [_key(0xAA)])
        ],
        shared=shared,
        dropped_events=dropped_events,
    )


def _gate(*consumers: _RecordingConsumer | KeyDirectory) -> EventGate:
    broadcaster = CacheEventBroadcaster()
    for consumer in consumers:
        broadcaster.register_consumer(consumer)
    return EventGate(broadcaster, QuiesceLock())


# -- Admission ---------------------------------------------------------------


def test_admitted_batch_reaches_every_consumer():
    first, second = _RecordingConsumer(), _RecordingConsumer()
    gate = _gate(first, second)
    batch = _batch(keys=[_key(1)])

    assert gate.ingest(batch) == IngestResult.ADMITTED

    assert first.batches == [batch]
    assert second.batches == [batch]


def test_gate_with_no_consumers_admits():
    assert _gate().ingest(_batch()) == IngestResult.ADMITTED


def test_ingest_batches_forwards_in_source_order():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    first = _batch(incarnation=1, seq=1)
    second = _batch(incarnation=1, seq=2)

    summary = gate.ingest_batches([first, second])

    assert summary.applied == 2
    assert summary.duplicates == 0
    assert summary.stale == 0
    assert consumer.batches == [first, second]


def test_ingest_batches_aggregates_outcomes():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    current = _batch(incarnation=2, seq=1)
    fresh = _batch(incarnation=2, seq=2)

    summary = gate.ingest_batches(
        [
            current,
            _batch(incarnation=2, seq=1),
            _batch(incarnation=1, seq=9),
            fresh,
        ]
    )

    assert summary.applied == 2
    assert summary.duplicates == 1
    assert summary.stale == 1
    assert consumer.batches == [current, fresh]


def test_ingest_batches_preserves_incarnation_fencing():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)

    summary = gate.ingest_batches(
        [
            _batch(incarnation=1, seq=1),
            _batch(incarnation=2, seq=1),
        ]
    )

    assert summary.applied == 2
    assert consumer.fenced == ["node-a"]


# -- Seq handling ------------------------------------------------------------


def test_duplicate_seq_is_dropped_before_the_consumers():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(seq=1, size_bytes=100))

    assert gate.ingest(_batch(seq=1, size_bytes=999)) == IngestResult.DUPLICATE
    assert len(consumer.batches) == 1


def test_replayed_older_seq_is_dropped():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2, event_type=CacheEventType.DELETE))

    assert gate.ingest(_batch(seq=1)) == IngestResult.DUPLICATE
    assert len(consumer.batches) == 2


def test_seq_gap_sets_the_gap_flag_but_admits():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(seq=1))

    assert gate.ingest(_batch(seq=5)) == IngestResult.ADMITTED

    stream = gate.stats()["node-a"]
    assert stream.gap_detected is True
    assert stream.last_seq == 5
    assert len(consumer.batches) == 2


# -- Loss accounting -----------------------------------------------------------


def _loss(gate: EventGate) -> tuple[int, int, int]:
    stream = gate.stats()["node-a"]
    return (
        stream.loss_incidents_total,
        stream.lost_events_total,
        stream.admitted_events_total,
    )


def test_every_seq_gap_is_an_incident():
    """The flag latches on the first gap; the counter keeps counting."""
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=4))  # 2..3 missing
    gate.ingest(_batch(seq=5))
    gate.ingest(_batch(seq=7))  # 6 missing

    assert gate.stats()["node-a"].gap_detected is True
    assert _loss(gate) == (2, 0, 4)


def test_lost_events_are_the_reported_count_deltas():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2, dropped_events=3))
    gate.ingest(_batch(seq=3, dropped_events=3))
    gate.ingest(_batch(seq=4, dropped_events=10))

    assert _loss(gate) == (2, 10, 4)


def test_reported_loss_sets_the_gap_flag_and_warns_once(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        event_gate.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2, dropped_events=2))
    gate.ingest(_batch(seq=3, dropped_events=5))

    assert gate.stats()["node-a"].gap_detected is True
    assert warnings == [
        "Event loss for instance node-a (incarnation 1): seq 1 -> 2, "
        "2 events reported lost; slice needs replay"
    ]


def test_seq_jump_and_reported_loss_in_one_batch_are_one_incident():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=3, dropped_events=4))

    assert _loss(gate) == (1, 4, 2)


def test_first_batch_of_an_unseen_stream_only_sets_the_baseline():
    """Seqs and losses from before the gate first saw the stream (e.g.
    before a coordinator restart) are not counted; the flag still marks
    the slice stale."""
    gate = _gate()
    gate.ingest(_batch(seq=1000, dropped_events=5))
    assert gate.stats()["node-a"].gap_detected is True
    assert _loss(gate) == (0, 0, 1)

    gate.ingest(_batch(seq=1001, dropped_events=7))
    assert _loss(gate) == (1, 2, 2)


def test_loss_before_a_streams_first_seq_is_counted():
    """Nothing precedes seq 1, so its reported loss is all real."""
    gate = _gate()
    gate.ingest(_batch(seq=1, dropped_events=3))

    assert gate.stats()["node-a"].gap_detected is True
    assert _loss(gate) == (1, 3, 1)


def test_a_restored_cursor_only_sets_the_loss_baseline():
    """The checkpoint holds no baseline, so the first batch after a
    restore sets it; a seq jump is still measured from the restored
    ``last_seq``."""
    gate = _gate()
    gate.ingest(_batch(seq=1, dropped_events=2))
    state = gate.capture()

    restored = _gate()
    restored.restore(state)
    restored.ingest(_batch(seq=2, dropped_events=6))
    assert _loss(restored) == (0, 0, 1)

    restored.ingest(_batch(seq=4, dropped_events=9))  # seq 3 missing too
    assert _loss(restored) == (1, 3, 2)


def test_a_new_incarnation_counts_loss_from_zero():
    gate = _gate()
    gate.ingest(_batch(incarnation=1, seq=1, dropped_events=4))
    gate.ingest(_batch(incarnation=2, seq=1, dropped_events=2))

    stream = gate.stats()["node-a"]
    assert stream.incarnation == 2
    assert _loss(gate) == (1, 2, 1)


def test_gap_at_the_start_of_a_new_incarnation_is_counted():
    gate = _gate()
    gate.ingest(_batch(incarnation=1, seq=1))
    gate.ingest(_batch(incarnation=2, seq=3))  # 1..2 of the restart lost

    assert _loss(gate) == (1, 0, 1)


def test_duplicates_count_nothing():
    gate = _gate()
    gate.ingest(_batch(seq=1, keys=[_key(1), _key(2)]))
    gate.ingest(_batch(seq=2, dropped_events=3))
    gate.ingest(_batch(seq=1, keys=[_key(1), _key(2)]))
    gate.ingest(_batch(seq=2, dropped_events=3))

    assert _loss(gate) == (1, 3, 3)


def test_contiguous_seqs_do_not_flag_gap():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2))

    assert gate.stats()["node-a"].gap_detected is False


def test_each_instance_has_its_own_cursor():
    gate = _gate()
    gate.ingest(_batch(instance_id="node-a", seq=1))

    assert gate.ingest(_batch(instance_id="node-b", seq=1)) == IngestResult.ADMITTED
    assert set(gate.stats()) == {"node-a", "node-b"}


# -- Incarnation fencing -----------------------------------------------------


def test_new_incarnation_fences_consumers_before_admitting():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(incarnation=1, seq=1))

    assert gate.ingest(_batch(incarnation=2, seq=1)) == IngestResult.ADMITTED

    assert consumer.fenced == ["node-a"]
    stream = gate.stats()["node-a"]
    assert stream.incarnation == 2
    assert stream.last_seq == 1


def test_new_incarnation_drops_the_directory_l1_placements():
    directory = KeyDirectory()
    gate = _gate(directory)
    gate.ingest(_batch(incarnation=1, seq=1, keys=[_key(1), _key(2)]))

    gate.ingest(_batch(incarnation=2, seq=1, keys=[_key(3)]))

    assert directory.lookup([_key(1)]) == [[]]
    assert directory.lookup([_key(2)]) == [[]]
    [placements] = directory.lookup([_key(3)])
    assert placements[0].incarnation == 2


def test_fence_spares_other_instances_placements():
    directory = KeyDirectory()
    gate = _gate(directory)
    gate.ingest(_batch(instance_id="node-a", keys=[_key(1)]))
    gate.ingest(_batch(instance_id="node-b", keys=[_key(1)]))

    gate.ingest(_batch(instance_id="node-a", incarnation=2, keys=[_key(9)]))

    [placements] = directory.lookup([_key(1)])
    assert [p.instance_id for p in placements] == ["node-b"]


def test_stale_incarnation_batch_is_dropped():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(incarnation=2, seq=1))

    outcome = gate.ingest(_batch(incarnation=1, seq=99))

    assert outcome == IngestResult.STALE_INCARNATION
    assert len(consumer.batches) == 1
    assert consumer.fenced == []


def test_same_incarnation_never_fences():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(incarnation=3, seq=1))
    gate.ingest(_batch(incarnation=3, seq=2))

    assert consumer.fenced == []


# -- drop_instance -----------------------------------------------------------


def test_drop_instance_fences_consumers_and_forgets_the_cursor():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(incarnation=5, seq=9))

    gate.drop_instance("node-a")

    assert consumer.fenced == ["node-a"]
    assert gate.stats() == {}
    # A reconnect starts fresh with any incarnation.
    assert gate.ingest(_batch(incarnation=1, seq=1)) == IngestResult.ADMITTED


def test_drop_unknown_instance_is_noop_for_the_cursor():
    gate = _gate()
    gate.drop_instance("ghost")

    assert gate.stats() == {}


# -- Stats -------------------------------------------------------------------


def test_stats_are_empty_before_any_event():
    assert _gate().stats() == {}
