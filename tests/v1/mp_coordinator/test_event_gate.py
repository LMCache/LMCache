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
from lmcache.v1.mp_coordinator.ingest.event_gate import (
    EventGate,
    IngestResult,
)
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory


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


def _loss(gate: EventGate, instance_id: str = "node-a") -> tuple[int, int, bool]:
    stream = gate.stats().streams[instance_id]
    return stream.missing_batches, stream.events_dropped, stream.gap_detected


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

    stream = gate.stats().streams["node-a"]
    assert stream.gap_detected is True
    assert stream.last_seq == 5
    assert len(consumer.batches) == 2


def test_contiguous_seqs_do_not_flag_gap():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2))

    assert gate.stats().streams["node-a"].gap_detected is False


def test_each_instance_has_its_own_cursor():
    gate = _gate()
    gate.ingest(_batch(instance_id="node-a", seq=1))

    assert gate.ingest(_batch(instance_id="node-b", seq=1)) == IngestResult.ADMITTED
    assert set(gate.stats().streams) == {"node-a", "node-b"}


# -- Incarnation fencing -----------------------------------------------------


def test_new_incarnation_fences_consumers_before_admitting():
    consumer = _RecordingConsumer()
    gate = _gate(consumer)
    gate.ingest(_batch(incarnation=1, seq=1))

    assert gate.ingest(_batch(incarnation=2, seq=1)) == IngestResult.ADMITTED

    assert consumer.fenced == ["node-a"]
    stream = gate.stats().streams["node-a"]
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
    assert gate.stats().streams == {}
    # A reconnect starts fresh with any incarnation.
    assert gate.ingest(_batch(incarnation=1, seq=1)) == IngestResult.ADMITTED


def test_drop_unknown_instance_is_noop_for_the_cursor():
    gate = _gate()
    gate.drop_instance("ghost")

    assert gate.stats().streams == {}


# -- Stats -------------------------------------------------------------------


def test_stats_are_empty_before_any_event():
    assert _gate().stats().streams == {}


# -- Loss accounting ---------------------------------------------------------


def test_every_skipped_seq_counts_while_the_flag_stays_set():
    """The flag says the slice is incomplete; the count says how much."""
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=4))  # 2..3 missing
    gate.ingest(_batch(seq=5))
    gate.ingest(_batch(seq=7))  # 6 missing

    assert _loss(gate) == (3, 0, True)


def test_first_batch_of_an_untracked_stream_counts_no_missing_batches():
    """Earlier batches were sent before the gate was listening: the slice
    is incomplete, but nothing was lost in transit."""
    gate = _gate()
    gate.ingest(_batch(seq=50))

    assert _loss(gate) == (0, 0, True)


def test_a_restarted_emitter_that_skips_its_first_seqs_counts_them():
    gate = _gate()
    gate.ingest(_batch(incarnation=1, seq=1))
    gate.ingest(_batch(incarnation=2, seq=3))  # 1..2 of the new run missing

    assert _loss(gate) == (2, 0, True)


def test_dropped_events_count_the_increases_of_the_reported_total():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=2, dropped_events=3))
    gate.ingest(_batch(seq=3, dropped_events=3))
    gate.ingest(_batch(seq=4, dropped_events=10))

    assert _loss(gate) == (0, 10, True)


def test_a_stream_joined_midway_only_learns_the_dropped_baseline():
    gate = _gate()
    gate.ingest(_batch(seq=7, dropped_events=40))
    gate.ingest(_batch(seq=8, dropped_events=45))

    assert gate.stats().streams["node-a"].events_dropped == 5


def test_a_stream_seen_from_its_first_batch_counts_its_first_report():
    gate = _gate()
    gate.ingest(_batch(seq=1, dropped_events=4))

    assert _loss(gate) == (0, 4, True)


def test_a_restart_counts_the_new_run_s_drops_from_zero():
    gate = _gate()
    gate.ingest(_batch(incarnation=1, seq=1, dropped_events=9))
    gate.ingest(_batch(incarnation=2, seq=1, dropped_events=2))

    assert gate.stats().streams["node-a"].events_dropped == 2


def test_rejected_batches_do_not_move_the_dropped_baseline():
    gate = _gate()
    gate.ingest(_batch(incarnation=2, seq=1))
    gate.ingest(_batch(incarnation=2, seq=1, dropped_events=50))  # duplicate
    gate.ingest(_batch(incarnation=1, seq=9, dropped_events=70))  # stale
    gate.ingest(_batch(incarnation=2, seq=2, dropped_events=5))

    assert gate.stats().streams["node-a"].events_dropped == 5


def test_loss_counts_start_over_when_the_emitter_leaves():
    gate = _gate()
    gate.ingest(_batch(seq=1))
    gate.ingest(_batch(seq=3, dropped_events=2))
    gate.drop_instance("node-a")
    gate.ingest(_batch(seq=4))

    assert _loss(gate) == (0, 0, True)


def test_totals_cover_the_whole_fleet_and_outlive_departures():
    gate = _gate()
    gate.ingest(_batch(instance_id="node-a", seq=1))
    gate.ingest(_batch(instance_id="node-a", seq=3, dropped_events=4))
    gate.ingest(_batch(instance_id="node-a", seq=3))  # duplicate
    gate.ingest(_batch(instance_id="node-b", incarnation=2, seq=1))
    gate.ingest(_batch(instance_id="node-b", incarnation=1, seq=2))  # stale
    gate.ingest(_batch(instance_id="node-b", incarnation=2, seq=4))
    gate.drop_instance("node-a")

    stats = gate.stats()
    assert (
        stats.batches_applied,
        stats.batches_duplicate,
        stats.batches_stale,
        stats.batches_missing,
        stats.events_dropped,
    ) == (4, 1, 1, 3, 4)


def test_restored_gate_measures_drops_against_the_checkpointed_baseline():
    live = _gate()
    live.ingest(_batch(seq=1, dropped_events=6))
    restored = _gate()
    restored.restore(live.capture())

    restored.ingest(_batch(seq=2, dropped_events=8))

    assert _loss(restored) == (0, 2, True)


def test_a_cursor_checkpointed_without_a_baseline_restores_with_none_known():
    restored = _gate()
    restored.restore({"cursors": {"node-a": (1, 4, False)}})

    restored.ingest(_batch(seq=5, dropped_events=30))
    restored.ingest(_batch(seq=6, dropped_events=31))

    assert _loss(restored) == (0, 1, True)
