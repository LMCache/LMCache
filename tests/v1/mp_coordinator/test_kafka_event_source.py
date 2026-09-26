# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's Kafka cache-event source."""

# Standard
from collections.abc import Callable, Mapping
import asyncio
import time

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import (
    HttpCacheEventSourceConfig,
    KafkaCacheEventSourceConfig,
    MPCoordinatorConfig,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.event_source import (
    UNKNOWN_LAG,
    EventReplayCapability,
)
from lmcache.v1.mp_coordinator.ingest.kafka_event_source import (
    KafkaCacheEventSource,
)
from lmcache.v1.mp_coordinator.ingest.stream_position import StreamPosition
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
import lmcache.v1.mp_coordinator.ingest.kafka_event_source as kafka_event_source

# Local
from .fake_kafka import (
    FakeKafkaBroker,
    FakeKafkaConsumer,
    install_fake_confluent_kafka,
    uninstall_confluent_kafka,
)

_TOPIC = "cache-events"
_BOOTSTRAP_SERVERS = "broker-a:9092,broker-b:9092"
_GROUP_ID = "coordinator-tests"


class _RecordingConsumer:
    """Cache-event consumer that records admitted batches."""

    def __init__(self) -> None:
        self.batches: list[CacheEventBatch] = []

    def consume(self, batch: CacheEventBatch) -> None:
        """Record an admitted batch."""
        self.batches.append(batch)

    def fence_instance(self, instance_id: str) -> None:
        """Accept an instance fence without storing state."""


class _FailingOnceConsumer(_RecordingConsumer):
    """Consumer that raises on the first batch it sees, then records."""

    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def consume(self, batch: CacheEventBatch) -> None:
        """Raise once, then record like the base class."""
        if not self._failed:
            self._failed = True
            raise RuntimeError("consumer exploded")
        super().consume(batch)


def _batch(seq: int) -> CacheEventBatch:
    """Build one L1 store batch from instance ``node-a``.

    Args:
        seq: Emitter sequence number; also seeds the chunk hash.

    Returns:
        One valid cache-event batch.
    """
    key = ObjectKey(chunk_hash=seq.to_bytes(4, "big"), model_name="model", kv_rank=0)
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L1,
        backend="dram",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=1024)],
    )


def _record(broker: FakeKafkaBroker, *batches: CacheEventBatch) -> None:
    """Retain one envelope carrying ``batches`` on the topic, as the sink would.

    Args:
        broker: Fake broker to append to.
        batches: Batches the envelope carries, in order.
    """
    broker.append(
        _TOPIC,
        key=b"node-a",
        value=CacheEventsRequest(batches=list(batches)).model_dump_json().encode(),
    )


def _config() -> KafkaCacheEventSourceConfig:
    """Kafka source settings pointing at the fake broker and ``_TOPIC``."""
    return KafkaCacheEventSourceConfig(
        bootstrap_servers=_BOOTSTRAP_SERVERS, topic=_TOPIC, group_id=_GROUP_ID
    )


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``condition`` until it holds or ``timeout`` seconds pass.

    Args:
        condition: Predicate evaluated on the calling thread.
        timeout: Seconds to keep polling.

    Returns:
        The final value of ``condition``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def _source(
    monkeypatch: pytest.MonkeyPatch,
    broker: FakeKafkaBroker,
    recording: _RecordingConsumer | None = None,
    position: StreamPosition | None = None,
) -> tuple[KafkaCacheEventSource, FakeKafkaConsumer, _RecordingConsumer]:
    """Build a Kafka source over the fake consumer, feeding one recorder.

    Args:
        monkeypatch: Fixture scoping the ``confluent_kafka`` module swap.
        broker: Broker the fake consumer reads.
        recording: Cache-event consumer to register; a fresh recorder when
            ``None``.
        position: Checkpoint position to seek from; a fresh (empty) one
            when ``None`` -- pass one in to inspect or pre-populate it.

    Returns:
        The source, its fake Kafka consumer, and the cache-event consumer.
    """
    kafka_consumer: FakeKafkaConsumer | None = None

    def _consumer_factory(config: dict[str, str | int | bool]) -> FakeKafkaConsumer:
        nonlocal kafka_consumer
        kafka_consumer = FakeKafkaConsumer(broker, config)
        return kafka_consumer

    install_fake_confluent_kafka(monkeypatch, consumer_factory=_consumer_factory)
    recording = recording if recording is not None else _RecordingConsumer()
    position = position if position is not None else StreamPosition()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(recording)
    source = KafkaCacheEventSource(
        EventGate(broadcaster, QuiesceLock()), _config(), position
    )
    if kafka_consumer is None:
        raise RuntimeError("Kafka consumer factory was not called")
    return source, kafka_consumer, recording


async def _run_until(source: KafkaCacheEventSource, done: Callable[[], bool]) -> bool:
    """Start ``source``, wait for ``done`` (bounded), then stop it.

    Args:
        source: Source under test.
        done: Predicate the poll thread should eventually satisfy.

    Returns:
        Whether ``done`` held before the wait timed out.
    """
    await source.start()
    try:
        return await asyncio.to_thread(_wait_until, done)
    finally:
        await source.stop()


def test_kafka_source_ingests_records_in_partition_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    _record(broker, _batch(2), _batch(3))
    source, kafka_consumer, recording = _source(monkeypatch, broker)

    assert asyncio.run(_run_until(source, lambda: len(recording.batches) == 3))

    assert [batch.seq for batch in recording.batches] == [1, 2, 3]
    assert kafka_consumer.subscribed == (_TOPIC,)
    assert kafka_consumer.stored_offsets == (0, 1)
    assert kafka_consumer.closed is True


def test_kafka_source_consumes_records_produced_after_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    source, _, recording = _source(monkeypatch, broker)

    async def _produce_while_running() -> bool:
        await source.start()
        try:
            _record(broker, _batch(1))
            return await asyncio.to_thread(
                _wait_until, lambda: len(recording.batches) == 1
            )
        finally:
            await source.stop()

    assert asyncio.run(_produce_while_running())


def test_a_seeked_position_skips_what_it_already_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A position that already reached record 1 makes a fresh source
    resume at record 2 -- not at the start of the topic."""
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    _record(broker, _batch(2))
    position = StreamPosition()
    position.record(_TOPIC, 0, 0)  # record 1 landed at offset 0
    source, _, recording = _source(monkeypatch, broker, position=position)

    assert asyncio.run(_run_until(source, lambda: len(recording.batches) == 1))

    assert [batch.seq for batch in recording.batches] == [2]


def test_restart_from_a_stale_checkpoint_replays_the_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this exists to close: the checkpoint that gets restored is
    older than what was actually consumed before a crash, because the two
    save on independent timers -- a checkpoint taken after record 1 is
    already stale once records 2 and 3 are applied afterward, with no
    second checkpoint before the crash. Resuming from Kafka's own
    tracking (this fake's own ``_positions``, a fresh instance's the
    moment a new process attaches) would skip the gap for good; resuming
    from the checkpoint's own position replays it instead.
    """
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    position = StreamPosition()
    first_source, _, first_recording = _source(monkeypatch, broker, position=position)

    async def _run_first() -> Mapping[str, object]:
        await first_source.start()
        try:
            await asyncio.to_thread(
                _wait_until, lambda: len(first_recording.batches) == 1
            )
            checkpoint = position.capture()  # stale the moment more arrives
            _record(broker, _batch(2))
            _record(broker, _batch(3))
            await asyncio.to_thread(
                _wait_until, lambda: len(first_recording.batches) == 3
            )
            return checkpoint
        finally:
            await first_source.stop()

    stale_checkpoint = asyncio.run(_run_first())

    # A fresh process: a new consumer (this fake never persists a group's
    # committed offset across instances), and a position restored from
    # the stale checkpoint rather than the live one above.
    restored_position = StreamPosition()
    restored_position.restore(stale_checkpoint)
    second_source, _, second_recording = _source(
        monkeypatch, broker, position=restored_position
    )

    assert asyncio.run(
        _run_until(second_source, lambda: len(second_recording.batches) == 2)
    )
    assert [batch.seq for batch in second_recording.batches] == [2, 3]


def test_kafka_source_skips_undecodable_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    broker.append(_TOPIC, key=None, value=b"not json")
    _record(broker, _batch(1))
    source, kafka_consumer, recording = _source(monkeypatch, broker)
    warnings: list[str] = []
    monkeypatch.setattr(
        kafka_event_source.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    assert asyncio.run(_run_until(source, lambda: len(recording.batches) == 1))

    # The bad record's offset is stored too, so the partition keeps moving.
    assert kafka_consumer.stored_offsets == (0, 1)
    [warning] = warnings
    assert warning.startswith(f"Dropping undecodable cache-event record {_TOPIC}[0]@0")


def test_kafka_source_logs_consumer_errors_and_keeps_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    source, kafka_consumer, recording = _source(monkeypatch, broker)
    kafka_consumer.inject_error("broker transport failure")
    warnings: list[str] = []
    monkeypatch.setattr(
        kafka_event_source.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    assert asyncio.run(_run_until(source, lambda: len(recording.batches) == 1))

    assert warnings == [
        f"Kafka cache-event consumer error on topic {_TOPIC}: broker transport failure"
    ]
    assert kafka_consumer.stored_offsets == (0,)


def test_kafka_source_skips_record_that_makes_a_consumer_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    _record(broker, _batch(2))
    source, kafka_consumer, recording = _source(
        monkeypatch, broker, _FailingOnceConsumer()
    )
    failures: list[str] = []
    monkeypatch.setattr(
        kafka_event_source.logger,
        "exception",
        lambda msg, *args: failures.append(msg % args),
    )

    assert asyncio.run(_run_until(source, lambda: len(recording.batches) == 1))

    assert [batch.seq for batch in recording.batches] == [2]
    assert failures == [f"Cache-event consumers failed on record {_TOPIC}[0]@0"]
    assert kafka_consumer.stored_offsets == (0, 1)


def test_kafka_source_reports_seekable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, _ = _source(monkeypatch, FakeKafkaBroker())

    status = source.status()

    assert status.source_name == "kafka"
    assert status.replay_capability == EventReplayCapability.SEEKABLE


def test_kafka_source_reports_unknown_lag_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _ = _source(monkeypatch, FakeKafkaBroker())

    assert source.status().lag == UNKNOWN_LAG


def test_kafka_source_lag_reaches_zero_once_caught_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached lag comes from the same position/watermark query a
    restart's readiness check reads, not a value left over from init."""
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    _record(broker, _batch(2))
    source, _, recording = _source(monkeypatch, broker)
    # The real cadence is a deliberate one query per second; a fast refresh
    # here just keeps the test from waiting on it.
    monkeypatch.setattr(kafka_event_source, "_WATERMARK_INTERVAL", 0.01)

    async def _run() -> bool:
        await source.start()
        try:
            drained = await asyncio.to_thread(
                _wait_until, lambda: len(recording.batches) == 2
            )
            caught_up = await asyncio.to_thread(
                _wait_until, lambda: source.status().lag == 0
            )
            return drained and caught_up
        finally:
            await source.stop()

    assert asyncio.run(_run())


def test_kafka_source_configures_resumable_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, kafka_consumer, _ = _source(monkeypatch, FakeKafkaBroker())

    assert kafka_consumer.config == {
        "bootstrap.servers": _BOOTSTRAP_SERVERS,
        "group.id": _GROUP_ID,
        "client.id": "lmcache-coordinator",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "enable.auto.offset.store": False,
    }


def test_kafka_source_stop_without_start_closes_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, kafka_consumer, _ = _source(monkeypatch, FakeKafkaBroker())

    asyncio.run(source.stop())

    assert kafka_consumer.closed is True


def test_kafka_source_requires_the_kafka_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_confluent_kafka(monkeypatch)
    gate = EventGate(CacheEventBroadcaster(), QuiesceLock())

    with pytest.raises(ImportError, match=r"pip install 'lmcache\[kafka\]'"):
        KafkaCacheEventSource(gate, _config(), StreamPosition())


# -- App wiring -------------------------------------------------------------------


def _app_config(
    event_source_config: HttpCacheEventSourceConfig | KafkaCacheEventSourceConfig,
) -> MPCoordinatorConfig:
    """Coordinator config with the background loops off.

    Args:
        event_source_config: The one cache-event source the app runs.

    Returns:
        The config.
    """
    return MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        event_source_config=event_source_config,
    )


def test_app_consumes_kafka_topic_into_the_directory_and_closes_http_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    kafka_consumers: list[FakeKafkaConsumer] = []

    def _consumer_factory(config: dict[str, str | int | bool]) -> FakeKafkaConsumer:
        kafka_consumers.append(FakeKafkaConsumer(broker, config))
        return kafka_consumers[-1]

    install_fake_confluent_kafka(monkeypatch, consumer_factory=_consumer_factory)
    app = create_app(_app_config(_config()))

    with TestClient(app) as client:
        _record(broker, _batch(1))

        def _listed() -> bool:
            return client.get("/directory/keys").json()["total"] == 1

        assert _wait_until(_listed)
        [listed] = client.get("/directory/keys").json()["keys"]
        assert [p["instance_id"] for p in listed["placements"]] == ["node-a"]
        # Exactly one door: the HTTP push endpoint is closed under Kafka.
        pushed = client.post(
            "/events",
            json=CacheEventsRequest(batches=[_batch(2)]).model_dump(mode="json"),
        )
        assert pushed.status_code == 404
        assert "--event-transport kafka" in pushed.json()["detail"]
        assert client.get("/directory/keys").json()["total"] == 1

    [kafka_consumer] = kafka_consumers
    assert kafka_consumer.subscribed == (_TOPIC,)
    assert kafka_consumer.closed is True


def test_app_startup_blocks_until_the_backlog_is_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backlog already on the topic before the app exists is fully
    applied by the time startup finishes -- nothing here polls or waits
    afterward, unlike the test above."""
    broker = FakeKafkaBroker()
    _record(broker, _batch(1))
    _record(broker, _batch(2))
    install_fake_confluent_kafka(
        monkeypatch,
        consumer_factory=lambda config: FakeKafkaConsumer(broker, config),
    )

    with TestClient(create_app(_app_config(_config()))) as client:
        assert client.get("/directory/keys").json()["total"] == 2


def test_app_defaults_to_http_source_and_builds_no_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected(config: dict[str, str | int | bool]) -> FakeKafkaConsumer:
        raise AssertionError(f"no Kafka consumer expected, got config {config}")

    install_fake_confluent_kafka(monkeypatch, consumer_factory=_unexpected)

    with TestClient(create_app(_app_config(HttpCacheEventSourceConfig()))) as client:
        pushed = client.post(
            "/events",
            json=CacheEventsRequest(batches=[_batch(1)]).model_dump(mode="json"),
        )
        assert pushed.status_code == 200
        assert client.get("/directory/keys").json()["total"] == 1
