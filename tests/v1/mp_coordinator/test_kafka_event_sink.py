# SPDX-License-Identifier: Apache-2.0
"""Tests for MP-server cache-event publication to Kafka."""

# Standard
from uuid import uuid4
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventPublishError,
    HttpCacheEventSink,
    KafkaCacheEventSink,
    create_cache_event_sink,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    KafkaCacheEventSinkConfig,
)
import lmcache.v1.mp_coordinator.cache_events as cache_events

# Local
from .fake_kafka import (
    FakeKafkaBroker,
    FakeKafkaException,
    FakeKafkaProducer,
    install_fake_confluent_kafka,
    uninstall_confluent_kafka,
)

_TOPIC = "cache-events"
_BOOTSTRAP_SERVERS = "broker-a:9092,broker-b:9092"
_REAL_KAFKA_BOOTSTRAP_ENV = "LMCACHE_TEST_KAFKA_BOOTSTRAP_SERVERS"


def _batch(instance_id: str, seq: int) -> CacheEventBatch:
    """Build one placement batch for transport tests.

    Args:
        instance_id: Emitter identity.
        seq: Emitter sequence number.

    Returns:
        One valid cache-event batch.
    """
    key = ObjectKey(
        chunk_hash=seq.to_bytes(4, "big"),
        model_name="model",
        kv_rank=0,
    )
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L1,
        backend="dram",
        entries=[
            CacheEventEntry(
                key=key.to_encoded_object_key(),
                size_bytes=1024,
            )
        ],
    )


def _config(delivery_timeout: float = 3.0) -> KafkaCacheEventSinkConfig:
    """Build a Kafka sink configuration for the fake broker.

    Args:
        delivery_timeout: Seconds the sink waits for broker acknowledgement.

    Returns:
        A valid configuration targeting ``_TOPIC``.
    """
    return KafkaCacheEventSinkConfig(
        bootstrap_servers=_BOOTSTRAP_SERVERS,
        topic=_TOPIC,
        delivery_timeout=delivery_timeout,
    )


def _sink(
    monkeypatch: pytest.MonkeyPatch,
    broker: FakeKafkaBroker,
    delivery_error: object | None = None,
    produce_error: Exception | None = None,
    max_queued: int | None = None,
) -> tuple[KafkaCacheEventSink, FakeKafkaProducer]:
    """Build a Kafka sink on top of the in-memory producer.

    Args:
        monkeypatch: Fixture scoping the ``confluent_kafka`` module swap.
        broker: Shared fake broker.
        delivery_error: Error the fake producer reports for every record.
        produce_error: Error the fake ``produce`` raises instead of queueing.
        max_queued: Records the fake buffer holds before ``BufferError``.

    Returns:
        The sink and its fake producer.
    """
    producer: FakeKafkaProducer | None = None

    def _producer_factory(
        config: dict[str, str | int | bool],
    ) -> FakeKafkaProducer:
        nonlocal producer
        producer = FakeKafkaProducer(
            broker,
            config,
            delivery_error=delivery_error,
            produce_error=produce_error,
            max_queued=max_queued,
        )
        return producer

    install_fake_confluent_kafka(monkeypatch, _producer_factory)
    sink = KafkaCacheEventSink(_config())
    if producer is None:
        raise RuntimeError("Kafka producer factory was not called")
    return sink, producer


def test_kafka_sink_publishes_ordered_keyed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(monkeypatch, broker)
    batches = [_batch("node-a", 1), _batch("node-a", 2)]

    sink.publish(batches)

    records = broker.records(_TOPIC)
    assert [record.offset for record in records] == [0, 1]
    for record, expected in zip(records, batches, strict=True):
        assert record.key == expected.instance_id.encode()
        assert record.value is not None
        envelope = CacheEventsRequest.model_validate_json(record.value)
        assert envelope.batches == [expected]


def test_kafka_sink_round_trips_against_real_broker() -> None:
    bootstrap_servers = os.environ.get(_REAL_KAFKA_BOOTSTRAP_ENV, "")
    if not bootstrap_servers:
        pytest.skip(f"set {_REAL_KAFKA_BOOTSTRAP_ENV} to test against Kafka")
    confluent_kafka = pytest.importorskip("confluent_kafka")
    kafka_admin = pytest.importorskip("confluent_kafka.admin")

    topic = f"lmcache-cache-events-{uuid4().hex}"
    admin = kafka_admin.AdminClient({"bootstrap.servers": bootstrap_servers})
    topic_result = admin.create_topics(
        [kafka_admin.NewTopic(topic=topic, num_partitions=1, replication_factor=1)]
    )[topic]
    topic_result.result(timeout=30.0)

    consumer = confluent_kafka.Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"lmcache-cache-events-{uuid4().hex}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    sink = KafkaCacheEventSink(
        KafkaCacheEventSinkConfig(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            delivery_timeout=10.0,
        )
    )
    batches = [_batch("node-a", 1), _batch("node-a", 2)]

    try:
        consumer.subscribe([topic])
        sink.publish(batches)

        received = 0
        deadline = time.monotonic() + 30.0
        while received < len(batches):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(f"received {received} of {len(batches)} Kafka records")
            message = consumer.poll(timeout=min(1.0, remaining))
            if message is None:
                continue
            if message.error() is not None:
                pytest.fail(f"Kafka consumer error: {message.error()}")
            expected = batches[received]
            assert message.key() == expected.instance_id.encode()
            assert message.partition() == 0
            assert message.offset() == received
            value = message.value()
            if value is None:
                pytest.fail("Kafka record value is missing")
            envelope = CacheEventsRequest.model_validate_json(value)
            assert envelope.batches == [expected]
            received += 1
    finally:
        sink.close()
        consumer.close()
        delete_result = admin.delete_topics([topic], operation_timeout=10.0)[topic]
        delete_result.result(timeout=30.0)


def test_kafka_sink_configures_durable_ordered_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, producer = _sink(monkeypatch, FakeKafkaBroker())

    assert producer.config == {
        "bootstrap.servers": _BOOTSTRAP_SERVERS,
        "client.id": "lmcache-cache-events",
        "enable.idempotence": True,
        "acks": "all",
        "message.timeout.ms": 3000,
        "queue.buffering.max.kbytes": 64 * 1024,
    }


def test_kafka_sink_requires_the_kafka_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninstall_confluent_kafka(monkeypatch)

    with pytest.raises(ImportError, match=r"pip install 'lmcache\[kafka\]'"):
        KafkaCacheEventSink(_config())


def test_kafka_sink_factory_uses_kafka_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    install_fake_confluent_kafka(
        monkeypatch, lambda config: FakeKafkaProducer(broker, config)
    )

    sink = create_cache_event_sink(CoordinatorConfig(event_sink_config=_config()))

    assert isinstance(sink, KafkaCacheEventSink)
    sink.publish([_batch("node-a", 1)])
    assert len(broker.records(_TOPIC)) == 1


def test_sink_factory_preserves_default_http_transport() -> None:
    sink = create_cache_event_sink(CoordinatorConfig(url="http://coordinator:9300"))

    assert isinstance(sink, HttpCacheEventSink)
    sink.close()


def test_http_sink_factory_requires_coordinator_url() -> None:
    with pytest.raises(ValueError, match="requires a coordinator URL"):
        create_cache_event_sink(CoordinatorConfig())


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the module logger's warnings; it does not propagate to root.

    Args:
        monkeypatch: Fixture scoping the logger patch.

    Returns:
        The list each formatted warning is appended to.
    """
    warnings: list[str] = []
    monkeypatch.setattr(
        cache_events.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    return warnings


def test_kafka_sink_publish_does_not_wait_for_an_unreachable_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer holds records through an outage and sends them in order."""
    broker = FakeKafkaBroker()
    sink, producer = _sink(monkeypatch, broker)
    batches = [_batch("node-a", seq) for seq in (1, 2, 3)]
    producer.reachable = False

    sink.publish(batches[:2])
    sink.publish(batches[2:])

    assert broker.records(_TOPIC) == ()
    assert producer.queued == 3

    producer.reachable = True
    sink.publish([_batch("node-a", 4)])

    delivered = [
        CacheEventsRequest.model_validate_json(record.value or b"").batches[0].seq
        for record in broker.records(_TOPIC)
    ]
    assert delivered == [1, 2, 3, 4]
    assert sink.dropped_batches == 0


def test_kafka_sink_counts_a_record_the_producer_gave_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed delivery report does not raise; it is counted and logged."""
    broker = FakeKafkaBroker()
    sink, _ = _sink(monkeypatch, broker, delivery_error=RuntimeError("timed out"))
    warnings = _capture_warnings(monkeypatch)

    sink.publish([_batch("node-a", 1)])

    assert broker.records(_TOPIC) == ()
    assert sink.dropped_batches == 1
    assert len(warnings) == 1
    assert "1 dropped so far" in warnings[0]
    assert "timed out" in warnings[0]


def test_kafka_sink_drops_what_does_not_fit_the_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full buffer refuses the rest of the call; what was queued still goes."""
    broker = FakeKafkaBroker()
    sink, producer = _sink(monkeypatch, broker, max_queued=1)
    producer.reachable = False

    with pytest.raises(CacheEventPublishError, match="refused 1 of 2"):
        sink.publish([_batch("node-a", 1), _batch("node-a", 2)])

    assert sink.dropped_batches == 1
    producer.reachable = True
    sink.close()
    delivered = [
        CacheEventsRequest.model_validate_json(record.value or b"").batches[0].seq
        for record in broker.records(_TOPIC)
    ]
    assert delivered == [1]


def test_kafka_sink_wraps_producer_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(
        monkeypatch, broker, produce_error=FakeKafkaException("unknown topic")
    )

    with pytest.raises(CacheEventPublishError, match="unknown topic"):
        sink.publish([_batch("node-a", 1)])
    assert broker.records(_TOPIC) == ()
    assert sink.dropped_batches == 1


def test_kafka_sink_close_delivers_queued_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, producer = _sink(monkeypatch, broker)
    producer.reachable = False
    sink.publish([_batch("node-a", 1)])
    producer.reachable = True

    sink.close()

    assert len(broker.records(_TOPIC)) == 1


def test_kafka_sink_close_warns_about_unacknowledged_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown flushes what it can and reports records the broker never acked."""
    sink, producer = _sink(monkeypatch, FakeKafkaBroker())
    producer.reachable = False
    sink.publish([_batch("node-a", 1), _batch("node-a", 2)])
    warnings = _capture_warnings(monkeypatch)

    sink.close()

    assert warnings == ["2 Kafka cache-event record(s) remained queued at shutdown"]
