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
    remaining_after_flush: int = 0,
    produce_error: Exception | None = None,
) -> tuple[KafkaCacheEventSink, FakeKafkaProducer]:
    """Build a Kafka sink on top of the in-memory producer.

    Args:
        monkeypatch: Fixture scoping the ``confluent_kafka`` module swap.
        broker: Shared fake broker.
        delivery_error: Error the fake producer reports for every record.
        remaining_after_flush: Undelivered count the fake ``flush`` returns.
        produce_error: Error the fake ``produce`` raises instead of queueing.

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
            remaining_after_flush=remaining_after_flush,
            produce_error=produce_error,
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


def test_kafka_sink_raises_on_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(monkeypatch, broker, delivery_error=RuntimeError("delivery failed"))

    with pytest.raises(CacheEventPublishError, match="rejected 1 of 1"):
        sink.publish([_batch("node-a", 1)])
    assert broker.records(_TOPIC) == ()


def test_kafka_sink_wraps_producer_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(monkeypatch, broker, produce_error=BufferError("queue full"))

    with pytest.raises(CacheEventPublishError, match="queue full"):
        sink.publish([_batch("node-a", 1)])
    assert broker.records(_TOPIC) == ()


def test_kafka_sink_raises_when_flush_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(monkeypatch, broker, remaining_after_flush=1)

    with pytest.raises(CacheEventPublishError, match="not acknowledged"):
        sink.publish([_batch("node-a", 1)])
    assert broker.records(_TOPIC) == ()


def test_kafka_sink_close_warns_about_unacknowledged_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown flushes what it can and reports records the broker never acked."""
    sink, _ = _sink(monkeypatch, FakeKafkaBroker(), remaining_after_flush=2)
    warnings: list[str] = []
    # The module logger does not propagate (see ``lmcache.logging``), so
    # record the call instead of relying on root-handler capture.
    monkeypatch.setattr(
        cache_events.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    sink.close()

    assert warnings == ["2 Kafka cache-event record(s) remained queued at shutdown"]
