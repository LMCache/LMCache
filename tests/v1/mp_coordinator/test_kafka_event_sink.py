# SPDX-License-Identifier: Apache-2.0
"""Tests for MP-server cache-event publication to Kafka."""

# Standard
from unittest.mock import patch

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

# Local
from .fake_kafka import FakeKafkaBroker, FakeKafkaConsumer, FakeKafkaProducer

_TOPIC = "cache-events"


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


def _sink(
    broker: FakeKafkaBroker,
) -> tuple[KafkaCacheEventSink, FakeKafkaProducer]:
    """Build a Kafka sink backed by the in-memory producer.

    Args:
        broker: Shared fake broker.

    Returns:
        The sink and its fake producer.
    """
    producer: FakeKafkaProducer | None = None

    def _producer_factory(
        config: dict[str, str | int | bool],
    ) -> FakeKafkaProducer:
        nonlocal producer
        producer = FakeKafkaProducer(broker, config)
        return producer

    with patch(
        "lmcache.v1.mp_coordinator.cache_events.Producer",
        side_effect=_producer_factory,
    ):
        sink = KafkaCacheEventSink(
            bootstrap_servers="broker-a:9092,broker-b:9092",
            topic=_TOPIC,
            delivery_timeout=3.0,
        )
    if producer is None:
        raise RuntimeError("Kafka producer factory was not called")
    return sink, producer


def test_kafka_sink_round_trips_ordered_keyed_records() -> None:
    broker = FakeKafkaBroker()
    sink, _ = _sink(broker)
    batches = [
        _batch("node-a", 1),
        _batch("node-a", 2),
    ]

    sink.publish(batches)

    consumer = FakeKafkaConsumer(broker)
    consumer.subscribe([_TOPIC])
    messages = [consumer.poll(), consumer.poll()]
    assert consumer.poll() is None
    for offset, (message, expected) in enumerate(zip(messages, batches, strict=True)):
        assert message is not None
        assert message.key() == expected.instance_id.encode()
        assert message.partition() == 0
        assert message.offset() == offset
        value = message.value()
        assert value is not None
        envelope = CacheEventsRequest.model_validate_json(value)
        assert envelope.batches == [expected]


def test_kafka_sink_configures_durable_ordered_producer() -> None:
    sink, producer = _sink(FakeKafkaBroker())

    assert isinstance(sink, KafkaCacheEventSink)
    assert producer.config == {
        "bootstrap.servers": "broker-a:9092,broker-b:9092",
        "client.id": "lmcache-cache-events",
        "enable.idempotence": True,
        "acks": "all",
        "message.timeout.ms": 3000,
    }


def test_kafka_sink_factory_uses_kafka_config() -> None:
    broker = FakeKafkaBroker()

    with patch(
        "lmcache.v1.mp_coordinator.cache_events.Producer",
        side_effect=lambda config: FakeKafkaProducer(broker, config),
    ):
        sink = create_cache_event_sink(
            CoordinatorConfig(
                event_sink_config=KafkaCacheEventSinkConfig(
                    bootstrap_servers="broker:9092",
                    topic=_TOPIC,
                ),
            )
        )

    assert isinstance(sink, KafkaCacheEventSink)
    sink.publish([_batch("node-a", 1)])
    assert len(broker.messages(_TOPIC)) == 1


def test_sink_factory_preserves_default_http_transport() -> None:
    sink = create_cache_event_sink(CoordinatorConfig(url="http://coordinator:9300"))

    assert isinstance(sink, HttpCacheEventSink)
    sink.close()


def test_http_sink_factory_requires_coordinator_url() -> None:
    with pytest.raises(ValueError, match="requires a coordinator URL"):
        create_cache_event_sink(CoordinatorConfig())


def test_kafka_sink_validates_direct_configuration() -> None:
    with pytest.raises(ValueError, match="bootstrap servers must be non-empty"):
        KafkaCacheEventSink(
            bootstrap_servers="",
            topic=_TOPIC,
            delivery_timeout=1.0,
        )
    with pytest.raises(ValueError, match="topic must be non-empty"):
        KafkaCacheEventSink(
            bootstrap_servers="broker:9092",
            topic="",
            delivery_timeout=1.0,
        )
    with pytest.raises(ValueError, match="delivery timeout must be a finite"):
        KafkaCacheEventSink(
            bootstrap_servers="broker:9092",
            topic=_TOPIC,
            delivery_timeout=0,
        )


def test_kafka_sink_raises_on_delivery_failure() -> None:
    broker = FakeKafkaBroker()

    with patch(
        "lmcache.v1.mp_coordinator.cache_events.Producer",
        side_effect=lambda config: FakeKafkaProducer(
            broker,
            config,
            delivery_error=RuntimeError("delivery failed"),
        ),
    ):
        sink = KafkaCacheEventSink(
            bootstrap_servers="broker:9092",
            topic=_TOPIC,
            delivery_timeout=1.0,
        )

    with pytest.raises(CacheEventPublishError, match="rejected 1 of 1"):
        sink.publish([_batch("node-a", 1)])
    assert broker.messages(_TOPIC) == ()


def test_kafka_sink_wraps_producer_enqueue_failure() -> None:
    broker = FakeKafkaBroker()

    with patch(
        "lmcache.v1.mp_coordinator.cache_events.Producer",
        side_effect=lambda config: FakeKafkaProducer(
            broker,
            config,
            produce_error=BufferError("queue full"),
        ),
    ):
        sink = KafkaCacheEventSink(
            bootstrap_servers="broker:9092",
            topic=_TOPIC,
            delivery_timeout=1.0,
        )

    with pytest.raises(CacheEventPublishError, match="queue full"):
        sink.publish([_batch("node-a", 1)])
    assert broker.messages(_TOPIC) == ()


def test_kafka_sink_raises_when_flush_times_out() -> None:
    broker = FakeKafkaBroker()

    with patch(
        "lmcache.v1.mp_coordinator.cache_events.Producer",
        side_effect=lambda config: FakeKafkaProducer(
            broker,
            config,
            remaining_after_flush=1,
        ),
    ):
        sink = KafkaCacheEventSink(
            bootstrap_servers="broker:9092",
            topic=_TOPIC,
            delivery_timeout=1.0,
        )

    with pytest.raises(CacheEventPublishError, match="not acknowledged"):
        sink.publish([_batch("node-a", 1)])
    assert broker.messages(_TOPIC) == ()
