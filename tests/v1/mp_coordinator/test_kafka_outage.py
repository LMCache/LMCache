# SPDX-License-Identifier: Apache-2.0
"""A Kafka broker outage between an MP server and the coordinator (MP-5).

The server's half is real: its ``CacheEventSubscriber`` stamps the sequence
numbers and its ``KafkaCacheEventSink`` publishes them. The coordinator's
half is real too: ``create_app`` with a Kafka event source. Only the broker
between them is the in-memory fake, which can go unreachable and come back.
Everything runs in one process, so this needs no Kafka in CI.
"""

# Standard
from collections.abc import Callable
import time

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventSubscriber,
    KafkaCacheEventSink,
)
from lmcache.v1.mp_coordinator.config import (
    KafkaCacheEventSourceConfig,
    MPCoordinatorConfig,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.config import KafkaCacheEventSinkConfig

# Local
from .fake_kafka import (
    FakeKafkaBroker,
    FakeKafkaConsumer,
    FakeKafkaProducer,
    install_fake_confluent_kafka,
)

_TOPIC = "cache-events"
_BOOTSTRAP_SERVERS = "broker-a:9092"
_WAIT_S = 5.0


def _key(n: int) -> ObjectKey:
    """One object key, distinguished by ``n``."""
    return ObjectKey(chunk_hash=bytes([n]) * 4, model_name="m", kv_rank=0)


def _store(subscriber: CacheEventSubscriber, n: int) -> None:
    """Have the server store one key in L2 and flush it as one batch.

    Args:
        subscriber: The server's subscriber.
        n: Which key.
    """
    event = Event(
        event_type=EventType.L2_KEYS_STORED,
        metadata={"keys": [_key(n)], "sizes": [100], "backend": "fs"},
    )
    subscriber.get_subscriptions()[event.event_type](event)
    subscriber.flush()


def _wait_until(condition: Callable[[], bool]) -> bool:
    """Poll ``condition`` until it holds or the wait runs out."""
    deadline = time.monotonic() + _WAIT_S
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def test_batches_sent_during_a_broker_outage_reach_the_coordinator_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MP-5: what the server publishes while the broker is down is held by
    its producer and delivered once the broker is back, in order and with
    the same sequence numbers. The coordinator ends with every store and no
    sequence gap."""
    broker = FakeKafkaBroker()
    producers: list[FakeKafkaProducer] = []

    def producer_factory(config: dict[str, str | int | bool]) -> FakeKafkaProducer:
        producers.append(FakeKafkaProducer(broker, config))
        return producers[-1]

    install_fake_confluent_kafka(
        monkeypatch,
        producer_factory=producer_factory,
        consumer_factory=lambda config: FakeKafkaConsumer(broker, config),
    )
    sink = KafkaCacheEventSink(
        KafkaCacheEventSinkConfig(bootstrap_servers=_BOOTSTRAP_SERVERS, topic=_TOPIC)
    )
    [producer] = producers
    subscriber = CacheEventSubscriber(
        sink=sink, instance_id="node-a", incarnation=7, flush_interval=3600.0
    )
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        event_source_config=KafkaCacheEventSourceConfig(
            bootstrap_servers=_BOOTSTRAP_SERVERS, topic=_TOPIC, group_id="coordinator"
        ),
    )

    with TestClient(create_app(config)) as client:

        def listed() -> int:
            return client.get("/directory/keys").json()["total"]

        _store(subscriber, 1)
        assert _wait_until(lambda: listed() == 1)

        producer.reachable = False
        _store(subscriber, 2)
        _store(subscriber, 3)
        assert len(broker.records(_TOPIC)) == 1
        assert listed() == 1

        producer.reachable = True
        _store(subscriber, 4)
        assert _wait_until(lambda: listed() == 4)

    sent = [
        CacheEventsRequest.model_validate_json(record.value or b"").batches[0]
        for record in broker.records(_TOPIC)
    ]
    assert [(b.seq, b.entries[0].key.to_object_key()) for b in sent] == [
        (n, _key(n)) for n in (1, 2, 3, 4)
    ]
    assert sink.dropped_batches == 0
