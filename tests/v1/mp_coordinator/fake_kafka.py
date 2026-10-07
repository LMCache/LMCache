# SPDX-License-Identifier: Apache-2.0
"""In-memory stand-ins for the confluent-kafka surface LMCache uses.

Modelled: ``Producer.produce`` / ``poll`` / ``flush`` with delivery callbacks, and a
single-reader ``Consumer`` (``subscribe`` / ``assign`` / ``poll`` /
``close``) over one logical partition per topic. Consumer groups,
retention, and rebalancing are out of scope; the single partition is
assigned once, synchronously, in :meth:`FakeKafkaConsumer.subscribe`.
Offset commits are not modelled at all -- the coordinator does not
commit, because its checkpoint is its cursor. :func:`install_fake_confluent_kafka`
swaps the stand-ins in for the real module, so tests run without
``confluent-kafka`` installed.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import sys
import time
import types

# Third Party
import pytest

DeliveryCallback = Callable[[object | None, "FakeKafkaMessage"], None]
ProducerFactory = Callable[[dict[str, str | int | bool]], "FakeKafkaProducer"]
ConsumerFactory = Callable[[dict[str, str | int | bool]], "FakeKafkaConsumer"]
AssignCallback = Callable[[object, list["FakeTopicPartition"]], None]

OFFSET_INVALID = -1001
"""Stands in for ``confluent_kafka.OFFSET_INVALID``."""


@dataclass
class FakeTopicPartition:
    """Stands in for ``confluent_kafka.TopicPartition``.

    Attributes:
        topic: The partition's topic.
        partition: The partition number (always ``0``: one logical
            partition per topic).
        offset: Where to resume the partition from, as
            :meth:`FakeKafkaConsumer.assign` reads it.
    """

    topic: str
    partition: int = 0
    offset: int = OFFSET_INVALID


class FakeKafkaException(Exception):
    """Stands in for ``confluent_kafka.KafkaException``."""


@dataclass(frozen=True)
class FakeKafkaRecord:
    """One record as the fake broker retained it.

    Attributes:
        topic: Destination topic.
        key: Record key.
        value: Record value.
        offset: Position in the topic, or ``-1`` when delivery failed.
    """

    topic: str
    key: bytes | None
    value: bytes | None
    offset: int


class FakeKafkaBroker:
    """Append-only per-topic record store shared by fake producers."""

    def __init__(self) -> None:
        self._records: dict[str, list[FakeKafkaRecord]] = {}

    def append(
        self, topic: str, key: bytes | None, value: bytes | None
    ) -> FakeKafkaRecord:
        """Retain a record at the topic's next offset.

        Args:
            topic: Destination topic.
            key: Record key.
            value: Record value.

        Returns:
            The retained record with its assigned offset.
        """
        records = self._records.setdefault(topic, [])
        record = FakeKafkaRecord(topic=topic, key=key, value=value, offset=len(records))
        records.append(record)
        return record

    def records(self, topic: str) -> tuple[FakeKafkaRecord, ...]:
        """Return one topic's retained records in offset order.

        Args:
            topic: Topic to inspect.

        Returns:
            The records, empty when nothing was retained.
        """
        return tuple(self._records.get(topic, ()))


class FakeKafkaProducer:
    """Producer double backed by :class:`FakeKafkaBroker`.

    Records queue on :meth:`produce` and are delivered, callbacks fired, on
    :meth:`poll` or :meth:`flush`. While :attr:`reachable` is false they stay
    queued.

    Args:
        broker: Broker retaining produced records.
        config: Producer configuration, exposed for assertions.
        delivery_error: Error passed to every delivery callback.
        produce_error: Error :meth:`produce` raises instead of queueing.
        max_queued: Buffer size in records past which :meth:`produce` raises
            ``BufferError``; ``None`` is unbounded.
    """

    def __init__(
        self,
        broker: FakeKafkaBroker,
        config: dict[str, str | int | bool],
        delivery_error: object | None = None,
        produce_error: Exception | None = None,
        max_queued: int | None = None,
    ) -> None:
        self._broker = broker
        self._max_queued = max_queued
        self._config = dict(config)
        self._delivery_error = delivery_error
        self._produce_error = produce_error
        self.reachable = True
        self._pending: list[
            tuple[str, bytes | None, bytes | None, DeliveryCallback | None]
        ] = []

    @property
    def config(self) -> dict[str, str | int | bool]:
        """Return a copy of the producer configuration."""
        return dict(self._config)

    def produce(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        on_delivery: DeliveryCallback | None = None,
    ) -> None:
        """Queue a record for delivery during :meth:`poll` or :meth:`flush`.

        Args:
            topic: Destination topic.
            value: Record value.
            key: Record key.
            on_delivery: Callback notified when the record is delivered.

        Raises:
            Exception: The configured ``produce_error``, when set.
            BufferError: The buffer already holds ``max_queued`` records.
        """
        if self._produce_error is not None:
            raise self._produce_error
        if self._max_queued is not None and len(self._pending) >= self._max_queued:
            raise BufferError("Local: Queue full")
        self._pending.append((topic, key, value, on_delivery))

    @property
    def queued(self) -> int:
        """Return how many records wait for delivery."""
        return len(self._pending)

    def poll(self, timeout: float | None = None) -> int:
        """Deliver every queued record and fire its callback.

        Args:
            timeout: Accepted for producer API compatibility.

        Returns:
            The number of callbacks fired.
        """
        del timeout
        return self._deliver()

    def flush(self, timeout: float | None = None) -> int:
        """Deliver every queued record and fire its callback.

        Args:
            timeout: Accepted for producer API compatibility.

        Returns:
            How many records are still queued.
        """
        del timeout
        self._deliver()
        return len(self._pending)

    def _deliver(self) -> int:
        """Deliver the queue in order while the broker is reachable.

        Returns:
            The number of callbacks fired.
        """
        if not self.reachable:
            return 0
        pending, self._pending = self._pending, []
        for topic, key, value, callback in pending:
            if self._delivery_error is None:
                record = self._broker.append(topic=topic, key=key, value=value)
            else:
                record = FakeKafkaRecord(topic=topic, key=key, value=value, offset=-1)
            if callback is not None:
                callback(self._delivery_error, FakeKafkaMessage(record))
        return len(pending)


class FakeKafkaMessage:
    """Consumer-side view of a record with the ``confluent_kafka.Message``
    accessors the coordinator source reads.

    Args:
        record: The record read from the broker.
        error: Error surfaced by :meth:`error`; the record then only carries
            the topic, as a real error message does.
    """

    def __init__(self, record: FakeKafkaRecord, error: object | None = None) -> None:
        self._record = record
        self._error = error

    def error(self) -> object | None:
        """Return the consumer error this message carries, if any."""
        return self._error

    def topic(self) -> str:
        """Return the record topic."""
        return self._record.topic

    def partition(self) -> int:
        """Return the logical partition (always ``0``)."""
        return 0

    def offset(self) -> int:
        """Return the record offset."""
        return self._record.offset

    def key(self) -> bytes | None:
        """Return the record key."""
        return self._record.key

    def value(self) -> bytes | None:
        """Return the record value."""
        return self._record.value


class FakeKafkaConsumer:
    """Single-reader consumer double over a :class:`FakeKafkaBroker`.

    :meth:`poll` returns injected errors first, then unread records of the
    subscribed topics in offset order, then ``None`` after a short sleep so a
    polling thread does not spin.

    Args:
        broker: Broker whose records are read.
        config: Consumer configuration, exposed for assertions.
    """

    def __init__(
        self, broker: FakeKafkaBroker, config: dict[str, str | int | bool]
    ) -> None:
        self._broker = broker
        self._config = dict(config)
        self._topics: list[str] = []
        self._positions: dict[str, int] = {}
        self._errors: list[object] = []
        self._closed = False

    @property
    def config(self) -> dict[str, str | int | bool]:
        """Return a copy of the consumer configuration."""
        return dict(self._config)

    @property
    def subscribed(self) -> tuple[str, ...]:
        """Return the topics passed to :meth:`subscribe`."""
        return tuple(self._topics)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` was called."""
        return self._closed

    def inject_error(self, error: object) -> None:
        """Queue ``error`` to be surfaced as the next polled message.

        Args:
            error: Object returned by that message's ``error()``.
        """
        self._errors.append(error)

    def subscribe(
        self, topics: list[str], on_assign: "AssignCallback | None" = None
    ) -> None:
        """Subscribe to ``topics``, polled in list order.

        Args:
            topics: Topics to read from their first record.
            on_assign: If given, called once, synchronously, with the
                single partition assigned for each topic -- this fake
                never rebalances, so there is only ever one assignment,
                and it never happens later on ``poll()`` the way a real
                consumer's does.
        """
        self._topics = list(topics)
        if on_assign is not None:
            partitions = [
                FakeTopicPartition(topic, offset=self._positions.get(topic, 0))
                for topic in self._topics
            ]
            on_assign(self, partitions)

    def assign(self, partitions: list[FakeTopicPartition]) -> None:
        """Seek each partition to the offset given, as a real consumer's
        ``assign`` does when called from an ``on_assign`` callback.

        Args:
            partitions: Partitions with the position to resume each from.
        """
        for partition in partitions:
            self._positions[partition.topic] = partition.offset

    def poll(self, timeout: float | None = None) -> FakeKafkaMessage | None:
        """Return the next error or unread record, or ``None`` when caught up.

        Args:
            timeout: Upper bound (seconds) on the sleep taken when caught up.

        Returns:
            The next message, or ``None`` when every subscribed topic is read.
        """
        if self._errors:
            topic = self._topics[0] if self._topics else ""
            placeholder = FakeKafkaRecord(topic=topic, key=None, value=None, offset=-1)
            return FakeKafkaMessage(placeholder, error=self._errors.pop(0))
        for topic in self._topics:
            records = self._broker.records(topic)
            position = self._positions.get(topic, 0)
            if position < len(records):
                self._positions[topic] = position + 1
                return FakeKafkaMessage(records[position])
        time.sleep(min(timeout or 0.0, 0.005))
        return None

    def close(self) -> None:
        """Mark the consumer closed."""
        self._closed = True


def install_fake_confluent_kafka(
    monkeypatch: pytest.MonkeyPatch,
    producer_factory: ProducerFactory | None = None,
    consumer_factory: ConsumerFactory | None = None,
) -> None:
    """Make ``import confluent_kafka`` resolve to the in-memory stand-ins.

    Args:
        monkeypatch: Fixture that undoes the module swap after the test.
        producer_factory: Called in place of ``confluent_kafka.Producer``.
        consumer_factory: Called in place of ``confluent_kafka.Consumer``.
    """
    module = types.ModuleType("confluent_kafka")
    module.__dict__["KafkaException"] = FakeKafkaException
    if producer_factory is not None:
        module.__dict__["Producer"] = producer_factory
    if consumer_factory is not None:
        module.__dict__["Consumer"] = consumer_factory
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)


def uninstall_confluent_kafka(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import confluent_kafka`` fail, as without the ``kafka`` extra.

    Args:
        monkeypatch: Fixture that undoes the block after the test.
    """
    # ``None`` in ``sys.modules`` is Python's documented way to block an import.
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)  # type: ignore[arg-type]
