# SPDX-License-Identifier: Apache-2.0
"""In-memory stand-ins for the confluent-kafka surface LMCache uses.

Modelled: ``Producer.produce`` / ``flush`` with delivery callbacks, and a
single-reader ``Consumer`` (``subscribe`` / ``poll`` / ``store_offsets`` /
``position`` / ``get_watermark_offsets`` / ``close``) over one logical
partition per topic. Consumer groups, retention, and rebalancing are out
of scope; the single partition is assigned once, synchronously, in
:meth:`FakeKafkaConsumer.subscribe`. :func:`install_fake_confluent_kafka`
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

DeliveryCallback = Callable[[object | None, "FakeKafkaRecord"], None]
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
        offset: Meaning depends on context, exactly as the real type: an
            assignment offset, a queried position, or (from
            ``get_watermark_offsets``) unused.
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

    Records queue on :meth:`produce` and are delivered on :meth:`flush`, when
    every queued delivery callback fires, mirroring the real producer.

    Args:
        broker: Broker retaining produced records.
        config: Producer configuration, exposed for assertions.
        delivery_error: Error passed to every delivery callback.
        remaining_after_flush: Undelivered count :meth:`flush` reports.
        produce_error: Error :meth:`produce` raises instead of queueing.
    """

    def __init__(
        self,
        broker: FakeKafkaBroker,
        config: dict[str, str | int | bool],
        delivery_error: object | None = None,
        remaining_after_flush: int = 0,
        produce_error: Exception | None = None,
    ) -> None:
        self._broker = broker
        self._config = dict(config)
        self._delivery_error = delivery_error
        self._remaining_after_flush = remaining_after_flush
        self._produce_error = produce_error
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
        """Queue a record for delivery during :meth:`flush`.

        Args:
            topic: Destination topic.
            value: Record value.
            key: Record key.
            on_delivery: Callback notified during :meth:`flush`.

        Raises:
            Exception: The configured ``produce_error``, when set.
        """
        if self._produce_error is not None:
            raise self._produce_error
        self._pending.append((topic, key, value, on_delivery))

    def flush(self, timeout: float | None = None) -> int:
        """Deliver every queued record and fire its callback.

        Args:
            timeout: Accepted for producer API compatibility.

        Returns:
            The configured ``remaining_after_flush``; when it is non-zero the
            queue is left untouched, as a timed-out real flush would.
        """
        del timeout
        if self._remaining_after_flush:
            return self._remaining_after_flush
        pending, self._pending = self._pending, []
        for topic, key, value, callback in pending:
            if self._delivery_error is None:
                record = self._broker.append(topic=topic, key=key, value=value)
            else:
                record = FakeKafkaRecord(topic=topic, key=key, value=value, offset=-1)
            if callback is not None:
                callback(self._delivery_error, record)
        return 0


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
        self._stored_offsets: list[int] = []
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
    def stored_offsets(self) -> tuple[int, ...]:
        """Return the offsets passed to :meth:`store_offsets`, in call order."""
        return tuple(self._stored_offsets)

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

    def position(
        self, partitions: list[FakeTopicPartition]
    ) -> list[FakeTopicPartition]:
        """Return each partition's current read position.

        Args:
            partitions: Partitions to report; only ``.topic`` is read.

        Returns:
            One :class:`FakeTopicPartition` per input, ``.offset`` set to
            the next offset :meth:`poll` will read from that topic.
        """
        return [
            FakeTopicPartition(p.topic, offset=self._positions.get(p.topic, 0))
            for p in partitions
        ]

    def get_watermark_offsets(
        self,
        partition: FakeTopicPartition,
        timeout: float | None = None,
        cached: bool = False,
    ) -> tuple[int, int]:
        """Return ``(low, high)`` for one topic's single partition.

        Args:
            partition: The partition to measure; only ``.topic`` is read.
            timeout: Accepted for signature compatibility; unused --
                nothing here blocks.
            cached: Accepted for signature compatibility; unused -- this
                fake has no separate cached value to distinguish from a
                live query.

        Returns:
            ``(0, len(records))``: this fake never trims retention.
        """
        return (0, len(self._broker.records(partition.topic)))

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

    def store_offsets(self, message: FakeKafkaMessage) -> None:
        """Record ``message``'s offset as stored for the next commit.

        Args:
            message: The message whose offset the caller has finished with.
        """
        self._stored_offsets.append(message.offset())

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
    module.__dict__["OFFSET_INVALID"] = OFFSET_INVALID
    module.__dict__["TopicPartition"] = FakeTopicPartition
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
