# SPDX-License-Identifier: Apache-2.0
"""Small in-memory Kafka test double for cache-event transport tests.

This models only the producer/consumer surface LMCache uses. It is not a Kafka
protocol implementation and deliberately omits consumer groups, retention,
rebalance, seeking, and replay; those can be added when the coordinator source
needs them.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass

KafkaDeliveryCallback = Callable[[object | None, "FakeKafkaMessage"], None]


@dataclass(frozen=True)
class FakeKafkaMessage:
    """One retained record or failed delivery-report message.

    Attributes:
        topic_name: Topic containing the record.
        message_key: Record key.
        message_value: Record value.
        partition_id: Logical partition number.
        message_offset: Monotonic partition offset, or ``-1`` when delivery
            failed before retention.
    """

    topic_name: str
    message_key: bytes | None
    message_value: bytes | None
    partition_id: int
    message_offset: int

    def topic(self) -> str:
        """Return the record topic."""
        return self.topic_name

    def key(self) -> bytes | None:
        """Return the record key."""
        return self.message_key

    def value(self) -> bytes | None:
        """Return the record value."""
        return self.message_value

    def partition(self) -> int:
        """Return the logical partition."""
        return self.partition_id

    def offset(self) -> int:
        """Return the partition offset."""
        return self.message_offset


class FakeKafkaBroker:
    """In-memory append-only topic store shared by fake clients."""

    def __init__(self) -> None:
        self._messages: dict[str, list[FakeKafkaMessage]] = {}

    def append(
        self,
        topic: str,
        key: bytes | None,
        value: bytes | None,
    ) -> FakeKafkaMessage:
        """Append a record to the topic's single logical partition.

        Args:
            topic: Destination topic.
            key: Record key.
            value: Record value.

        Returns:
            The retained record with its assigned offset.
        """
        messages = self._messages.setdefault(topic, [])
        message = FakeKafkaMessage(
            topic_name=topic,
            message_key=key,
            message_value=value,
            partition_id=0,
            message_offset=len(messages),
        )
        messages.append(message)
        return message

    def messages(self, topic: str) -> tuple[FakeKafkaMessage, ...]:
        """Return an immutable snapshot of one topic's records.

        Args:
            topic: Topic to inspect.

        Returns:
            Records in offset order.
        """
        return tuple(self._messages.get(topic, ()))


class FakeKafkaProducer:
    """Synchronous producer double backed by :class:`FakeKafkaBroker`.

    Args:
        broker: Broker retaining produced records.
        config: Producer configuration to expose to assertions.
        delivery_error: Error passed to every delivery callback.
        remaining_after_flush: Undelivered count returned by :meth:`flush`.
        produce_error: Error raised instead of queueing a record.
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
        self._pending_records: list[
            tuple[
                str,
                bytes | None,
                bytes | None,
                KafkaDeliveryCallback | None,
            ]
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
        on_delivery: KafkaDeliveryCallback | None = None,
    ) -> None:
        """Queue a record for delivery during ``flush``.

        Args:
            topic: Destination topic.
            value: Record value.
            key: Record key.
            on_delivery: Callback notified during :meth:`flush`.

        Raises:
            Exception: The configured producer failure, when present.
        """
        if self._produce_error is not None:
            raise self._produce_error
        self._pending_records.append((topic, key, value, on_delivery))

    def flush(self, timeout: float | None = None) -> int:
        """Complete every queued delivery.

        Args:
            timeout: Accepted for producer API compatibility.

        Returns:
            The configured undelivered count, or zero after completing all
            queued records.
        """
        del timeout
        if self._remaining_after_flush:
            return self._remaining_after_flush
        records = self._pending_records
        self._pending_records = []
        for topic, key, value, callback in records:
            if self._delivery_error is None:
                message = self._broker.append(topic=topic, key=key, value=value)
            else:
                message = FakeKafkaMessage(
                    topic_name=topic,
                    message_key=key,
                    message_value=value,
                    partition_id=0,
                    message_offset=-1,
                )
            if callback is not None:
                callback(self._delivery_error, message)
        return 0


class FakeKafkaConsumer:
    """Single-consumer reader for records retained by the fake broker."""

    def __init__(self, broker: FakeKafkaBroker) -> None:
        self._broker = broker
        self._topics: list[str] = []
        self._positions: dict[str, int] = {}

    def subscribe(self, topics: list[str]) -> None:
        """Subscribe to topics in deterministic polling order.

        Args:
            topics: Topics to read.
        """
        self._topics = list(topics)
        for topic in topics:
            self._positions.setdefault(topic, 0)

    def poll(self, timeout: float | None = None) -> FakeKafkaMessage | None:
        """Return the next retained record, or ``None`` when exhausted.

        Args:
            timeout: Accepted for consumer API compatibility.

        Returns:
            The next message in subscription order, or ``None``.
        """
        del timeout
        for topic in self._topics:
            position = self._positions[topic]
            messages = self._broker.messages(topic)
            if position >= len(messages):
                continue
            self._positions[topic] = position + 1
            return messages[position]
        return None
