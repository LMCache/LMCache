# SPDX-License-Identifier: Apache-2.0
"""In-memory stand-ins for the confluent-kafka producer surface the sink uses.

Only ``Producer.produce`` / ``Producer.flush`` and their delivery callbacks
are modelled. Consumer groups, retention, and rebalancing are out of scope.
:func:`install_fake_confluent_kafka` swaps the stand-ins in for the real
module, so the sink tests run without ``confluent-kafka`` installed.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import sys
import types

# Third Party
import pytest

DeliveryCallback = Callable[[object | None, "FakeKafkaRecord"], None]
ProducerFactory = Callable[[dict[str, str | int | bool]], "FakeKafkaProducer"]


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


def install_fake_confluent_kafka(
    monkeypatch: pytest.MonkeyPatch, producer_factory: ProducerFactory
) -> None:
    """Make ``import confluent_kafka`` resolve to the in-memory stand-ins.

    Args:
        monkeypatch: Fixture that undoes the module swap after the test.
        producer_factory: Called in place of ``confluent_kafka.Producer``.
    """
    module = types.ModuleType("confluent_kafka")
    module.__dict__.update(Producer=producer_factory, KafkaException=FakeKafkaException)
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)


def uninstall_confluent_kafka(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import confluent_kafka`` fail, as without the ``kafka`` extra.

    Args:
        monkeypatch: Fixture that undoes the block after the test.
    """
    # ``None`` in ``sys.modules`` is Python's documented way to block an import.
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)  # type: ignore[arg-type]
