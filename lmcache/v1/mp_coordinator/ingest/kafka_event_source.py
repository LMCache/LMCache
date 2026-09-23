# SPDX-License-Identifier: Apache-2.0
"""Kafka pull implementation of the coordinator cache-event source.

Consumes the records a ``KafkaCacheEventSink`` produces -- one
``CacheEventsRequest`` envelope per record, keyed by ``instance_id`` -- and
offers each record's batches to :class:`EventGate` in partition order. Kafka retains
the stream, so this is the coordinator's durable door; the gate's per-emitter
sequence cursors stay separate from Kafka's partition offsets.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from typing import TYPE_CHECKING
import asyncio
import threading

# Third Party
from pydantic import ValidationError

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.config import KafkaCacheEventSourceConfig
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.event_source import (
    CacheEventSource,
    CacheEventSourceStatus,
    EventReplayCapability,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest

if TYPE_CHECKING:
    # Third Party
    from confluent_kafka import Message

logger = init_logger(__name__)

# Bounds how long stop() waits for the poll loop to notice the stop flag.
_POLL_TIMEOUT_SECONDS = 1.0


class KafkaCacheEventSource(CacheEventSource):
    """Durable pull source that consumes cache events from a Kafka topic.

    One thread polls the consumer and offers each record's batches to the
    gate, so a partition's records -- one instance's stream -- arrive in
    order. A record's offset is stored only after the gate has seen it and is
    committed by the consumer's auto-commit, so delivery is at-least-once;
    the gate's sequence dedup makes redelivery harmless. A record that cannot
    be decoded, or that makes a consumer raise, is logged and skipped so one
    bad record cannot stall its partition.

    ``confluent-kafka`` is the optional ``lmcache[kafka]`` extra and is
    imported only here.

    Args:
        event_gate: Admission authority for consumed event batches.
        config: Validated broker, topic, and consumer-group settings.

    Raises:
        ImportError: If ``confluent-kafka`` is not installed.
    """

    def __init__(
        self, event_gate: EventGate, config: KafkaCacheEventSourceConfig
    ) -> None:
        try:
            # Third Party
            from confluent_kafka import Consumer
        except ImportError as e:
            raise ImportError(
                "The Kafka cache-event source needs confluent-kafka: "
                "pip install 'lmcache[kafka]'"
            ) from e
        self._event_gate = event_gate
        self._topic = config.topic
        self._consumer = Consumer(
            {
                "bootstrap.servers": config.bootstrap_servers,
                "group.id": config.group_id,
                "client.id": "lmcache-coordinator",
                "auto.offset.reset": "earliest",
                # Auto-commit only the offsets the poll loop stored after the
                # gate saw the record, never merely-polled ones.
                "enable.auto.commit": True,
                "enable.auto.offset.store": False,
            }
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._consume_until_stopped,
            name="lmcache-kafka-cache-events",
            daemon=True,
        )

    async def start(self) -> None:
        """Subscribe to the topic and start the poll thread."""
        self._consumer.subscribe([self._topic])
        self._thread.start()

    async def stop(self) -> None:
        """Stop the poll thread, then close the consumer.

        Closing commits the offsets the loop stored, so a restart resumes
        after the last record the gate saw.
        """
        self._stop.set()
        if self._thread.is_alive():
            await asyncio.to_thread(self._thread.join)
        self._consumer.close()

    def status(self) -> CacheEventSourceStatus:
        """Return the Kafka source's status.

        Returns:
            Source identity ``kafka`` with replay capability ``SEEKABLE``:
            the topic retains the stream, so it can be replayed by resetting
            the consumer group's offsets.
        """
        return CacheEventSourceStatus(
            source_name="kafka",
            replay_capability=EventReplayCapability.SEEKABLE,
        )

    # -- Poll thread -----------------------------------------------------------

    def _consume_until_stopped(self) -> None:
        """Poll until :meth:`stop`, offering every record to the gate in order."""
        while not self._stop.is_set():
            message = self._consumer.poll(_POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            error = message.error()
            if error is not None:
                logger.warning(
                    "Kafka cache-event consumer error on topic %s: %s",
                    self._topic,
                    error,
                )
                continue
            self._ingest(message)
            self._consumer.store_offsets(message=message)

    def _ingest(self, message: "Message") -> None:
        """Decode one record and offer its batches to the gate.

        Decode failures and consumer exceptions are logged and swallowed; the
        caller stores the offset either way so the partition keeps moving.
        """
        position = f"{message.topic()}[{message.partition()}]@{message.offset()}"
        try:
            batches = CacheEventsRequest.model_validate_json(
                message.value() or b""
            ).batches
        except ValidationError as e:
            logger.warning(
                "Dropping undecodable cache-event record %s: %s", position, e
            )
            return
        try:
            summary = self._event_gate.ingest_batches(batches)
        except Exception:
            logger.exception("Cache-event consumers failed on record %s", position)
            return
        logger.debug(
            "Kafka cache-event record %s: %d applied, %d duplicate, %d stale",
            position,
            summary.applied,
            summary.duplicates,
            summary.stale,
        )
