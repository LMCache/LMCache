# SPDX-License-Identifier: Apache-2.0
"""Kafka pull implementation of the coordinator cache-event source.

Consumes the records a ``KafkaCacheEventSink`` produces -- one
``CacheEventsRequest`` envelope per record, keyed by ``instance_id`` -- and
offers each record's batches to :class:`EventGate` in partition order. Kafka retains
the stream, so this is the coordinator's durable door; the gate's per-emitter
sequence cursors stay separate from Kafka's partition offsets.

A restart resumes from :class:`~.stream_position.StreamPosition` -- the
checkpoint's own record of how far it got -- rather than from the
consumer group's committed offset: the two are saved on independent
timers, so the committed offset can be ahead of the last saved
checkpoint, and resuming from it would skip the records in between for
good. A partition :class:`StreamPosition` has never seen (no checkpoint
yet, or a new partition) falls back to the group's committed offset, or
``auto.offset.reset`` if it has none either.

That resumption can still mean a real backlog to read (a coordinator
down for a while, replaying from a checkpoint that is properly behind
but far behind): :meth:`status` reports how many records the assigned
partitions are still behind by, and
:class:`~.readiness.IngestReadiness` is what turns that into a verdict a
controller can act on.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from typing import TYPE_CHECKING
import asyncio
import threading
import time

# Third Party
from pydantic import ValidationError

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.config import KafkaCacheEventSourceConfig
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.event_source import (
    UNKNOWN_LAG,
    CacheEventSource,
    CacheEventSourceStatus,
    EventReplayCapability,
)
from lmcache.v1.mp_coordinator.ingest.stream_position import StreamPosition
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest

if TYPE_CHECKING:
    # Third Party
    from confluent_kafka import Consumer, Message, TopicPartition

logger = init_logger(__name__)

# Bounds how long stop() waits for the poll loop to notice the stop flag.
_POLL_TIMEOUT_SECONDS = 1.0

# Seconds between broker watermark queries for the cached lag. Lag only
# gates readiness and is reported to operators, so it is worth one round
# trip a second and not one per record.
_WATERMARK_INTERVAL = 1.0


class KafkaCacheEventSource(CacheEventSource):
    """Durable pull source that consumes cache events from a Kafka topic.

    One thread polls the consumer and offers each record's batches to the
    gate, so a partition's records -- one instance's stream -- arrive in
    order. A record's offset is recorded into ``position`` (and stored for
    the consumer group's own auto-commit) only after the gate has seen
    it, so delivery is at-least-once; the gate's sequence dedup makes
    redelivery harmless. A record that cannot be decoded, or that makes a
    consumer raise, is logged and skipped -- and still recorded -- so one
    bad record cannot stall its partition.

    ``confluent-kafka`` is the optional ``lmcache[kafka]`` extra and is
    imported only here.

    Args:
        event_gate: Admission authority for consumed event batches.
        config: Validated broker, topic, and consumer-group settings.
        position: Where the last-saved checkpoint reaches; seeked to on
            assignment, and recorded into as records are admitted, so a
            future checkpoint captures a position consistent with it.

    Raises:
        ImportError: If ``confluent-kafka`` is not installed.
    """

    def __init__(
        self,
        event_gate: EventGate,
        config: KafkaCacheEventSourceConfig,
        position: StreamPosition,
    ) -> None:
        try:
            # Third Party
            from confluent_kafka import OFFSET_INVALID, Consumer
        except ImportError as e:
            raise ImportError(
                "The Kafka cache-event source needs confluent-kafka: "
                "pip install 'lmcache[kafka]'"
            ) from e
        self._event_gate = event_gate
        self._topic = config.topic
        self._position = position
        self._offset_invalid = OFFSET_INVALID
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
        # Written by the poll thread, read by request threads calling
        # status(); the lock is the only thing shared between them.
        self._lock = threading.Lock()
        self._assignment: list[TopicPartition] = []
        self._lag = UNKNOWN_LAG

    async def start(self) -> None:
        """Subscribe to the topic and start the poll thread."""
        self._consumer.subscribe([self._topic], on_assign=self._on_assign)
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
            Source identity ``kafka`` with replay capability ``SEEKABLE``
            (the topic retains the stream, so it can be replayed by
            resetting the consumer group's offsets) and the cached lag --
            :data:`UNKNOWN_LAG` before the first watermark query resolves.
        """
        with self._lock:
            lag = self._lag
        return CacheEventSourceStatus(
            source_name="kafka",
            replay_capability=EventReplayCapability.SEEKABLE,
            lag=lag,
        )

    # -- Poll thread -----------------------------------------------------------

    def _on_assign(
        self, consumer: "Consumer", partitions: list["TopicPartition"]
    ) -> None:
        """Seek each assigned partition to the checkpoint's position.

        Runs on the poll thread, during ``poll()``. A partition
        :attr:`_position` has recorded an offset for is repositioned
        there; one it has never seen keeps whatever confluent-kafka
        assigned by default (the group's committed offset, or
        ``auto.offset.reset`` if the group has none either). Calling
        ``assign`` here is what makes the override take effect --
        without it, the default assignment stands.

        Args:
            consumer: The consumer being assigned to.
            partitions: The partitions assigned to this member.
        """
        for partition in partitions:
            next_offset = self._position.next_offset(
                partition.topic, partition.partition
            )
            if next_offset is not None:
                partition.offset = next_offset
        consumer.assign(partitions)
        with self._lock:
            self._assignment = list(partitions)
        logger.info(
            "Kafka ingest assigned %d partition(s): %s",
            len(partitions),
            ", ".join(f"{p.topic}[{p.partition}]@{p.offset}" for p in partitions)
            or "nothing",
        )

    def _consume_until_stopped(self) -> None:
        """Poll until :meth:`stop`, offering every record to the gate in
        order, and refreshing the cached lag on a timer."""
        next_watermark_check = 0.0
        while not self._stop.is_set():
            message = self._consumer.poll(_POLL_TIMEOUT_SECONDS)
            if message is not None:
                error = message.error()
                if error is not None:
                    logger.warning(
                        "Kafka cache-event consumer error on topic %s: %s",
                        self._topic,
                        error,
                    )
                else:
                    self._ingest(message)
                    self._consumer.store_offsets(message=message)
                    # Recorded for every message this source is done with,
                    # including one it could not decode: leaving it
                    # unrecorded would make the next restart replay the
                    # same bad record forever.
                    self._position.record(
                        message.topic(), message.partition(), message.offset()
                    )
            now = time.monotonic()
            if now >= next_watermark_check:
                next_watermark_check = now + _WATERMARK_INTERVAL
                self._refresh_lag()

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

    def _refresh_lag(self) -> None:
        """Recompute the cached lag from the broker's high watermarks.

        Runs on the poll thread, so a query that blocks is a partition
        not being consumed -- watermarks are queried cached-first (what
        the last fetch response carried, free) and only queried live for
        a partition nothing has been fetched from yet, which resolves
        after the first fetch; while it does not, the broker is
        unreachable and there is nothing to consume anyway.

        Lag stays :data:`UNKNOWN_LAG` while any assigned partition's
        current position or high watermark is unknown, so an unreachable
        broker reads as "cannot say", never as "caught up".
        """
        with self._lock:
            assignment = list(self._assignment)
        if not assignment:
            self._set_lag(UNKNOWN_LAG)
            return
        try:
            positions = self._consumer.position(assignment)
        except Exception as e:  # noqa: BLE001 - a lag probe must not crash the loop
            logger.debug("Kafka position query failed: %s", e)
            self._set_lag(UNKNOWN_LAG)
            return
        total = 0
        for partition in positions:
            if partition.offset == self._offset_invalid:
                self._set_lag(UNKNOWN_LAG)
                return
            high = self._high_watermark(partition)
            if high == self._offset_invalid:
                self._set_lag(UNKNOWN_LAG)
                return
            total += max(0, high - partition.offset)
        self._set_lag(total)

    def _high_watermark(self, partition: "TopicPartition") -> int:
        """Return one partition's high watermark, or ``OFFSET_INVALID``.

        Args:
            partition: The partition to measure (its own ``.offset`` is
                ignored; only ``.topic`` / ``.partition`` address the
                query).

        Returns:
            The high watermark, or ``OFFSET_INVALID`` if neither the
            cached value nor a live query produced one.
        """
        for cached in (True, False):
            try:
                _, high = self._consumer.get_watermark_offsets(
                    partition, timeout=_POLL_TIMEOUT_SECONDS, cached=cached
                )
            except Exception as e:  # noqa: BLE001 - see _refresh_lag
                logger.debug("Kafka watermark query failed: %s", e)
                return self._offset_invalid
            if high != self._offset_invalid:
                return high
        return self._offset_invalid

    def _set_lag(self, lag: int) -> None:
        """Publish a freshly computed lag.

        Args:
            lag: The new lag, or :data:`UNKNOWN_LAG`.
        """
        with self._lock:
            self._lag = lag
