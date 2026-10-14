# SPDX-License-Identifier: Apache-2.0
"""Kafka pull implementation of the coordinator cache-event source.

Consumes the records a ``KafkaCacheEventSink`` produces -- one
``CacheEventsRequest`` envelope per record, keyed by ``instance_id`` -- and
offers each record's batches to :class:`EventGate` in partition order. Kafka retains
the stream, so this is the coordinator's durable door; the gate's per-emitter
sequence cursors stay separate from Kafka's partition offsets.

Where a restart resumes is
:class:`~.stream_position.StreamPosition`'s answer alone: the consumer
group never commits, so the checkpoint is the only cursor there is, and
it advances only when a checkpoint is written. That is what makes a
restart correct without anything having to compare the two -- the
position was captured under the same quiesce as the state it rides
beside (see ``persistence/checkpoint.py``), so it describes exactly the
view that was restored. Records applied after that capture are simply
read again, and the gate's sequence dedup absorbs them.

A partition :class:`StreamPosition` has never seen -- a first start, or
a partition added since -- has no checkpointed cursor, and falls back to
``auto.offset.reset``.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Standard
from dataclasses import dataclass, field, replace
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
from lmcache.v1.mp_coordinator.ingest.stream_position import StreamPosition
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest

if TYPE_CHECKING:
    # Third Party
    from confluent_kafka import Consumer, Message, TopicPartition

logger = init_logger(__name__)

# Bounds how long stop() waits for the poll loop to notice the stop flag.
_POLL_TIMEOUT_SECONDS = 1.0


@dataclass
class KafkaSourceStats:
    """What the Kafka source has read since this process started.

    Attributes:
        records_applied: Records decoded and handed to the gate.
        records_undecodable: Records skipped because they did not decode.
        records_failed: Records skipped because applying them raised.
        partitions_assigned: Partitions of the topic this coordinator holds.
            Fewer than the topic has means another member of the consumer
            group holds the rest, and its events never reach this one.
        unread_records: Records not yet read, per assigned partition: the
            partition's high watermark minus the next offset to read. A
            partition is absent until both are known. Kafka's own lag tools
            cannot report this, because the group never commits offsets.
    """

    records_applied: int = 0
    records_undecodable: int = 0
    records_failed: int = 0
    partitions_assigned: int = 0
    unread_records: dict[int, int] = field(default_factory=dict)


class KafkaCacheEventSource(CacheEventSource):
    """Durable pull source that consumes cache events from a Kafka topic.

    One thread polls the consumer and offers each record's batches to the
    gate, so a partition's records -- one instance's stream -- arrive in
    order. A record's offset is recorded into ``position`` only after the
    gate has seen it, so delivery is at-least-once; the gate's sequence
    dedup makes redelivery harmless. A record that cannot be decoded, or
    whose ingest raises, is logged and skipped -- and still recorded -- so
    one bad record cannot stall its partition. A consumer that raises is
    handled by the broadcaster and does not reach here.

    ``confluent-kafka`` is the optional ``lmcache[kafka]`` extra and is
    imported only here.

    Args:
        event_gate: Admission authority for consumed event batches.
        config: Validated broker, topic, and consumer-group settings.
        position: The cursor, and the only one: seeked to on assignment,
            recorded into as records are admitted, and persisted by
            whatever checkpoint captures it next.

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
            from confluent_kafka import Consumer, KafkaException
        except ImportError as e:
            raise ImportError(
                "The Kafka cache-event source needs confluent-kafka: "
                "pip install 'lmcache[kafka]'"
            ) from e
        self._event_gate = event_gate
        self._topic = config.topic
        self._position = position
        self._kafka_exception: type[Exception] = KafkaException
        # Counts written by the poll thread; unread_records is computed when
        # stats() is read.
        self._stats = KafkaSourceStats()
        # Replaced whole by each assignment, never changed in place.
        self._assigned: list["TopicPartition"] = []
        self._consumer = Consumer(
            {
                "bootstrap.servers": config.bootstrap_servers,
                "group.id": config.group_id,
                "client.id": "lmcache-coordinator",
                "auto.offset.reset": "earliest",
                # The checkpoint is the cursor. A committed offset would
                # be a second one, advancing on its own timer, and a
                # restart that trusted it would skip whatever the last
                # checkpoint had not captured yet.
                "enable.auto.commit": False,
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
        self._consumer.subscribe([self._topic], on_assign=self._on_assign)
        self._thread.start()

    async def stop(self) -> None:
        """Stop the poll thread, then close the consumer.

        Nothing is committed on the way out: the checkpoint written
        during shutdown is what a restart resumes from.
        """
        self._stop.set()
        if self._thread.is_alive():
            await asyncio.to_thread(self._thread.join)
        self._consumer.close()

    def status(self) -> CacheEventSourceStatus:
        """Return the Kafka source's status.

        Returns:
            Source identity ``kafka`` with replay capability
            ``SEEKABLE``: the topic retains the stream, so it can be
            replayed by rewinding the checkpointed position.
        """
        return CacheEventSourceStatus(
            source_name="kafka",
            replay_capability=EventReplayCapability.SEEKABLE,
        )

    def stats(self) -> KafkaSourceStats:
        """Return a copy of what the source has read; see
        :class:`KafkaSourceStats`.

        ``unread_records`` is computed here, on the caller's thread, from
        each assigned partition's high watermark in the consumer's cache
        (updated by every fetch, no broker round trip). A failure there
        reaches only the caller, never the poll thread doing the ingest.
        """
        return replace(self._stats, unread_records=self._unread_records())

    # -- Poll thread -----------------------------------------------------------

    def _on_assign(
        self, consumer: "Consumer", partitions: list["TopicPartition"]
    ) -> None:
        """Seek each assigned partition to the checkpoint's position.

        Runs on the poll thread, during ``poll()``. A partition
        :attr:`_position` has recorded an offset for is repositioned
        there; one it has never seen keeps whatever confluent-kafka
        assigned by default, which with no committed offset to find is
        ``auto.offset.reset``. Calling ``assign`` here is what makes the
        override take effect -- without it, the default assignment
        stands.

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
        self._assigned = list(partitions)
        self._stats.partitions_assigned = len(partitions)
        logger.info(
            "Kafka ingest assigned %d partition(s): %s",
            len(partitions),
            ", ".join(f"{p.topic}[{p.partition}]@{p.offset}" for p in partitions)
            or "nothing",
        )

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
            # Recorded for every message this source is done with,
            # including one it could not decode: leaving it unrecorded
            # would make the next restart replay the same bad record
            # forever.
            self._position.record(
                message.topic(), message.partition(), message.offset()
            )

    def _ingest(self, message: "Message") -> None:
        """Decode one record and offer its batches to the gate.

        Decode and ingest failures are logged and swallowed; the caller
        records the offset either way so the partition keeps moving.
        """
        position = f"{message.topic()}[{message.partition()}]@{message.offset()}"
        try:
            batches = CacheEventsRequest.model_validate_json(
                message.value() or b""
            ).batches
        except ValidationError as e:
            self._stats.records_undecodable += 1
            logger.warning(
                "Dropping undecodable cache-event record %s: %s", position, e
            )
            return
        try:
            summary = self._event_gate.ingest_batches(batches)
        except Exception:
            self._stats.records_failed += 1
            logger.exception("Cache-event consumers failed on record %s", position)
            return
        self._stats.records_applied += 1
        logger.debug(
            "Kafka cache-event record %s: %d applied, %d duplicate, %d stale",
            position,
            summary.applied,
            summary.duplicates,
            summary.stale,
        )

    def _unread_records(self) -> dict[int, int]:
        """Return the records not yet read, per assigned partition whose
        high watermark and next offset are both known."""
        unread_records: dict[int, int] = {}
        for partition in self._assigned:
            try:
                _, high_watermark = self._consumer.get_watermark_offsets(
                    partition, cached=True
                )
            except self._kafka_exception:
                logger.debug(
                    "No cached high watermark for %s[%d]",
                    partition.topic,
                    partition.partition,
                )
                continue
            next_offset = self._position.next_offset(
                partition.topic, partition.partition
            )
            # A negative watermark means none fetched yet.
            if high_watermark >= 0 and next_offset is not None:
                unread_records[partition.partition] = max(
                    0, high_watermark - next_offset
                )
        return unread_records
