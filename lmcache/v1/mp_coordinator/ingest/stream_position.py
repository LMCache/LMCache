# SPDX-License-Identifier: Apache-2.0
"""Where in the Kafka stream a saved checkpoint reaches.

Kafka's own consumer-group committed offset and the coordinator's own
checkpoint are two independent things, saved on two independent timers
(offsets commit roughly every few seconds; a checkpoint every
``--checkpoint-interval``, default 60s, or only at a clean shutdown). On
an ungraceful crash between two checkpoint saves, the committed offset
can already be ahead of what the last saved checkpoint reflects -- and a
consumer that resumes from that committed offset would never re-read the
records in between: the broker considers them delivered, and the
checkpoint that was actually restored does not.

:class:`StreamPosition` closes that gap by checkpointing the read
position itself, in the same artifact as the state it describes, so a
restart replays from what the coordinator's own restored state proves it
has seen -- never from a second opinion that can race ahead of it.
"""

# Standard
from collections.abc import Mapping
from typing import cast
import threading

# First Party
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType


class StreamPosition:
    """The last offset applied per ``(topic, partition)``.

    Recorded only after the gate has admitted the record it names, so a
    captured position can only ever lag the state it rides beside in the
    checkpoint -- never lead it: a checkpoint taken in between replays a
    few already-applied records on the next restart, which the gate's
    sequence dedup absorbs; one taken the other way around would lose
    them for good.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offsets: dict[tuple[str, int], int] = {}

    def record(self, topic: str, partition: int, offset: int) -> None:
        """Record the offset of a record the gate has just admitted.

        Args:
            topic: The record's topic.
            partition: The record's partition.
            offset: The record's offset.
        """
        with self._lock:
            self._offsets[(topic, partition)] = offset

    def next_offset(self, topic: str, partition: int) -> int | None:
        """Return where to resume ``(topic, partition)``.

        Args:
            topic: The partition's topic.
            partition: The partition.

        Returns:
            One past the last recorded offset, or ``None`` if nothing was
            ever recorded for it -- the caller then leaves positioning to
            the consumer's own default (the group's committed offset, or
            ``auto.offset.reset`` if the group has none either).
        """
        with self._lock:
            offset = self._offsets.get((topic, partition))
        return None if offset is None else offset + 1

    # -- DurableComponent --------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this component's section in the checkpoint."""
        return "kafka_stream_position"

    @property
    def persistence_type(self) -> PersistenceType:
        """Rides in the same checkpoint as the state it positions."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return the recorded offsets.

        Returns:
            ``{"offsets": [[topic, partition, offset], ...]}``.
        """
        with self._lock:
            return {
                "offsets": [
                    [topic, partition, offset]
                    for (topic, partition), offset in self._offsets.items()
                ]
            }

    def restore(self, state: Mapping[str, object]) -> None:
        """Replace the recorded offsets with a captured one.

        Args:
            state: A :meth:`capture` value.
        """
        entries = cast("list[list[object]]", state["offsets"])
        with self._lock:
            self._offsets = {
                (cast(str, topic), cast(int, partition)): cast(int, offset)
                for topic, partition, offset in entries
            }
