# SPDX-License-Identifier: Apache-2.0
"""How far into the Kafka stream a saved checkpoint reaches.

The coordinator's read position is checkpointed state like any other,
captured under the same quiesce as the views it describes (see
``persistence/checkpoint.py``). That is the whole mechanism: a restored
checkpoint carries both a view and the exact offset that produced it, so
resuming there is correct by construction and nothing has to measure how
far behind the restored view is.

Which is why the consumer group does not commit at all (see
``kafka_event_source.py``). A committed offset would be a second cursor
on its own timer -- offsets commit every few seconds, a checkpoint every
``--checkpoint-interval`` -- and after an ungraceful crash it could sit
ahead of the last saved checkpoint. Resuming from it would skip the
records in between for good: the broker considers them delivered, and
the restored state does not contain them. One cursor, moved only by a
checkpoint, has no such gap to fall into.
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
