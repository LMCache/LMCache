# SPDX-License-Identifier: Apache-2.0
"""Versioned, read-only CPU residency observations.

The backend calls this journal while holding its mutation lock. Consumers poll
the public backend API; there are no user callbacks or network operations in a
put/evict critical section. Losing the bounded log requires a new snapshot.
"""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Hashable
from uuid import uuid4


class ResidencySnapshotRequired(RuntimeError):
    """The cursor belongs to another epoch or is outside the retained log."""


@dataclass(frozen=True)
class ResidencyEvent:
    """One committed mutation; sequence also orders each object's versions."""

    seq: int
    key: Hashable
    readable: bool
    size_bytes: int


@dataclass(frozen=True)
class ResidencySnapshot:
    """An immutable readable set and its exact replay cut in one boot epoch."""

    source_epoch: str
    cut_seq: int
    entries: tuple[ResidencyEvent, ...]
    protocol_version: int = 1


@dataclass(frozen=True)
class ResidencyEvents:
    """A contiguous replay page, including an idle source's current cursor."""

    source_epoch: str
    cut_seq: int
    events: tuple[ResidencyEvent, ...]
    protocol_version: int = 1


class ResidencyJournal:
    """Maintain residency metadata with bounded replay, under an owner's lock.

    Args:
        max_events: Maximum number of retained mutations (positive).

    Raises:
        ValueError: If the replay capacity is not positive.

    This class is not independently synchronized. All methods must use the
    same backend mutation lock. Keys must be immutable native cache keys.
    """

    def __init__(self, max_events: int = 16384) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.source_epoch = uuid4().hex
        self.sequence = 0
        self._events: deque[ResidencyEvent] = deque(maxlen=max_events)
        self._readable: dict[Hashable, ResidencyEvent] = {}

    def commit(self, key: Hashable, readable: bool, size_bytes: int = 0) -> None:
        """Record a committed put/delete; duplicate/no-op mutations are ignored.

        Args:
            key: Native object key.
            readable: Whether this backend now owns a readable object.
            size_bytes: Readable object's byte size; ignored for deletion.
        """
        if readable == (key in self._readable):
            return
        self.sequence += 1
        event = ResidencyEvent(
            self.sequence, key, readable, size_bytes if readable else 0
        )
        if readable:
            self._readable[key] = event
        else:
            del self._readable[key]
        self._events.append(event)

    def snapshot(self) -> ResidencySnapshot:
        """Return immutable entries and the cursor captured under the owner lock."""
        return ResidencySnapshot(
            self.source_epoch, self.sequence, tuple(self._readable.values())
        )

    def events(self, source_epoch: str, after_seq: int) -> ResidencyEvents:
        """Read mutations after a cursor without pinning or touching cached KV.

        Args:
            source_epoch: Epoch returned by snapshot().
            after_seq: Last applied sequence, including zero for an empty source.

        Returns:
            A contiguous, immutable replay page and current source cursor.

        Raises:
            ResidencySnapshotRequired: For another boot, an expired cursor, or
                a cursor ahead of the source.
        """
        oldest = self._events[0].seq if self._events else self.sequence + 1
        if (
            source_epoch != self.source_epoch
            or not oldest - 1 <= after_seq <= self.sequence
        ):
            raise ResidencySnapshotRequired("CPU residency snapshot required")
        return ResidencyEvents(
            self.source_epoch,
            self.sequence,
            tuple(event for event in self._events if event.seq > after_seq),
        )
