# SPDX-License-Identifier: Apache-2.0
"""Revisioned missing-range bookkeeping, independent of stored-prefix cursors."""

# Standard
from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class WritebackTicket:
    """Identity and revision of one exact-range external write-back attempt."""

    identity: str
    start: int
    end: int
    revision: int


class RecomputedWriteback:
    """Track dirty token ranges without owning engine allocations or frontiers."""

    def __init__(self) -> None:
        """Initialize bookkeeping without owning transfers or engine allocations."""
        self._dirty: dict[tuple[int, int], int] = {}
        self._inflight: dict[tuple[int, int], WritebackTicket] = {}
        self._revision = 0

    @property
    def has_pending(self) -> bool:
        """Whether any missing range still needs ordinary STORE admission."""
        return bool(self._dirty)

    def observe(self, start: int, end: int) -> None:
        """Record a missing range without accepting any earlier acknowledgment.

        Args:
            start: Inclusive token boundary.
            end: Exclusive token boundary. Callers split coverage into chunks.

        Raises:
            ValueError: The range is empty, negative or overlaps a different range.
        """
        span = (start, end)
        if (
            start < 0
            or end <= start
            or any(
                span != other and start < other[1] and other[0] < end
                for other in self._dirty
            )
        ):
            raise ValueError("Write-back ranges must be positive and non-overlapping")
        self._revision += 1
        self._dirty[span] = self._revision

    def ready(self, committed_tokens: int) -> list[tuple[int, int]]:
        """Return dirty ranges below a committed boundary, excluding in-flight IO.

        Args:
            committed_tokens: Engine-certified completed prefix, never planned work.

        Returns:
            Sorted exact ranges that can be submitted using a valid pinned source.
        """
        return sorted(
            span
            for span in self._dirty
            if span[1] <= committed_tokens and span not in self._inflight
        )

    def begin(self, start: int, end: int) -> WritebackTicket:
        """Reserve a dirty range and snapshot its revision.

        Args:
            start: Inclusive token boundary.
            end: Exclusive token boundary.

        Returns:
            A unique attempt ticket, suitable for asynchronous completion routing.

        Raises:
            ValueError: This range is unknown or already in flight.
        """
        span = (start, end)
        if span not in self._dirty or span in self._inflight:
            raise ValueError("Write-back range is not ready")
        ticket = WritebackTicket(uuid.uuid4().hex, start, end, self._dirty[span])
        self._inflight[span] = ticket
        return ticket

    def complete(self, ticket: WritebackTicket, success: bool) -> bool:
        """Settle an attempt, retaining dirty state on failure or newer evidence.

        Args:
            ticket: Ticket returned by :meth:`begin`.
            success: Whether the ordinary STORE acknowledged successful admission.

        Returns:
            Whether this identity settled an in-flight attempt. Duplicate/stale
            identities are ignored and never clear a newer revision.
        """
        span = (ticket.start, ticket.end)
        if self._inflight.get(span) != ticket:
            return False
        del self._inflight[span]
        if success and self._dirty.get(span) == ticket.revision:
            del self._dirty[span]
        return True
