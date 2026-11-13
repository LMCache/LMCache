# SPDX-License-Identifier: Apache-2.0
"""Lifetime management for CPU read reservations shared by worker copies."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading
import time
import uuid

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.recurrent_lease import RecurrentReadLease
from lmcache.v1.transport.recurrent_history import HistoryResponse


@dataclass
class HistoryReservation:
    """Keep the source prefix, read lease, and completed result together."""

    key: IPCCacheServerKey
    reads: RecurrentReadLease
    engine_groups: list[int]
    expires: float = field(default_factory=lambda: time.monotonic() + 120)
    outcome: bool | None = None

    def result(self, token: str) -> HistoryResponse:
        """Return the stable completion result and release fenced reads."""
        if self.outcome is None:
            self.outcome = self.reads.release_completed()
        return HistoryResponse(
            lease=token,
            positions=list(self.reads.keys),
            groups=self.engine_groups,
            settled=self.outcome is not None,
            successful=self.outcome is True,
        )


class RecurrentReservations:
    """Serialize claims, cancellation, and retirement of history reads."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entries: dict[str, HistoryReservation] = {}
        self._generation = uuid.uuid4().hex

    def add(self, reservation: HistoryReservation) -> str:
        """Register a read reservation and return its unguessable token."""
        with self.lock:
            token = f"{self._generation}-{uuid.uuid4().hex}"
            self.entries[token] = reservation
            return token

    def query(self, token: str, cancel: bool = False) -> HistoryResponse:
        """Get completion status, optionally canceling unstarted workers."""
        with self.lock:
            entry = self.entries.get(token)
            if entry is None:
                # This generation only retires plans after every copy is fenced.
                return HistoryResponse(
                    lease=token, settled=token.startswith(self._generation + "-")
                )
            if cancel:
                entry.reads.progress.cancel()
            return entry.result(token)

    def reap(self, now: float) -> None:
        """Cancel expired plans and retire only plans whose copies are fenced."""
        with self.lock:
            for token, entry in list(self.entries.items()):
                if now >= entry.expires and self.query(token, cancel=True).settled:
                    del self.entries[token]

    def acknowledge(self, token: str) -> None:
        """Forget a settled reservation after the scheduler releases its blocks."""
        with self.lock:
            if self.query(token).settled:
                self.entries.pop(token, None)
