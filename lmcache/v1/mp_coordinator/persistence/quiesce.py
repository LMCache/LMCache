# SPDX-License-Identifier: Apache-2.0
"""Holding ingest still long enough to read state consistently.

Durable state is spread across several consumers of the cache-event
stream, each with its own lock, and one batch is applied by more than one
of them. Reading them one after another races the stream: a batch applied
between two reads lands in one and not the other, so the captured state
is a moment that never existed. Quiescing closes the window -- new
batches park, the one in flight finishes, and only then does the capture
read anything.

The cost lands on the ingest path, so a quiesce must cover the capture
and nothing more: reads of in-memory state, never I/O.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import threading

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

DEFAULT_QUIESCE_TIMEOUT = 5.0
"""Seconds to wait for in-flight batches before giving up on a capture."""


class QuiesceLock:
    """Mutual exclusion between applying batches and capturing state.

    Two roles share one condition variable: the ingest path holds
    :meth:`applying` around each batch, a capture holds :meth:`quiesced`.
    Requesting it parks arriving batches and waits for the one running,
    which is what lets a capture read several components as one.

    Take it *outside* any lock the ingest path also takes, and never from
    inside :meth:`applying` -- either way a capture ends up waiting on a
    batch that is waiting on the capture.

    One capturer at a time. Two overlapping captures each clear the
    request on the way out, so the first to finish would let ingest
    resume while the second was still reading. Nothing enforces that
    today because nothing captures concurrently; revisit when something
    does.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._quiesce_requested = False
        self._in_flight = 0

    @contextmanager
    def applying(self) -> Iterator[None]:
        """Hold while applying one batch; parks while a quiesce is pending.

        Blocks before the batch rather than during it, so a quiesce never
        interrupts work half-done.

        Yields:
            Once the batch may be applied.
        """
        with self._condition:
            while self._quiesce_requested:
                self._condition.wait()
            self._in_flight += 1
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                if self._in_flight == 0:
                    # The capture is waiting for exactly this.
                    self._condition.notify_all()

    @contextmanager
    def quiesced(self, timeout: float = DEFAULT_QUIESCE_TIMEOUT) -> Iterator[None]:
        """Hold while capturing; no batch is applied for the duration.

        Args:
            timeout: Seconds to wait for in-flight batches. Exceeding it
                abandons the quiesce rather than stalling ingest further.

        Yields:
            Once no batch is in flight and none can start.

        Raises:
            LMCacheTimeoutError: If a batch is still running after
                ``timeout``. Nothing is held and ingest is unaffected.
        """
        with self._condition:
            self._quiesce_requested = True
            if not self._condition.wait_for(lambda: not self._in_flight, timeout):
                self._release_locked()
                raise LMCacheTimeoutError(
                    f"quiesce timed out after {timeout}s with "
                    f"{self._in_flight} batch(es) in flight"
                )
        try:
            yield
        finally:
            with self._condition:
                self._release_locked()

    # -- Internals ------------------------------------------------------------

    def _release_locked(self) -> None:
        """Let parked consumers run; call holding the condition."""
        self._quiesce_requested = False
        self._condition.notify_all()
