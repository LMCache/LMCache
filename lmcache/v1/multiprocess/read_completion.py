# SPDX-License-Identifier: Apache-2.0
"""Retain active read handles until a device event proves their stream completed."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)
from lmcache.v1.platform import stream as platform_stream

logger = init_logger(__name__)
ActiveReadCompletion = list[tuple[int, list[int]]]


@dataclass
class _PendingRead:
    completion: ActiveReadCompletion
    device: object
    stream: object
    event: platform_stream.CompletionEvent | None = None


class ActiveReadCompletionTracker:
    """Release exact acquisitions only after their final stream use completes.

    Args:
        on_complete: Idempotent storage-manager release accepting groups of
            manager IDs and active read handles.

    Polling uses platform completion events, so an early or lost native host
    callback cannot release an allocation. Call close after producers stop,
    before closing storage. There is no timeout-based release.
    """

    def __init__(self, on_complete: Callable[[ActiveReadCompletion], None]) -> None:
        self._on_complete = on_complete
        self._pending: dict[int, _PendingRead] = {}
        self._ids = count()
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._thread: PeriodicThread | None = None

    def defer(
        self, completion: ActiveReadCompletion, device: object, stream: object
    ) -> None:
        """Take ownership after the last source-buffer use was enqueued.

        Args:
            completion: Manager/handle groups acquired for the transfer.
            device: Device owning the transfer stream.
            stream: Stream on which the last source-buffer use was submitted.

        Recording failures take a synchronous drain path. If draining also
        fails, handles remain tracked for recovery and the error is propagated.

        Raises:
            RuntimeError: Both recording an event and draining the stream failed.
        """
        pending = _PendingRead(completion, device, stream)
        # Register before recording so even a failing device API cannot orphan
        # the handles. Polling and shutdown serialize with this handoff.
        with self._poll_lock:
            with self._lock:
                identity = next(self._ids)
                self._pending[identity] = pending
            try:
                pending.event = platform_stream.record_completion_event(device, stream)
            except RuntimeError:
                logger.exception("Read completion event failed; draining stream")
                platform_stream.synchronize_stream(device, stream)
                self._on_complete(completion)
                with self._lock:
                    del self._pending[identity]
        with self._lock:
            if self._thread is None:
                self._thread = create_periodic_thread(
                    name=f"active-read-completions-{id(self):x}",
                    interval=0.005,
                    execute_fn=self._execute,
                    level=ThreadLevel.HIGH,
                )
                self._thread.start()

    def poll(self) -> int:
        """Release proven completions, returning the number of batches released.

        Device-query and release errors retain the current batch for retry and
        propagate to the caller. Other batches can be checked on the next poll.
        """
        released = 0
        with self._poll_lock:
            with self._lock:
                snapshot = list(self._pending.items())
            for identity, pending in snapshot:
                if pending.event is None:
                    pending.event = platform_stream.record_completion_event(
                        pending.device, pending.stream
                    )
                if pending.event.is_complete():
                    self._on_complete(pending.completion)
                    with self._lock:
                        del self._pending[identity]
                    released += 1
        return released

    def pending_count(self) -> int:
        """Return the number of batches still retaining source allocations."""
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        """Drain queued streams and release their holds before storage shutdown.

        Producers must have stopped. A device-drain failure propagates and
        leaves the affected holds retained, preventing unsafe L1 destruction.
        """
        if self._thread is not None:
            self._thread.stop()
            PeriodicThreadRegistry.get_instance().unregister(self._thread.name)
            self._thread = None
        with self._poll_lock:
            with self._lock:
                snapshot = list(self._pending.items())
            for identity, pending in snapshot:
                platform_stream.synchronize_stream(pending.device, pending.stream)
                self._on_complete(pending.completion)
                with self._lock:
                    del self._pending[identity]

    def _execute(self) -> ThreadRunSummary:
        released = self.poll()
        return ThreadRunSummary(
            success=True, message=f"completed read batches={released}"
        )
