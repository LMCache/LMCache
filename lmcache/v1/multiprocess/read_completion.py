# SPDX-License-Identifier: Apache-2.0
"""Track async device transfers and release untimed L1 active reads on completion."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)
from lmcache.v1.distributed.storage_manager import ActiveReadCompletion
from lmcache.v1.platform.stream import (
    CompletionEvent,
    record_completion_event,
    synchronize_stream,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PendingReadCompletion:
    event: CompletionEvent
    completion: ActiveReadCompletion
    device: object
    stream: object


class ActiveReadCompletionTracker:
    """Release untimed L1 active-read handles once their stream copies finish."""

    def __init__(
        self,
        on_complete: Callable[[ActiveReadCompletion], None],
        poll_interval_seconds: float = 0.005,
    ) -> None:
        self._on_complete = on_complete
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = threading.Lock()
        self._pending: list[_PendingReadCompletion] = []
        self._closed = False
        self._poller: PeriodicThread | None = None

    def defer(
        self,
        completion: ActiveReadCompletion | None,
        device: object,
        stream: object,
    ) -> None:
        """Defer releasing ``completion`` until work on ``stream`` finishes."""
        if completion is None or completion.is_empty:
            return
        if stream is None or getattr(device, "type", None) == "cpu":
            self._on_complete(completion)
            return
        try:
            event = record_completion_event(device, stream)
        except Exception:
            logger.exception(
                "ActiveReadCompletionTracker: failed to record completion event; "
                "synchronizing stream before releasing active reads"
            )
            self._sync_and_release(completion, device, stream)
            return

        try:
            if event.is_complete():
                self._on_complete(completion)
                return
        except Exception:
            logger.exception(
                "ActiveReadCompletionTracker: event query failed; "
                "synchronizing stream before releasing active reads"
            )
            self._sync_and_release(completion, device, stream)
            return

        with self._lock:
            if self._closed:
                closed = True
            else:
                closed = False
                self._pending.append(
                    _PendingReadCompletion(event, completion, device, stream)
                )
                self._ensure_poller_locked()
        if closed:
            self._sync_and_release(completion, device, stream)

    def poll(self) -> int:
        """Release all pending completions whose recorded stream events finished."""
        with self._lock:
            snapshot = list(self._pending)
            self._pending.clear()
        still_pending: list[_PendingReadCompletion] = []
        ready: list[ActiveReadCompletion] = []
        for item in snapshot:
            try:
                done = item.event.is_complete()
            except Exception:
                logger.exception(
                    "ActiveReadCompletionTracker: event query failed during poll; "
                    "keeping active reads pinned"
                )
                still_pending.append(item)
                continue
            if done:
                ready.append(item.completion)
            else:
                still_pending.append(item)
        if still_pending:
            with self._lock:
                self._pending = still_pending + self._pending
        for completion in ready:
            self._on_complete(completion)
        return len(ready)

    def pending_count(self) -> int:
        """Return the number of pending async read completions."""
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        """Stop polling and synchronously drain all pending stream completions."""
        with self._lock:
            self._closed = True
            poller = self._poller
            self._poller = None
        if poller is not None:
            PeriodicThreadRegistry.get_instance().unregister(poller.name)
            poller.stop()

        with self._lock:
            remaining = list(self._pending)
            self._pending.clear()
        failed: list[_PendingReadCompletion] = []
        first_error: Exception | None = None
        for item in remaining:
            try:
                if not item.event.is_complete():
                    synchronize_stream(item.device, item.stream)
                self._on_complete(item.completion)
            except Exception as exc:
                logger.exception(
                    "ActiveReadCompletionTracker: failed to drain stream on close; "
                    "keeping active reads pinned to prevent use-after-free"
                )
                failed.append(item)
                if first_error is None:
                    first_error = exc
        if failed:
            with self._lock:
                self._pending = failed + self._pending
        if first_error is not None:
            raise first_error

    def _sync_and_release(
        self, completion: ActiveReadCompletion, device: object, stream: object
    ) -> None:
        try:
            synchronize_stream(device, stream)
        except Exception:
            logger.exception(
                "ActiveReadCompletionTracker: stream synchronize failed; "
                "keeping active reads pinned to prevent use-after-free"
            )
            return
        self._on_complete(completion)

    def _ensure_poller_locked(self) -> None:
        if self._poller is not None:
            return

        def _poll_once() -> ThreadRunSummary:
            released = self.poll()
            return ThreadRunSummary(
                success=True,
                message=f"Released {released} active read completion(s)",
            )

        self._poller = create_periodic_thread(
            name=f"ActiveReadCompletionTracker-{id(self):x}",
            interval=self._poll_interval_seconds,
            execute_fn=_poll_once,
            level=ThreadLevel.HIGH,
            init_wait=self._poll_interval_seconds,
        )
        self._poller.start()
