# SPDX-License-Identifier: Apache-2.0
"""CPU-only ``ExternalStreamLike`` fallback.

Only instantiated on hosts where no real accelerator-backed external
stream is available (CPU-only CI, macOS, etc.).  The shape of this
module is identical to other ``<device>/stream.py`` siblings so the
dispatcher in :mod:`lmcache.v1.platform.stream` can treat every backend
uniformly.

Design notes:

* One worker per stream preserves cupy's "callbacks on the same stream
  run serially" guarantee while letting distinct streams make progress
  independently (mirrors the high-priority vs. normal stream split in
  :mod:`gpu_context`).
* Running callbacks off the caller thread avoids deadlocks when the
  caller holds a lock that the callback also needs — a hazard the
  synchronous version would otherwise surface only on CPU-only hosts.
* Workers are daemon threads and the queue is drained either when the
  instance is garbage collected or at interpreter shutdown via a
  :func:`weakref.finalize` hook, so pending callbacks are not lost.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable
import queue
import threading
import weakref

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class MockExternalStream:
    """Pure-Python emulation of cupy's ``ExternalStream`` surface.

    Exposes the minimal API required by LMCache's multiprocess server
    (``ptr`` attribute + ``launch_host_func(func, arg)``) and executes
    callbacks asynchronously on a dedicated worker thread so host-side
    FIFO ordering is preserved.
    """

    __slots__ = ("ptr", "_queue", "_worker", "_closed", "_finalizer", "__weakref__")

    # Sentinel signalling the worker thread to exit.
    _SHUTDOWN = object()

    def __init__(self, stream_ptr: int) -> None:
        # Preserve the caller-provided handle so that downstream C++ code
        # (e.g. ``cudaLaunchHostFunc`` via ``record_event_on_stream``) sees
        # a valid CUDA stream pointer when one is available.  Only when
        # the caller has no meaningful pointer (stream_ptr == 0, typical
        # on CPU-only hosts) do we fall back to ``id(self)`` as a
        # guaranteed-unique non-zero fake handle, since some call sites
        # treat ``ptr == 0`` as "default stream".
        self.ptr = int(stream_ptr) if stream_ptr else id(self)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        # Pass queue/sentinel as arguments so the worker's target does
        # not close over ``self``; otherwise the running thread would
        # keep the instance alive and defeat the ``weakref.finalize``
        # based cleanup below.
        self._worker = threading.Thread(
            target=MockExternalStream._run,
            args=(self._queue, MockExternalStream._SHUTDOWN),
            name="mock-stream-cb-%x" % id(self),
            daemon=True,
        )
        self._worker.start()
        # Use ``weakref.finalize`` instead of ``atexit.register`` so the
        # shutdown hook does not keep a strong reference to ``self`` —
        # otherwise every stream created during the process lifetime
        # (notably in unit tests) would leak its worker thread.  The
        # finalizer runs either when the instance is garbage-collected or
        # at interpreter shutdown, whichever comes first.
        self._finalizer = weakref.finalize(
            self,
            MockExternalStream._finalize,
            self._queue,
            self._worker,
            MockExternalStream._SHUTDOWN,
        )

    def launch_host_func(self, func: Callable[[Any], None], arg: Any) -> None:
        if self._closed:
            # Post-shutdown: drain any still-pending work on the worker
            # thread first so the FIFO "in order" contract is preserved,
            # then execute the new callback synchronously on the caller
            # thread (rather than silently dropping it during teardown).
            if self._worker.is_alive():
                self._worker.join(timeout=1.0)
            self._invoke(func, arg)
            return
        self._queue.put((func, arg))

    @staticmethod
    def _run(q: "queue.Queue[Any]", shutdown_sentinel: Any) -> None:
        while True:
            item = q.get()
            if item is shutdown_sentinel:
                return
            func, arg = item
            MockExternalStream._invoke(func, arg)

    @staticmethod
    def _invoke(func: Callable[[Any], None], arg: Any) -> None:
        try:
            func(arg)
        except Exception:
            logger.exception("MockExternalStream host callback raised an exception")

    @staticmethod
    def _finalize(
        q: "queue.Queue[Any]",
        worker: threading.Thread,
        shutdown_sentinel: Any,
    ) -> None:
        """Module-level finalizer bound only to worker state, not self."""
        q.put(shutdown_sentinel)
        # Best-effort drain; daemon thread won't block process exit.
        worker.join(timeout=1.0)

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Detach the finalizer and run the cleanup eagerly so explicit
        # shutdown is deterministic (important for tests).
        self._finalizer.detach()
        MockExternalStream._finalize(
            self._queue, self._worker, MockExternalStream._SHUTDOWN
        )
