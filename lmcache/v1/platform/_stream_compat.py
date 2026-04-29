# SPDX-License-Identifier: Apache-2.0
"""Cross-platform compatibility shim for CUDA stream ``launch_host_func``.

The LMCache multiprocess server only needs one capability from ``cupy``:
the ability to attach a Python host callback to a CUDA stream via
``launch_host_func`` (ultimately ``cudaLaunchHostFunc``).  ``torch``'s
native ``torch.cuda.Stream`` does not expose this API, which is why the
code base historically imported ``cupy``.

On platforms without a CUDA toolchain (e.g. macOS, CPU-only CI), importing
``cupy`` either fails outright or is undesirable.  This module abstracts
the construction of an external stream wrapper behind a factory function
so callers can remain oblivious to the backend:

* When CUDA is available AND ``cupy`` imports successfully, we return a
  real ``cupy.cuda.ExternalStream`` — preserving the existing behavior on
  Linux/CUDA hosts exactly.
* Otherwise we return a ``_MockExternalStream`` that implements the same
  minimal surface (``ptr`` attribute, ``launch_host_func(func, arg)``)
  but executes callbacks asynchronously on a dedicated worker thread.

The factory is intentionally a simple strategy-pattern hook: swapping the
CUDA backend (e.g. to a ``ctypes`` binding of ``cudaLaunchHostFunc``) in
the future requires only editing this file.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Protocol
import queue
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class ExternalStreamLike(Protocol):
    """Structural type describing the subset of the cupy stream API we use."""

    ptr: int

    def launch_host_func(self, func: Callable[[Any], None], arg: Any) -> None: ...


class _MockExternalStream:
    """CPU/non-CUDA fallback emulating cupy ``ExternalStream`` semantics.

    Only instantiated on platforms where ``cupy`` is unavailable (macOS,
    CPU-only CI).  Each instance owns a dedicated worker thread that
    consumes an FIFO queue, so host callbacks enqueued via
    ``launch_host_func`` are executed **asynchronously and in order** —
    matching the observable behavior of a CUDA stream without any real
    GPU work to order against.

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
            target=_MockExternalStream._run,
            args=(self._queue, _MockExternalStream._SHUTDOWN),
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
            _MockExternalStream._finalize,
            self._queue,
            self._worker,
            _MockExternalStream._SHUTDOWN,
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
            _MockExternalStream._invoke(func, arg)

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
        _MockExternalStream._finalize(
            self._queue, self._worker, _MockExternalStream._SHUTDOWN
        )


def _try_import_cupy() -> Any | None:
    """Import ``cupy`` lazily; return ``None`` if unavailable."""
    try:
        # Third Party
        import cupy  # type: ignore[import-not-found]

        return cupy
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("cupy not available, using mock stream: %s", exc)
        return None


def make_external_stream(
    torch_stream: torch.cuda.Stream, device_index: int
) -> ExternalStreamLike:
    """Build an external stream wrapper around a ``torch.cuda.Stream``.

    Returns a real ``cupy.cuda.ExternalStream`` when cupy + CUDA are
    available, otherwise a synchronous ``_MockExternalStream``.
    """
    # ``cuda_stream`` only exists on a real CUDA-backed ``torch.cuda.Stream``.
    # On CPU-only hosts the attribute may be missing or raise when accessed,
    # so guard it and fall back to ``0`` — the mock stream treats that as
    # "no usable handle" and synthesizes a fake non-zero id.
    try:
        raw_ptr = int(torch_stream.cuda_stream) if torch_stream is not None else 0
    except Exception:  # pragma: no cover - platform dependent
        raw_ptr = 0

    if torch.cuda.is_available():
        cupy = _try_import_cupy()
        if cupy is not None:
            return cupy.cuda.ExternalStream(raw_ptr, device_index)
    return _MockExternalStream(raw_ptr)
