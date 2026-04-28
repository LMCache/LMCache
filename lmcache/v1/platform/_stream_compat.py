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
import atexit
import queue
import threading

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
    * Workers are daemon threads and the queue is drained at interpreter
      shutdown via :mod:`atexit` so pending callbacks are not lost.
    """

    __slots__ = ("ptr", "_queue", "_worker", "_closed")

    # Sentinel signalling the worker thread to exit.
    _SHUTDOWN = object()

    def __init__(self, stream_ptr: int) -> None:
        # Use ``id(self)`` as a unique non-zero fake handle so downstream
        # code that treats ``ptr == 0`` as the default stream does not
        # misclassify us.  ``stream_ptr`` is intentionally ignored because
        # a torch CPU stream handle carries no meaningful value here.
        del stream_ptr
        self.ptr = id(self)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="mock-stream-cb-%x" % self.ptr,
            daemon=True,
        )
        self._worker.start()
        atexit.register(self._shutdown)

    def launch_host_func(self, func: Callable[[Any], None], arg: Any) -> None:
        if self._closed:
            # Post-shutdown: fall back to synchronous execution so the
            # callback is not silently dropped during teardown.
            self._invoke(func, arg)
            return
        self._queue.put((func, arg))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SHUTDOWN:
                return
            func, arg = item
            self._invoke(func, arg)

    @staticmethod
    def _invoke(func: Callable[[Any], None], arg: Any) -> None:
        try:
            func(arg)
        except Exception:
            logger.exception("MockExternalStream host callback raised an exception")

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._SHUTDOWN)
        # Best-effort drain; daemon thread won't block process exit.
        self._worker.join(timeout=1.0)


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
    if torch.cuda.is_available():
        cupy = _try_import_cupy()
        if cupy is not None:
            return cupy.cuda.ExternalStream(torch_stream.cuda_stream, device_index)
    return _MockExternalStream(int(torch_stream.cuda_stream))
