# SPDX-License-Identifier: Apache-2.0

"""Route ``Server.store`` / ``Server.retrieve`` completion callbacks
through a C++ host callback so the CUDA driver thread never touches
the GIL.

The legacy path used ``stream.launch_host_func(finish_write, keys)``,
which scheduled a Python callback on the CUDA stream. The driver thread
running the callback would block on the GIL while the calling thread
held both the GIL and the CUDA driver lock — the same deadlock that
PR #2952 fixed for ``EventBus.publish_on_stream``. This module extends
that pattern to the two remaining ``launch_host_func`` callsites in
``Server``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Iterable
import pickle
import threading

# First Party
from lmcache.logging import init_logger

try:
    # Third Party
    import torch  # noqa: F401 — must be imported before lmcache.c_ops

    # First Party
    import lmcache.c_ops as _lmc_ops

    _has_native = hasattr(_lmc_ops, "record_completion_on_stream") and hasattr(
        _lmc_ops, "drain_recorded_completions"
    )
except ImportError:
    _has_native = False

logger = init_logger(__name__)

# Handler receives the same Python objects the caller passed to
# ``record_on_stream``. Runs on the dispatcher thread under the GIL.
CompletionHandler = Callable[[Any], None]


class CompletionDispatcher:
    """Drain thread that pulls buffered completions and dispatches each
    to the handler registered for its ``kind``."""

    def __init__(self, drain_interval_seconds: float = 0.005) -> None:
        self._handlers: dict[str, CompletionHandler] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._drain_interval = drain_interval_seconds
        self._dispatched_count = 0
        self._exception_counts: dict[str, int] = {}

    def register(self, kind: str, handler: CompletionHandler) -> None:
        with self._lock:
            self._handlers[kind] = handler

    def start(self) -> None:
        if not _has_native:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="CompletionDispatcher",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        if _has_native:
            self._drain_once()

    def dispatched_count(self) -> int:
        return self._dispatched_count  # single-writer; read is GIL-atomic

    def handler_exception_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._exception_counts)

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            self._wake.wait(timeout=self._drain_interval)
            self._wake.clear()
            self._drain_once()

    def _drain_once(self) -> None:
        # Broad except keeps the drain thread alive across native/handler errors.
        try:
            completions = _lmc_ops.drain_recorded_completions()
        except Exception:
            logger.exception("CompletionDispatcher: drain failed")
            return
        if not completions:
            return
        with self._lock:
            handlers = dict(self._handlers)
        for kind, encoded_items in completions:
            handler = handlers.get(kind)
            if handler is None:
                logger.warning(
                    "CompletionDispatcher: no handler for kind=%r (dropped %d)",
                    kind,
                    len(encoded_items),
                )
                continue
            try:
                payload = [pickle.loads(item) for item in encoded_items]
                handler(payload)
                self._dispatched_count += 1
            except Exception:
                with self._lock:
                    self._exception_counts[kind] = (
                        self._exception_counts.get(kind, 0) + 1
                    )
                logger.exception("CompletionDispatcher: handler for %r raised", kind)


def is_native_available() -> bool:
    return _has_native


def record_on_stream(
    stream: Any,
    kind: str,
    payload: Iterable[Any],
    fallback_handler: CompletionHandler | None = None,
) -> None:
    """Schedule a completion record on *stream* without taking the GIL.

    Falls back to ``stream.launch_host_func(fallback_handler, payload)``
    when the native recorder isn't compiled into ``lmcache.c_ops``.
    """
    payload_list = list(payload)
    if _has_native:
        encoded = [
            pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL) for p in payload_list
        ]
        _lmc_ops.record_completion_on_stream(stream.ptr, kind, encoded)
        return

    if fallback_handler is None:
        raise RuntimeError(
            "record_on_stream: native recorder unavailable and no fallback_handler"
        )
    stream.launch_host_func(fallback_handler, payload_list)
