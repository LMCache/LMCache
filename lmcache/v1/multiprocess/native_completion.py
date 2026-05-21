# SPDX-License-Identifier: Apache-2.0

"""Dispatch device-stream-ordered host callbacks without GIL contention.

Call graph::

    Server.store / Server.retrieve  (Python, holds GIL)
        |
        v
    submit_callback_to_stream(stream, kind, payload)
        |
        v  Python: msgspec.msgpack.encode each payload item
        v  C++:    record_completion_on_stream (gil_scoped_release)
        v  C++:    cudaLaunchHostFunc / hipLaunchHostFunc
        |
        v  -------- on the CUDA/HIP driver thread (no GIL) --------
        v  completion_host_callback: append PendingCompletion to buffer
        |
        v  -------- on DeviceHostFuncDispatcher drain thread (Python) --
        v  drain_recorded_completions  (returns list of (kind, [bytes...]))
        v  msgspec.msgpack.decode each item with the kind's registered type
        v  handler(items)  ← finish_write / finish_read_prefetched

Extends PR #2952's AB-BA fix (GIL x driver lock) to Server.store/retrieve.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Iterable
import threading

# Third Party
import msgspec
import torch  # noqa: F401 — must be imported before lmcache.c_ops

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as _lmc_ops

logger = init_logger(__name__)

# Runs on dispatcher thread with the decoded items from one submission.
DeviceHostFunc = Callable[[list[Any]], None]


class _Registration(msgspec.Struct):
    handler: DeviceHostFunc
    decoder: msgspec.msgpack.Decoder


class DeviceHostFuncDispatcher:
    """Drain buffered C++ completions and dispatch each batch to the handler
    registered for its ``kind``. One instance per process; owned by
    ``MPCacheEngine``."""

    def __init__(self, drain_interval_seconds: float = 0.005) -> None:
        self._registry: dict[str, _Registration] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._drain_interval = drain_interval_seconds
        self._dispatched_count = 0
        self._exception_counts: dict[str, int] = {}

    def register(self, kind: str, handler: DeviceHostFunc, item_type: Any) -> None:
        """Register *handler* for *kind*; ``item_type`` is the per-payload-item
        msgspec decode type (e.g. ``ObjectKey``)."""
        with self._lock:
            self._registry[kind] = _Registration(
                handler=handler,
                decoder=msgspec.msgpack.Decoder(type=item_type),
            )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="DeviceHostFuncDispatcher",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
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
            logger.exception("DeviceHostFuncDispatcher: drain failed")
            return
        if not completions:
            return
        with self._lock:
            registry = dict(self._registry)
        for kind, encoded_items in completions:
            reg = registry.get(kind)
            if reg is None:
                logger.warning(
                    "DeviceHostFuncDispatcher: no handler for kind=%r (dropped %d)",
                    kind,
                    len(encoded_items),
                )
                continue
            try:
                items = [reg.decoder.decode(item) for item in encoded_items]
                reg.handler(items)
                self._dispatched_count += 1
            except Exception:
                with self._lock:
                    self._exception_counts[kind] = (
                        self._exception_counts.get(kind, 0) + 1
                    )
                logger.exception(
                    "DeviceHostFuncDispatcher: handler for %r raised", kind
                )


def submit_callback_to_stream(stream: Any, kind: str, payload: Iterable[Any]) -> None:
    """Schedule a stream-ordered host callback for *kind* with *payload*.
    Runs on the dispatcher worker registered for *kind*; never acquires the
    GIL on the driver thread."""
    encoded = [msgspec.msgpack.encode(p) for p in payload]
    _lmc_ops.record_completion_on_stream(stream.ptr, kind, encoded)
