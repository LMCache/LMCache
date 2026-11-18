# SPDX-License-Identifier: Apache-2.0
"""XPU device-event IPC backend.

The ``lmcache_driven`` MP handle path orders cross-process KV transfers
with device events. ``torch.xpu.Event`` is expected to grow the same
interprocess surface ``torch.cuda.Event`` already has, at
:class:`~lmcache.v1.platform.base.event_ipc.EventIPCBackend`

Event pool
----------
On Level Zero, closing an imported IPC event frees its counter allocation
but leaves it in the driver's residency list, so a later submission touches
freed memory (a use-after-free in NEO's ``Event::destroy``; seen with NEO
26.27, not with 26.47). With :data:`ENV_XPU_EVENT_POOL` set to ``1`` (off by
default) this backend never closes an imported event:

* Exporter: :meth:`XpuEventIPCBackend.create_event` hands out events from a
  per-device pool. An event the caller has dropped returns to the pool once
  its last record has completed, and is handed out again only after it has
  been free for :data:`ENV_XPU_EVENT_POOL_REUSE_DELAY_S` seconds, oldest
  first; otherwise a new event is created. The delay gives the importer time
  to check the previous transfer: a check after the re-record would see the
  next transfer and report the earlier one late, never early.
* Importer: a pooled event's handle is imported once per process and device
  and kept for the life of the process. Each import returns a view that
  remembers once it has seen the event complete.

Pooled handles carry a random per-process token, so a restarted exporter
that reuses a dead one's PID cannot match its cached imports. Handles of
events the pool did not create are exported and imported as before. The
pool is meant for the ``lmcache_driven`` transfer path, where both the
LMCache server and vLLM export events, so enable it in both processes.
Once enabled, the process's other users of this backend (such as the qstore
plugin module) pool their events too.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from typing import Any
import os
import threading
import time
import weakref

# Third Party
import torch

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    EventIPCBackend,
    _accepts_interprocess,
)

logger = init_logger(__name__)

#: Set to ``1`` to pool exported events on the ``lmcache_driven`` path;
#: off by default.
ENV_XPU_EVENT_POOL = "LMCACHE_XPU_EVENT_POOL"

#: Seconds a pooled event stays free before it is handed out again.
ENV_XPU_EVENT_POOL_REUSE_DELAY_S = "LMCACHE_XPU_EVENT_POOL_REUSE_DELAY_S"

_DEFAULT_REUSE_DELAY_S = 1.0

#: Prefix of a pooled event's handle, followed by the exporter's token.
_POOLED_HANDLE_MAGIC = b"LMCXEP1\0"
_TOKEN_LEN = 16

#: Instance/class methods ``DefaultEventIPCBackend`` calls on an event.
_REQUIRED_TORCH_XPU_EVENT_SYMBOLS = (
    "from_ipc_handle",
    "ipc_handle",
    "record",
    "wait",
    "query",
    "synchronize",
)

_TORCH_XPU_EVENT_REQUIREMENT = "torch.xpu.Event does not support interprocess handles."


def check_torch_xpu_event_support() -> bool:
    """Return whether ``torch.xpu.Event`` exposes the event IPC API.

    Covers everything ``DefaultEventIPCBackend.check_event_support``
    checks, reusing its
    :func:`~lmcache.v1.platform.base.event_ipc._accepts_interprocess`.
    ``interprocess`` must appear on the *Python* ``torch.xpu.Event``
    class: ``torch/xpu/streams.py`` defines its own ``__new__``, which
    shadows the C++ binding's signature.

    Returns:
        ``True`` when the event class accepts ``interprocess=True`` and
        provides every method the default backend calls.
    """
    xpu_module = getattr(torch, "xpu", None)
    event_cls = getattr(xpu_module, "Event", None)
    return all(
        callable(getattr(event_cls, symbol, None))
        for symbol in _REQUIRED_TORCH_XPU_EVENT_SYMBOLS
    ) and _accepts_interprocess(event_cls)


def _pool_enabled_from_env() -> bool:
    """Return whether :data:`ENV_XPU_EVENT_POOL` turns the pool on."""
    value = os.environ.get(ENV_XPU_EVENT_POOL, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _reuse_delay_from_env() -> float:
    """Return :data:`ENV_XPU_EVENT_POOL_REUSE_DELAY_S`, or the default."""
    raw = os.environ.get(ENV_XPU_EVENT_POOL_REUSE_DELAY_S)
    if raw is None:
        return _DEFAULT_REUSE_DELAY_S
    try:
        delay = float(raw)
    except ValueError:
        delay = -1.0
    if not delay >= 0.0:
        logger.warning(
            "Ignoring %s=%r: expected a non-negative number of seconds; using %s.",
            ENV_XPU_EVENT_POOL_REUSE_DELAY_S,
            raw,
            _DEFAULT_REUSE_DELAY_S,
        )
        return _DEFAULT_REUSE_DELAY_S
    return delay


def _device_key(device: object) -> str:
    """Return the pool and import-cache key of ``device``."""
    if isinstance(device, torch.device):
        return f"{device.type}:{device.index}"
    return str(device)


class _PooledEvent:
    """A pool event lent to a caller; it returns to the pool when dropped.

    Attribute access falls through to the torch event, so direct calls such
    as ``event.wait()`` keep working.
    """

    __slots__ = ("event", "_pool_key", "_released")

    def __init__(self, event: Any, pool_key: str, released: deque) -> None:
        self.event = event
        self._pool_key = pool_key
        self._released = released

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "event"), name)

    def __del__(self) -> None:
        # Only a thread-safe append: this can run inside a locked section
        # of the backend when garbage collection triggers there.
        try:
            self._released.append((self._pool_key, self.event))
        except AttributeError:
            pass


class _ImportedEvent:
    """One import of a cached pooled event.

    Attributes:
        event: The cached torch event, shared by every import of the handle.
        done: Set once this import has seen the event complete; later
            queries, waits and synchronizations return at once.
        waited: Set once a stream has been made to wait on this import.
    """

    __slots__ = ("event", "done", "waited", "__weakref__")

    def __init__(self, event: Any) -> None:
        self.event = event
        self.done = False
        self.waited = False

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "event"), name)


def _raw(event: object) -> object:
    """Return the torch event behind a pooled or imported view."""
    if isinstance(event, (_PooledEvent, _ImportedEvent)):
        return event.event
    return event


class XpuEventIPCBackend(EventIPCBackend):
    """Event IPC backend backed by torch XPU interprocess events.

    Pools exported events and caches imported ones; see the module
    docstring.
    """

    device_type: str = "xpu"

    def __init__(
        self,
        event_module: Any | None = None,
        *,
        pool_enabled: bool | None = None,
        reuse_delay_s: float | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        """Create an XPU event IPC backend.

        Args:
            event_module: Optional module exposing the ``Event`` class.
                Defaults to ``torch.xpu``; injectable for tests.
            pool_enabled: Whether to pool created events. Defaults to
                :data:`ENV_XPU_EVENT_POOL`.
            reuse_delay_s: Seconds a pooled event stays free before reuse.
                Defaults to :data:`ENV_XPU_EVENT_POOL_REUSE_DELAY_S`.
            clock: Monotonic clock in seconds; injectable for tests.
        """
        self._event_module = event_module
        self._default_backend_cache: DefaultEventIPCBackend | None = None
        self._pool_enabled = (
            _pool_enabled_from_env() if pool_enabled is None else pool_enabled
        )
        self._reuse_delay_s = (
            _reuse_delay_from_env() if reuse_delay_s is None else reuse_delay_s
        )
        self._clock = clock
        self._token = os.urandom(_TOKEN_LEN)
        self._lock = threading.Lock()
        # Appended by _PooledEvent.__del__ without the lock.
        self._released: deque[tuple[str, Any]] = deque()
        self._draining: list[tuple[str, Any]] = []
        self._free: dict[str, deque[tuple[float, Any]]] = {}
        self._pool_sizes: dict[str, int] = {}
        self._imported: dict[tuple[bytes, str], Any] = {}
        self._last_import: dict[tuple[bytes, str], weakref.ref] = {}
        self.early_reuse_count = 0
        """Imports of a pooled handle whose previous import was still alive
        and had neither seen completion nor been waited on. Each one may
        have reported its transfer late."""

    def _module(self) -> Any:
        """Return the torch XPU module, resolving it lazily."""
        if self._event_module is None:
            xpu_module = getattr(torch, "xpu", None)
            if xpu_module is None:
                raise RuntimeError("XPU event IPC requires torch.xpu")
            self._event_module = xpu_module
        return self._event_module

    def _default_backend(self) -> DefaultEventIPCBackend:
        """Return the default event adapter bound to ``torch.xpu``.

        Returns:
            A CUDA-style event adapter over the torch XPU event API.

        Raises:
            RuntimeError: If ``torch.xpu`` cannot be resolved.
        """
        backend = self._default_backend_cache
        if backend is not None:
            return backend

        backend = DefaultEventIPCBackend(
            event_module=self._module(),
            device_type=self.device_type,
        )
        self._default_backend_cache = backend
        return backend

    def check_event_support(self, device: object) -> None:
        """Fail closed unless torch XPU event IPC is available.

        Args:
            device: Target XPU device (unused; part of the protocol
                signature).

        Raises:
            RuntimeError: If the torch XPU event IPC API is unavailable.
        """
        if not check_torch_xpu_event_support():
            raise RuntimeError(
                "Device backend 'xpu' does not support interprocess "
                f"events: {_TORCH_XPU_EVENT_REQUIREMENT}"
            )

    def _take_free_event(self, pool_key: str) -> Any | None:
        """Return a reusable event of ``pool_key``'s pool, if any.

        Moves dropped events whose last record has completed to the free
        lists first. The caller must hold ``self._lock``.
        """
        backend = self._default_backend()
        released = self._released
        while released:
            self._draining.append(released.popleft())
        now = self._clock()
        still_running = []
        for key, event in self._draining:
            if backend.query_event(event):
                self._free.setdefault(key, deque()).append((now, event))
            else:
                still_running.append((key, event))
        self._draining = still_running

        free = self._free.get(pool_key)
        if free and now - free[0][0] >= self._reuse_delay_s:
            return free.popleft()[1]
        return None

    def create_event(self, device: object) -> object:
        """Return a torch XPU interprocess event, from the pool if enabled."""
        backend = self._default_backend()
        if not self._pool_enabled:
            return backend.create_event(device)

        pool_key = _device_key(device)
        with self._lock:
            event = self._take_free_event(pool_key)
            if event is None:
                event = backend.create_event(device)
                size = self._pool_sizes.get(pool_key, 0) + 1
                self._pool_sizes[pool_key] = size
                if size >= 64 and size & (size - 1) == 0:
                    logger.info(
                        "XPU IPC event pool for %s grew to %d events.",
                        pool_key,
                        size,
                    )
        return _PooledEvent(event, pool_key, self._released)

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize a torch XPU event for another process.

        Note:
            The returned bytes stay valid only while the exporting
            process keeps ``event`` alive. Pooled events live as long as
            the backend.
        """
        backend = self._default_backend()
        if isinstance(event, _PooledEvent):
            handle = bytes(backend.export_event(event.event, device))
            return _POOLED_HANDLE_MAGIC + self._token + handle
        return backend.export_event(_raw(event), device)

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a torch XPU event handle on ``device``.

        A pooled event's handle is imported once and cached; the event is
        never closed. Other handles are imported on every call.
        """
        backend = self._default_backend()
        handle = bytes(handle)
        if not handle.startswith(_POOLED_HANDLE_MAGIC):
            return backend.import_event(handle, device)

        raw_handle = handle[len(_POOLED_HANDLE_MAGIC) + _TOKEN_LEN :]
        key = (handle, _device_key(device))
        with self._lock:
            event = self._imported.get(key)
            if event is None:
                event = backend.import_event(raw_handle, device)
                self._imported[key] = event
            else:
                previous_ref = self._last_import.get(key)
                previous = previous_ref() if previous_ref is not None else None
                if previous is not None and not previous.done and not previous.waited:
                    self.early_reuse_count += 1
                    logger.debug(
                        "XPU IPC event reused before its previous import "
                        "saw it complete (%d so far); consider raising %s.",
                        self.early_reuse_count,
                        ENV_XPU_EVENT_POOL_REUSE_DELAY_S,
                    )
            view = _ImportedEvent(event)
            self._last_import[key] = weakref.ref(view)
        return view

    def record_event(self, event: object, stream: object) -> None:
        """Record an event on a torch XPU stream."""
        self._default_backend().record_event(_raw(event), stream)

    def wait_event(self, event: object, stream: object) -> None:
        """Make a torch XPU stream wait for an event."""
        if isinstance(event, _ImportedEvent):
            if event.done:
                return
            event.waited = True
        self._default_backend().wait_event(_raw(event), stream)

    def query_event(self, event: object) -> bool:
        """Return whether a torch XPU event has completed."""
        if isinstance(event, _ImportedEvent):
            if event.done:
                return True
            event.done = self._default_backend().query_event(event.event)
            return event.done
        return self._default_backend().query_event(event)

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until a torch XPU event completes."""
        if isinstance(event, _ImportedEvent):
            if not event.done:
                self._default_backend().synchronize_event(event.event, device)
                event.done = True
            return
        self._default_backend().synchronize_event(_raw(event), device)
