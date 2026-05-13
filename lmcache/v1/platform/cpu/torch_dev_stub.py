# SPDX-License-Identifier: Apache-2.0
"""CPU-only duck-typed replacement for ``torch.cuda``.

The multiprocess connector code paths use ``lmcache.torch_dev`` as a
single entry point that mirrors a small subset of the ``torch.cuda``
surface, e.g.

.. code-block:: python

    with torch_dev.stream(torch_dev.current_stream()):
        event = torch_dev.Event(interprocess=True)
        event.record()

On hosts without an accelerator (CPU-only CI, macOS) ``torch.cuda``
is technically importable but its calls fail at runtime.  Rather than
sprinkling availability guards across every call site, we expose this
module-level singleton ``cpu_torch_dev`` whose attributes implement
the exact same shape with no-op semantics.  ``lmcache.__init__`` wires
it up as the active ``torch_dev`` whenever no real accelerator is
detected.

Surface coverage
----------------
The following ``torch.cuda`` members are duck-typed here because the
LMCache code base reaches for them somewhere:

* ``Event``                -- ``Event(interprocess=...)`` /
  ``record`` / ``wait`` / ``query`` / ``synchronize`` /
  ``ipc_handle`` / ``from_ipc_handle``.
* ``Stream``               -- minimal ``ptr`` / ``cuda_stream`` /
  ``synchronize`` / ``record_event`` / ``wait_event`` /
  ``wait_stream`` so callers never touch ``None``.
* ``stream(...)``          -- context manager activating the stream.
* ``device(...)``          -- context manager activating a device
  (no-op).
* ``current_stream(...)``  -- returns a :class:`_CpuStream` so call
  sites that immediately call ``.synchronize()`` / pass it to
  ``torch_dev.stream(...)`` keep working.
* ``current_device()``     -- always ``0``.
* ``set_device(...)``      -- no-op.
* ``device_count()``       -- always ``0``.
* ``synchronize(...)``     -- no-op.
* ``empty_cache()``        -- no-op.
* ``init()``               -- no-op.
* ``is_available()``       -- always ``False`` so caller code can
  still detect the CPU fallback when it cares.

``cudart()`` is intentionally omitted: it is exclusively used by
``cache_engine.py`` to call ``cudaHostRegister`` and the call site
already guards on ``hasattr(torch_dev, "cudart")`` and raises a
descriptive error explaining the backend does not support pinned
host memory registration -- the right behaviour on CPU-only hosts.
"""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class _NoopDeviceContext:
    """No-op replacement for ``torch.cuda.device`` + ``torch.cuda.stream``."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:  # noqa: ARG002
        return None


class _CpuEvent:
    """``torch.cuda.Event``-shaped no-op event for CPU-only hosts.

    Implements the minimal subset of the Event API LMCache calls
    (``record`` / ``wait`` / ``query`` / ``synchronize`` /
    ``ipc_handle`` / ``from_ipc_handle``).  All methods are no-ops
    because there is no real GPU work to fence on a CPU-only host.

    The ``interprocess`` keyword + ``from_ipc_handle`` classmethod are
    exposed so :func:`lmcache.utils.check_interprocess_event_support`
    accepts the backend.
    """

    def __init__(
        self,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        # All flags are accepted purely for signature compatibility;
        # there is no real GPU work to fence on a CPU-only host.
        self._enable_timing = enable_timing
        self._blocking = blocking
        self._interprocess = interprocess

    def record(self, stream: Any = None) -> None:  # noqa: ARG002
        return None

    def wait(self, stream: Any = None) -> None:  # noqa: ARG002
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def ipc_handle(self) -> bytes:
        # 64-byte zero-filled handle keeps downstream length checks
        # happy without exposing any real shareable state.
        return b"\x00" * 64

    @classmethod
    def from_ipc_handle(cls, device: Any, handle: bytes) -> "_CpuEvent":  # noqa: ARG003
        return cls(interprocess=True)


class _CpuStream:
    """``torch.cuda.Stream``-shaped no-op stream for CPU-only hosts."""

    def __init__(self, device: Optional[Any] = None) -> None:
        self.device = device
        # ``ptr`` mirrors cupy's ``ExternalStream.ptr`` and
        # ``cuda_stream`` mirrors ``torch.cuda.Stream.cuda_stream`` so
        # downstream "extract a raw handle" helpers never crash.
        self.ptr = id(self)
        self.cuda_stream = self.ptr

    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True

    def record_event(self, event: Optional[_CpuEvent] = None) -> _CpuEvent:
        if event is None:
            event = _CpuEvent()
        event.record(self)
        return event

    def wait_event(self, event: Any) -> None:  # noqa: ARG002
        return None

    def wait_stream(self, stream: Any) -> None:  # noqa: ARG002
        return None


class _CpuTorchDev:
    """Module-like object that duck-types the ``torch.cuda`` surface.

    Only the members touched by LMCache are implemented.  The object is
    intentionally *not* a real Python module so that ``hasattr`` checks
    fail fast for unsupported APIs (e.g. ``cudart``) instead of silently
    returning ``None``.
    """

    Event = _CpuEvent
    Stream = _CpuStream

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def init() -> None:
        return None

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device: Any) -> None:  # noqa: ARG004
        return None

    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def synchronize(device: Optional[Any] = None) -> None:  # noqa: ARG004
        return None

    @staticmethod
    def current_stream(device: Optional[Any] = None) -> _CpuStream:  # noqa: ARG004
        # Return a real (no-op) stream so callers that immediately
        # forward it to ``torch_dev.stream(...)`` or call ``.synchronize()``
        # do not need ``None`` checks.
        return _CpuStream(device)

    @staticmethod
    @contextmanager
    def stream(stream: Optional[Any]) -> Iterator[None]:  # noqa: ARG004
        with _NoopDeviceContext():
            yield

    @staticmethod
    @contextmanager
    def device(device: Optional[Any]) -> Iterator[None]:  # noqa: ARG004
        with _NoopDeviceContext():
            yield


# Module-level singleton imported from ``lmcache.__init__``.
cpu_torch_dev = _CpuTorchDev()
