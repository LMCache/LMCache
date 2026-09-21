# SPDX-License-Identifier: Apache-2.0
"""Interfaces for stream-ordered GPU storage IO."""

# Standard
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, ClassVar
import ctypes
import os

# Third Party
import torch


class Submission:
    """Native IO arguments retained until the issuing stream completes.

    Keep the pointer-backed fields unchanged while IO is pending. After completion,
    bytes_done holds the transferred byte count or a negative IO error.
    """

    def __init__(self, size: int, file_offset: int, buf_offset: int) -> None:
        self.size = ctypes.c_size_t(size)
        self.file_offset = ctypes.c_int64(file_offset)
        self.buf_offset = ctypes.c_int64(buf_offset)
        self.result = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        """Return the completed byte count, or the driver's negative error."""
        return self.result.value


class GDSBackend(ABC):
    """Backend owned by one GDSContext.

    Construction and selection must not load native drivers. Platform requirements
    belong to each implementation. Complete DMA before releasing registrations,
    handles, and driver state; native errors propagate to the caller.
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self._driver_opened = False

    @classmethod
    def is_default(cls) -> bool:
        """Whether auto selection may use this backend, without loading its driver."""
        return False

    def validate_environment(self) -> None:
        """Raise ValueError for an unsupported runtime; the default accepts all."""
        return None

    @abstractmethod
    def open_slab(self, location: str, size: int, direct_io: bool) -> "GDSHandle":
        """Prepare and register size bytes at location; release resources on failure."""

    @abstractmethod
    def open_handle(self, fd: int, path: str) -> "GDSHandle":
        """Take ownership of fd; return a registered handle or close fd on failure."""

    @abstractmethod
    def register_handle(self, fd: int) -> Any:
        """Return a native registration for fd, leaving fd ownership with the caller."""

    @abstractmethod
    def deregister_handle(self, handle: Any) -> None:
        """Release a native registration without closing its descriptor."""

    @abstractmethod
    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a contiguous GPU tensor for DMA.

        Retain the tensor until all IO completes and the region is unregistered.
        """

    @abstractmethod
    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Unregister the same base pointer without freeing the caller's tensor."""

    @abstractmethod
    def register_stream(self, raw_stream: int) -> None:
        """Prepare a native stream handle for IO without taking ownership of it."""

    @abstractmethod
    def deregister_stream(self, raw_stream: int) -> None:
        """Release registration after IO completes, leaving the stream alive."""

    def close_driver(self) -> None:
        """Close an opened driver once, resetting state even if closing fails.

        Implicit-initialization backends may override this. Calls before opening do
        nothing; state is local to this instance.
        """
        if not self._driver_opened:
            return
        try:
            self._close_driver()
        finally:
            self._driver_opened = False

    def _ensure_driver_open(self) -> None:
        """Open once per instance; failed opens remain retryable.

        Subclasses supply _open_driver/_close_driver and any required synchronization.
        Backends without explicit initialization can ignore these helpers.
        """
        if self._driver_opened:
            return
        self._open_driver()
        self._driver_opened = True

    def _open_driver(self) -> None:
        """Perform native initialization; the default requires none."""
        return None

    def _close_driver(self) -> None:
        """Perform native cleanup; the default requires none."""
        return None


class GDSHandle(ABC):
    """Own a slab descriptor and its backend registration.

    IO sizes and offsets are in bytes: buf_offset is relative to the registered
    buf_base, and file_offset is relative to the slab. Operations are ordered on
    raw_stream. Keep the buffer, stream, handle, and Submission alive until completion.
    Submission errors raise immediately; deferred errors appear in bytes_done.
    """

    def __init__(self, backend: GDSBackend, fd: int, handle: Any, path: str) -> None:
        self._backend = backend
        self._fd = fd
        self._handle = handle
        self.path = path

    @property
    def fd(self) -> int:
        """Return the owned descriptor, or -1 after close()."""
        return self._fd

    @abstractmethod
    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a slab-to-GPU read and return its in-flight Submission."""

    @abstractmethod
    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a GPU-to-slab write and return its in-flight Submission."""

    def close(self) -> None:
        """Deregister and close fd once, even if deregistration fails.

        The caller must complete IO first. This does not close the backend's driver.
        """
        if self._fd < 0:
            return
        try:
            self._backend.deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "GDSHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
