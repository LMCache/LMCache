# SPDX-License-Identifier: Apache-2.0
"""Object interfaces for stream-ordered GPU storage IO.

A GDSContext owns one backend and its slab handle. Submissions retain native
argument storage until the context observes completion on the issuing stream.
Importing these interfaces does not load a storage driver.
"""

# Standard
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, ClassVar
import ctypes
import os

# Third Party
import torch


class Submission:
    """Native IO arguments and result, kept alive until stream completion.

    The pointer-backed fields must not be changed while IO is in flight.
    ``bytes_done`` is valid only after the issuing stream completes the IO.
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
    """Storage operations owned by the process's GDS context.

    Construction must not load native libraries. Implementations load their
    driver on first use and release it only after all handles and DMA are done.
    The base class imposes no platform restrictions.
    """

    name: ClassVar[str]

    @classmethod
    def is_default(cls) -> bool:
        """Whether auto selection should choose this implementation."""
        return False

    def validate_environment(self) -> None:
        """Raise ValueError for an environment the implementation cannot use."""
        return None

    @abstractmethod
    def open_slab(self, location: str, size: int, direct_io: bool) -> "GDSHandle":
        """Prepare a slab and return its owning handle; clean up on failure."""

    @abstractmethod
    def open_handle(self, fd: int, path: str) -> "GDSHandle":
        """Take ownership of fd, registering it or closing it on failure."""

    @abstractmethod
    def register_handle(self, fd: int) -> Any:
        """Register fd with the driver without taking ownership of fd."""

    @abstractmethod
    def deregister_handle(self, handle: Any) -> None:
        """Release a native registration after its outstanding IO completes."""

    @abstractmethod
    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a contiguous GPU region for DMA."""

    @abstractmethod
    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Release a region after its outstanding DMA completes."""

    @abstractmethod
    def register_stream(self, raw_stream: int) -> None:
        """Register a raw GPU stream for ordered IO."""

    @abstractmethod
    def deregister_stream(self, raw_stream: int) -> None:
        """Release a stream registration after its outstanding IO completes."""

    @abstractmethod
    def close_driver(self) -> None:
        """Release driver state after all registrations and handles are closed."""


class GDSHandle(ABC):
    """Own an open slab descriptor and its backend registration.

    Callers must complete outstanding IO before calling close(). The backend
    stays reachable through the handle for the whole registration lifetime.
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
        """Enqueue a slab read; retain the result until the stream completes."""

    @abstractmethod
    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a slab write; retain the result until the stream completes."""

    def close(self) -> None:
        """Deregister the handle and close fd exactly once, including on error."""
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
