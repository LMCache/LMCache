# SPDX-License-Identifier: Apache-2.0
"""Interfaces for stream-ordered GPU storage IO."""

# Standard
from abc import ABC, abstractmethod
from types import TracebackType
from typing import ClassVar
import ctypes

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver


class Submission:
    """Native IO arguments retained until the issuing stream completes.

    Keep the pointer-backed fields unchanged while IO is pending. After completion,
    bytes_done holds the transferred byte count or a negative IO error.
    """

    # Keep per-IO objects compact while allowing weakref lifetime checks.
    __slots__ = ("size", "file_offset", "buf_offset", "result", "__weakref__")

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

    The context opens a slab and manages GPU buffer/stream lifetimes. How the
    backend opens or registers its storage is internal to the implementation.
    Construction and selection must not load native drivers. Platform requirements
    belong to each implementation. Backends with process-wide native state define
    a class-level SharedDriver in _driver. Complete DMA and release this context's
    registrations and handles before releasing driver ownership. Only the last
    owner closes the native session; native errors propagate to the caller.
    """

    name: ClassVar[str]
    _driver: ClassVar[SharedDriver | None] = None

    @classmethod
    def is_default(cls) -> bool:
        """Whether auto selection may use this backend, without loading its driver."""
        return False

    def validate_environment(self) -> None:
        """Raise ValueError for an unsupported runtime; the default accepts all."""
        return None

    @abstractmethod
    def open_slab(self, location: str, size: int, direct_io: bool) -> "GDSHandle":
        """Prepare size bytes at location for IO; release resources on failure."""

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
        """Release this owner's session use; unused/repeated releases do nothing."""
        if self._driver is not None:
            self._driver.release(self, self._close_driver)

    def _ensure_driver_open(self) -> None:
        """Acquire session ownership before registration, opening on first use."""
        if self._driver is not None:
            self._driver.acquire(self, self._open_driver)

    def _open_driver(self) -> None:
        """Perform native initialization; the default requires none."""
        return None

    def _close_driver(self) -> None:
        """Perform native cleanup; the default requires none."""
        return None


class GDSHandle(ABC):
    """Own a slab's IO resources, with path identifying its location.

    IO sizes and offsets are in bytes: buf_offset is relative to the registered
    buf_base, and file_offset is relative to the slab. Operations are ordered on
    raw_stream. Keep the buffer, stream, handle, and Submission alive until completion.
    Submission errors raise immediately; deferred errors appear in bytes_done.
    """

    def __init__(self, path: str) -> None:
        self.path = path

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

    @abstractmethod
    def close(self) -> None:
        """Release this slab's resources; repeated calls do nothing.

        The caller must complete IO first. This does not close the backend's driver.
        """

    def __enter__(self) -> "GDSHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
