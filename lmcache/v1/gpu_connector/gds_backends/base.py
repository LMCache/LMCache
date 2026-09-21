# SPDX-License-Identifier: Apache-2.0
"""Shared contracts for stream-ordered GPU storage IO.

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

    Native async APIs receive pointers to these fields, not copies. The caller
    must retain this object and leave size/offset fields unchanged until the
    issuing stream completes. GDSContext manages this lifetime for its IO.
    ``bytes_done`` is valid only after completion; a negative value reports a
    deferred IO error rather than an error from the initial submission call.
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
    """Storage operations and registrations owned by one GDS context.

    A backend module exports a concrete ``Backend`` subclass whose ``name``
    matches the module's configuration name. Construction, default selection,
    and environment validation must not open a native driver. Load optional
    dependencies and acquire driver ownership on the first native operation.
    Platform requirements belong to implementations; this base imposes none.

    Each context exclusively owns its backend instance. Callers must stop new
    operations and complete outstanding DMA before deregistering buffers and
    streams, closing slab handles, and finally calling ``close_driver``.
    Separate instances may share a process-wide native driver: releasing one
    instance must not invalidate another instance's resources. SharedDriver
    provides this accounting for the built-in implementations.

    Implementations propagate native initialization/registration errors.
    Cleanup methods require completed IO; they do not synchronize GPU streams
    on the caller's behalf. Method contracts below are shared by all backends;
    subclass documentation only needs to describe backend-specific behavior.
    """

    name: ClassVar[str]

    @classmethod
    def is_default(cls) -> bool:
        """Return whether this implementation is the default for this environment.

        The factory checks candidates in module-name order, without constructing
        them, and stops at the first match. The default False means explicit
        selection is required. This check must not load a native storage driver.
        """
        return False

    def validate_environment(self) -> None:
        """Raise ValueError if the runtime cannot use this implementation.

        Called before slab setup for explicit and automatic selection. The
        default accepts every environment; subclasses own any required checks.
        """
        return None

    @abstractmethod
    def open_slab(self, location: str, size: int, direct_io: bool) -> "GDSHandle":
        """Prepare ``size`` bytes at ``location`` and return an owning GDSHandle.

        The implementation interprets the location (for example, a directory
        or raw device) and the direct-IO preference. It must validate capacity
        and prepare/register the backing storage before returning. On failure,
        close any descriptor or handle created here; the context then releases
        this backend's driver ownership with ``close_driver``.
        """

    @abstractmethod
    def open_handle(self, fd: int, path: str) -> "GDSHandle":
        """Take ownership of an open descriptor and register it for slab IO.

        ``path`` identifies the backing storage for logging. Ownership transfers
        at the call: close fd if registration fails, otherwise return a handle
        that deregisters and closes it. Unlike ``register_handle``, callers
        must not close fd themselves after this call.
        """

    @abstractmethod
    def register_handle(self, fd: int) -> Any:
        """Register fd and return its backend-specific native handle.

        Acquire driver ownership on first use. This low-level method leaves fd
        ownership with the caller, including on failure; ``open_handle`` wraps
        it when descriptor ownership should transfer to a GDSHandle.
        """

    @abstractmethod
    def deregister_handle(self, handle: Any) -> None:
        """Release a handle returned by register_handle, without closing its fd.

        All IO using the registration must already be complete. GDSHandle.close
        calls this before closing its descriptor, even if deregistration raises.
        """

    @abstractmethod
    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register the contiguous GPU allocation described by ``buf`` for DMA.

        The region is buf.data_ptr() through numel() * element_size() bytes.
        Acquire driver ownership on first use. Callers retain the tensor while
        registered and until all DMA completes; implementations own any device,
        alignment, and maximum-region-size validation.
        """

    @abstractmethod
    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Release a previously registered buffer region after DMA completes.

        Pass the same base pointer used for registration. This unregisters the
        region, but does not free the caller's tensor or close the driver.
        """

    @abstractmethod
    def register_stream(self, raw_stream: int) -> None:
        """Prepare the integer native stream handle for stream-ordered IO.

        Acquire driver ownership on first use. The caller owns the GPU stream;
        implementations may use a native registration or a no-op shim.
        """

    @abstractmethod
    def deregister_stream(self, raw_stream: int) -> None:
        """Release registration for raw_stream after all its IO completes.

        This does not destroy the caller's GPU stream or close the driver.
        """

    @abstractmethod
    def close_driver(self) -> None:
        """Release this instance's driver ownership after its resources are closed.

        Repeated calls and calls before first use are no-ops; they must not load
        a library. Only the last owner may close a shared native driver. Native
        cleanup errors propagate, but another backend's ownership is unaffected.
        """


class GDSHandle(ABC):
    """Own an open slab descriptor and its backend registration.

    ``backend`` owns the registration identified by the opaque native ``handle``;
    this object owns ``fd`` and retains the backend until the handle is closed.
    ``path`` identifies the backing storage for logging. Closing a handle does
    not close its backend's driver; the context releases that after all handles,
    buffer registrations, and stream registrations have been cleaned up.

    Both async methods use the same byte-based arguments: ``buf_base`` is the
    registered GPU base address, ``buf_offset`` is relative to that base,
    ``file_offset`` is relative to the slab, and ``size`` is the transfer length.
    ``raw_stream`` is the integer native GPU stream handle. The caller must keep
    the buffer, stream, handle, and returned Submission alive until completion.
    DMA is ordered after prior work on that stream and before work enqueued
    after it, including GPU operations that produce or consume the buffer data.
    Submission failures raise immediately; deferred DMA errors are reported by
    Submission.bytes_done after the issuing stream completes.
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
        """Enqueue a slab-to-GPU read using the common IO arguments above.

        Returns the Submission holding native argument storage and the eventual
        result. Returning from this method does not imply the DMA has finished.
        """

    @abstractmethod
    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a GPU-to-slab write using the common IO arguments above.

        Returns the Submission holding native argument storage and the eventual
        result. Read bytes_done only after the issuing stream completes.
        """

    def close(self) -> None:
        """Deregister and close fd once, after the caller completes all its IO.

        Descriptor cleanup still runs if deregistration raises. The handle then
        reports fd == -1 and subsequent closes are no-ops. Does not synchronize
        streams or release the backend's driver ownership.
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
