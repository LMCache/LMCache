# SPDX-License-Identifier: Apache-2.0
"""phx implementation of the object-based async GDS interface.

Native libraries are loaded lazily. The backend owns driver state; its handles
keep it alive, and the context retains submissions until their DMA completes.
"""

# Standard
from typing import Optional
import ctypes
import ctypes.util
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector._gds_async import GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FileGDSBackend

logger = init_logger(__name__)


def _declare_signatures(lib: ctypes.CDLL, path_hint: str) -> None:
    """Set argtypes/restype on the phxFile symbols used by this module."""
    # The frozen ABI always provides the full surface; the check guards
    # against a stale shim predating the async symbols, so it fails fast
    # with a clear message instead of at the first DMA.
    missing = [
        sym
        for sym in ("phxFileReadAsync", "phxFileWriteAsync")
        if not hasattr(lib, sym)
    ]
    if missing:
        raise RuntimeError(
            f"libphxfile at {path_hint} lacks the stream-ordered API "
            f"({', '.join(missing)}); reinstall "
            f"phoenix/adapters/lmcache/phxfile"
        )

    lib.phxFileDriverOpen.argtypes = []
    lib.phxFileDriverOpen.restype = ctypes.c_int

    lib.phxFileDriverClose.argtypes = []
    lib.phxFileDriverClose.restype = ctypes.c_int

    lib.phxFileBufRegister.argtypes = [
        ctypes.c_void_p,  # const void *addr
        ctypes.c_size_t,  # size_t length
    ]
    lib.phxFileBufRegister.restype = ctypes.c_int

    lib.phxFileBufDeregister.argtypes = [
        ctypes.c_void_p,  # const void *addr
    ]
    lib.phxFileBufDeregister.restype = ctypes.c_int

    lib.phxFileHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # void **fh (out)
        ctypes.c_int,  # int fd
    ]
    lib.phxFileHandleRegister.restype = ctypes.c_int

    lib.phxFileHandleDeregister.argtypes = [
        ctypes.c_void_p,  # void *fh
    ]
    lib.phxFileHandleDeregister.restype = ctypes.c_int

    lib.phxFileStreamRegister.argtypes = [
        ctypes.c_void_p,  # void *stream
    ]
    lib.phxFileStreamRegister.restype = ctypes.c_int

    lib.phxFileStreamDeregister.argtypes = [
        ctypes.c_void_p,  # void *stream
    ]
    lib.phxFileStreamDeregister.restype = ctypes.c_int

    # hipFile parameter order: (fh, buf, *nbytes, *file_offset,
    # *buf_offset, *bytes_done, stream). The libphoenix order differs
    # (buf_offset before f_offset); the shim performs the swap.
    io_params: list[type] = [
        ctypes.c_void_p,  # void *fh
        ctypes.c_void_p,  # void *buf_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *nbytes
        ctypes.POINTER(ctypes.c_int64),  # int64_t *file_offset
        ctypes.POINTER(ctypes.c_int64),  # int64_t *buf_offset
        ctypes.POINTER(ctypes.c_int64),  # int64_t *bytes_done
        ctypes.c_void_p,  # void *stream (vendor-opaque)
    ]
    lib.phxFileReadAsync.argtypes = io_params
    lib.phxFileReadAsync.restype = ctypes.c_int
    lib.phxFileWriteAsync.argtypes = io_params
    lib.phxFileWriteAsync.restype = ctypes.c_int


def _check(rc: int, op: str) -> None:
    """Convert a negative phxFile return code into a Python exception."""
    if rc < 0:
        try:
            why = os.strerror(-rc)
        except ValueError:
            why = "unknown error"
        raise RuntimeError(f"{op} failed: phxFileError(rc={rc} [{why}])")


class Backend(FileGDSBackend):
    """Own the phx driver and its registration operations."""

    name = "phx"
    _driver = SharedDriver()

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._driver_opened = False

    def validate_environment(self) -> None:
        """Preserve this implementation's existing PyTorch-build requirement."""
        if torch.version.hip is None and torch.version.cuda is None:
            raise ValueError("phx requires a ROCm or CUDA PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        """Take ownership of fd and register it; close fd on registration failure."""
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> int:
        """Accept an open fd for phx IO and return the "handle".

        Wraps ``phxFileHandleRegister`` — currently an identity boxing (the
        handle IS the fd; phxfs performs IO on plain POSIX fds); :class:
        `AsyncHandle` round-trips it and closes the fd on ``close()``. Loads
        ``libphxfile`` eagerly so a missing library fails at slab setup, not
        at first DMA.

        Args:
            fd: Open slab-file descriptor (as created by
                :meth:`gds_context.GDSContext.initialize`).

        Returns:
            The registered ``phxFileHandle_t`` (equal to ``fd`` today).
        """
        lib = self.library()
        fh = ctypes.c_void_p()
        _check(
            int(lib.phxFileHandleRegister(ctypes.byref(fh), ctypes.c_int(fd))),
            "phxFileHandleRegister",
        )
        if fh.value is None:
            raise RuntimeError("phxFileHandleRegister returned a null handle")
        return fh.value

    def deregister_handle(self, handle: int) -> None:
        """Reverse of :meth:`register_handle` (``phxFileHandleDeregister``)."""
        self.library().phxFileHandleDeregister(ctypes.c_void_p(handle))

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a device tensor for GDS DMA via the frozen shim.

        Wraps ``phxFileBufRegister`` — the frozen, device-free entry point.
        The shim resolves the buffer's device itself (probe-based: it opens
        every FULL-mode phxfs device and registers on the one whose BAR
        covers the buffer; failed probes roll back cleanly inside
        libphoenix). Page-size alignment and the registration bookkeeping
        also live inside the shim.

        Args:
            buf: Contiguous GPU tensor (a <=16 MiB slice of a staging buffer,
                as passed by :meth:`gds_context.GDSContext.register_gpu_buffer`).

        Raises:
            ValueError: If ``buf`` is not a GPU tensor or is empty.
            RuntimeError: If the shim rejects the registration (no phxfs
                device, all-staging, or the last probe error).
        """
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on a CUDA or ROCm GPU")
        nbytes = buf.numel() * buf.element_size()
        if nbytes == 0:
            raise ValueError("register_buffer: tensor is empty")
        _check(
            self.library().phxFileBufRegister(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
            ),
            "phxFileBufRegister",
        )
        logger.debug(
            "phx: registered 0x%x (%d bytes) via libphxfile probe",
            buf.data_ptr(),
            nbytes,
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse of :meth:`register_buffer`.

        Wraps ``phxFileBufDeregister``: only the base address is passed; the
        aligned registration length and phxfs device are played back from the
        shim's bookkeeping. Unregistered buffers are silently tolerated
        inside the shim (matching the tolerance of the cuFile path teardown).

        Args:
            buf: The tensor previously passed to :meth:`register_buffer`.

        Raises:
            RuntimeError: If ``phxFileBufDeregister`` fails.
        """
        _check(
            self.library().phxFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "phxFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Register a stream with the shim (``phxFileStreamRegister``).

        A frozen no-op today: phxfs has no stream registration (every
        submission carries the stream handle, unlike cuFile's optional
        cuFileStreamRegister hint). Kept on the shared backend surface for
        wrapper parity and reserved for future per-stream resource
        pre-claiming.

        Args:
            raw_stream: Raw CUDA/ROCm stream handle.
        """
        _check(
            self.library().phxFileStreamRegister(ctypes.c_void_p(raw_stream)),
            "phxFileStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        """Reverse of :meth:`register_stream` (``phxFileStreamDeregister``)."""
        _check(
            self.library().phxFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "phxFileStreamDeregister",
        )

    def close_driver(self) -> None:
        """Release this backend's ownership; the last owner closes the shim.

        Wraps ``phxFileDriverClose``: the shim sweeps any buffer registration
        still in its table, closes every phxfs device it opened, and resets
        its caches. Individual cleanup failures are reported by the shim on
        stderr and do not raise.
        """
        if not self._driver_opened:
            return
        try:
            self._driver.release(self, self._close_driver)
        finally:
            self._driver_opened = False

    def library(self) -> ctypes.CDLL:
        """Load ``libphxfile.so`` on first use and declare the frozen ABI."""
        if self._lib is None:
            search = ctypes.util.find_library("phxfile")
            path = search or "libphxfile.so"
            lib = ctypes.CDLL(path)
            _declare_signatures(lib, path)
            self._lib = lib
        if not self._driver_opened:
            self._driver.acquire(self, self._open_driver)
            self._driver_opened = True
        return self._lib

    def _open_driver(self) -> None:
        # phx opens devices lazily during registration, not at library load.
        pass

    def _close_driver(self) -> None:
        assert self._lib is not None
        _check(self._lib.phxFileDriverClose(), "phxFileDriverClose")


class AsyncHandle(GDSHandle):
    """An owning phx slab handle with stream-ordered IO."""

    _backend: Backend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Submit a stream-ordered read DMA into a registered GPU buffer.

        Returns immediately; the data is guaranteed to be visible to any
        op the caller enqueues on ``raw_stream`` after this call. The
        transfer outcome lands in ``submission.bytes_done`` once the
        stream is synchronized past this op.

        Args:
            buf_base: Base pointer of a registered GPU buffer region.
            size: Transfer length in bytes.
            file_offset: Slab-file offset to read from.
            buf_offset: Offset within the GPU buffer region.
            raw_stream: Raw CUDA/ROCm stream handle ordering the DMA.

        Returns:
            The in-flight submission (keep alive until the stream sync).

        Raises:
            RuntimeError: On a submission-level failure. Transfer
                failures are NOT raised -- they land in
                ``submission.bytes_done`` after the stream sync.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            self._backend.library().phxFileReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "phxFileReadAsync",
        )
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Submit a stream-ordered write DMA from the registered GPU buffer.

        The DMA is ordered after everything previously enqueued on
        ``raw_stream`` (e.g. the gather producing the data), so the buffer
        contents are stable when the DMA reads them. Returns immediately;
        the outcome lands in ``submission.bytes_done`` after the stream
        sync.

        Args:
            buf_base: Base pointer of a registered GPU buffer region.
            size: Transfer length in bytes.
            file_offset: Slab-file offset to write to.
            buf_offset: Offset within the GPU buffer region.
            raw_stream: Raw CUDA/ROCm stream handle ordering the DMA.

        Returns:
            The in-flight submission (keep alive until the stream sync).

        Raises:
            RuntimeError: On a submission-level failure. Transfer
                failures are NOT raised -- they land in
                ``submission.bytes_done`` after the stream sync.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            self._backend.library().phxFileWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "phxFileWriteAsync",
        )
        return sub
