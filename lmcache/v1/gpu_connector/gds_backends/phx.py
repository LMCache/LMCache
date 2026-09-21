# SPDX-License-Identifier: Apache-2.0
"""Phoenix implementation through the frozen phxFile shim ABI."""

# Standard
from typing import Optional
import ctypes
import ctypes.util
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle, Submission

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
        if torch.version.hip is None and torch.version.cuda is None:
            raise ValueError("phx requires a ROCm or CUDA PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> int:
        """Box fd as a phx handle; phxfs performs IO on ordinary POSIX fds.

        Load the shim here so a missing library fails at slab setup, not DMA.
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
        self.library().phxFileHandleDeregister(ctypes.c_void_p(handle))

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Let the shim probe FULL-mode devices for the BAR covering this buffer.

        The shim owns page alignment, probe rollback, and device bookkeeping.
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
        """The shim recovers length/device from the base and tolerates unknown bases."""
        _check(
            self.library().phxFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "phxFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Call the shim's no-op registration; each phx IO carries its stream."""
        _check(
            self.library().phxFileStreamRegister(ctypes.c_void_p(raw_stream)),
            "phxFileStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        _check(
            self.library().phxFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "phxFileStreamDeregister",
        )

    def close_driver(self) -> None:
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
        # Closing the shim sweeps its registration table and all opened devices.
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
