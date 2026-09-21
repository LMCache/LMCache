# SPDX-License-Identifier: Apache-2.0
"""cufile implementation of the object-based async GDS interface.

Native libraries are loaded lazily. The backend owns driver state; its handles
keep it alive, and the context retains submissions until their DMA completes.
"""

# Standard
from typing import TYPE_CHECKING, Any
import ctypes
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector._gds_async import GDSHandle, Submission
from lmcache.v1.gpu_connector._gds_file import FileGDSBackend

if TYPE_CHECKING:
    # Third Party
    from cufile.bindings import CUfileError


def _declare_signatures() -> None:
    """Set argtypes/restype on libcufile symbols. Idempotent."""
    # Third Party
    from cufile.bindings import CUfileError, libcufile

    if getattr(libcufile.cuFileReadAsync, "argtypes", None):
        return
    libcufile.cuFileReadAsync.argtypes = [
        ctypes.c_void_p,  # CUfileHandle_t fh
        ctypes.c_void_p,  # void *bufPtr_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *bufPtr_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p
        ctypes.c_void_p,  # CUstream stream
    ]
    libcufile.cuFileReadAsync.restype = CUfileError

    libcufile.cuFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    libcufile.cuFileWriteAsync.restype = CUfileError

    libcufile.cuFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libcufile.cuFileStreamRegister.restype = CUfileError

    libcufile.cuFileStreamDeregister.argtypes = [ctypes.c_void_p]
    libcufile.cuFileStreamDeregister.restype = CUfileError


def _check(err: "CUfileError", op: str) -> None:
    """Convert a non-zero ``CUfileError_t`` into a Python exception."""
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: cuFileError(err={err.err}, cu_err={err.cu_err})"
        )


_STREAM_REGISTER_FLAGS = 0x7


class CuFileBackend(FileGDSBackend):
    """Own the cufile driver and its registration operations."""

    name = "cufile"

    def __init__(self) -> None:
        self._driver_opened = False

    @classmethod
    def is_default(cls) -> bool:
        """Preserve the existing default choice for this PyTorch build."""
        return torch.version.cuda is not None and torch.version.hip is None

    def validate_environment(self) -> None:
        """Preserve this implementation's existing PyTorch-build requirement."""
        if torch.version.cuda is None:
            raise ValueError("cufile requires a CUDA PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        """Take ownership of fd and register it; close fd on registration failure."""
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def close_driver(self) -> None:
        """Close the cuFile driver. Optional — useful in tests."""
        if not self._driver_opened:
            return
        # Third Party
        from cufile.bindings import cuFileDriverClose

        try:
            cuFileDriverClose()
        finally:
            self._driver_opened = False

    def register_handle(self, fd: int) -> Any:
        """Register an open fd with cuFile and return the ``CUfileHandle_t``.

        Opens the cuFile driver on first use. The returned handle is accepted
        directly as the first argument of ``cuFileReadAsync`` / ``cuFileWriteAsync``.
        """
        self._ensure_driver_open()
        # Third Party
        from cufile.bindings import cuFileHandleRegister

        return cuFileHandleRegister(fd)

    def deregister_handle(self, handle: Any) -> None:
        """Reverse of :meth:`register_handle` (``cuFileHandleDeregister``)."""
        # Third Party
        from cufile.bindings import cuFileHandleDeregister

        cuFileHandleDeregister(handle)

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a device tensor with cuFile for GDS DMA.

        Must be called before any ``read_async`` / ``write_async`` whose
        ``buf_base`` falls inside this tensor's allocation. Implicitly
        opens the cuFile driver on first use.

        Uses ``libcufile.cuFileBufRegister`` directly (not the
        ``cufile.bindings`` wrapper) because the wrapper hides the error
        code by raising internally — we want the raw status so callers
        see ``cuFileError(err=…, cu_err=…)`` instead of a Python re-raise.
        """
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on CUDA")
        # Third Party
        from cufile.bindings import libcufile

        nbytes = buf.numel() * buf.element_size()
        _check(
            libcufile.cuFileBufRegister(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),
            ),
            "cuFileBufRegister",
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse of :meth:`register_buffer`."""
        # Third Party
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "cuFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Register a CUDA stream with cuFile.

        ``raw_stream`` is the integer ``CUstream`` handle — get it via
        ``torch_dev.current_stream().cuda_stream``.

        Optional for correctness (``read_async`` / ``write_async`` also take the
        stream per call). We register with the FIXED_* flags (0x7): cuFile still
        reads the size/offset pointers at stream-execution time -- so their storage
        must stay alive and unchanged until completion (see ``Submission``) -- but
        promising the values are fixed at submission lets cuFile skip per-op setup,
        worth ~12% higher read throughput in our benchmark.
        """
        # Third Party
        from cufile.bindings import libcufile

        self._ensure_driver_open()
        _check(
            libcufile.cuFileStreamRegister(
                ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
            ),
            "cuFileStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        """Reverse of :meth:`register_stream`."""
        # Third Party
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "cuFileStreamDeregister",
        )

    def _ensure_driver_open(self) -> None:
        """Idempotently open the cuFile driver and declare async signatures."""
        if self._driver_opened:
            return
        # Third Party
        from cufile.bindings import cuFileDriverOpen

        cuFileDriverOpen()
        _declare_signatures()
        self._driver_opened = True


class AsyncHandle(GDSHandle):
    """An owning cufile slab handle with stream-ordered IO."""

    _backend: CuFileBackend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``cuFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g.
        ``buf.data_ptr()``). ``buf_offset`` is the byte offset within
        that registration that the data should land at.
        """
        # Third Party
        from cufile.bindings import libcufile

        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            libcufile.cuFileReadAsync(
                self._handle,
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "cuFileReadAsync",
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
        """Enqueue a ``cuFileWriteAsync`` on the stream."""
        # Third Party
        from cufile.bindings import libcufile

        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            libcufile.cuFileWriteAsync(
                self._handle,
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "cuFileWriteAsync",
        )
        return sub
