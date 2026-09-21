# SPDX-License-Identifier: Apache-2.0
"""NVIDIA cuFile implementation of the shared GDS contracts."""

# Standard
from typing import TYPE_CHECKING, Any
import ctypes
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._file import FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle, Submission

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


class Backend(FileGDSBackend):
    """Own the cufile driver and its registration operations."""

    name = "cufile"

    @classmethod
    def is_default(cls) -> bool:
        return torch.version.cuda is not None and torch.version.hip is None

    def validate_environment(self) -> None:
        if torch.version.cuda is None:
            raise ValueError("cufile requires a CUDA PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> Any:
        self._ensure_driver_open()
        # Third Party
        from cufile.bindings import cuFileHandleRegister

        return cuFileHandleRegister(fd)

    def deregister_handle(self, handle: Any) -> None:
        # Third Party
        from cufile.bindings import cuFileHandleDeregister

        cuFileHandleDeregister(handle)

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Use the raw API to preserve cuFile status codes in raised errors."""
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
        # Third Party
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "cuFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Use cuFile's FIXED_* flags (0x7) for per-submission IO parameters."""
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
        # Third Party
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "cuFileStreamDeregister",
        )

    def _open_driver(self) -> None:
        # Third Party
        from cufile.bindings import cuFileDriverOpen

        cuFileDriverOpen()
        _declare_signatures()

    def _close_driver(self) -> None:
        # Third Party
        from cufile.bindings import cuFileDriverClose

        cuFileDriverClose()


class AsyncHandle(GDSHandle):
    """An owning cufile slab handle with stream-ordered IO."""

    _backend: Backend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
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
