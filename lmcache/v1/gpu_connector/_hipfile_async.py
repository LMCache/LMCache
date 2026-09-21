# SPDX-License-Identifier: Apache-2.0
"""hipfile implementation of the object-based async GDS interface.

Native libraries are loaded lazily. The backend owns driver state; its handles
keep it alive, and the context retains submissions until their DMA completes.
"""

# Standard
from typing import Optional
import ctypes
import os
import threading

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector._gds_async import GDSHandle, Submission
from lmcache.v1.gpu_connector._gds_driver import SharedDriver
from lmcache.v1.gpu_connector._gds_file import FileGDSBackend

_LIBHIPFILE_SONAME = "libhipfile.so"

_HIPFILE_HANDLE_TYPE_OPAQUE_FD = 1

_STREAM_REGISTER_FLAGS = 0x7

_HIPFILE_SUCCESS = 0


class _HipFileError(ctypes.Structure):
    """ctypes mirror of ``hipFileError_t`` (hipfile.h).

    Two ints: ``err`` is the ``hipFileOpError_t`` status (0 == success);
    ``hip_drv_err`` carries the underlying ``hipError_t`` when ``err`` indicates
    a GPU-driver failure.
    """

    _fields_ = [("err", ctypes.c_int), ("hip_drv_err", ctypes.c_int)]


class _HipFileHandleUnion(ctypes.Union):
    """ctypes mirror of the ``hipFileDescr_t.handle`` union (fd or Win32 HANDLE)."""

    _fields_ = [("fd", ctypes.c_int), ("hFile", ctypes.c_void_p)]


class _HipFileDescr(ctypes.Structure):
    """ctypes mirror of ``hipFileDescr_t`` (hipfile.h)."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HipFileHandleUnion),
        ("fs_ops", ctypes.c_void_p),
    ]


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype on the libhipfile symbols. Idempotent."""
    if getattr(lib.hipFileReadAsync, "argtypes", None):
        return

    lib.hipFileDriverOpen.argtypes = []
    lib.hipFileDriverOpen.restype = _HipFileError

    lib.hipFileDriverClose.argtypes = []
    lib.hipFileDriverClose.restype = _HipFileError

    lib.hipFileHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # hipFileHandle_t *fh
        ctypes.POINTER(_HipFileDescr),  # hipFileDescr_t *descr
    ]
    lib.hipFileHandleRegister.restype = _HipFileError

    lib.hipFileHandleDeregister.argtypes = [ctypes.c_void_p]  # hipFileHandle_t fh
    lib.hipFileHandleDeregister.restype = None  # void

    lib.hipFileBufRegister.argtypes = [
        ctypes.c_void_p,  # const void *buffer_base
        ctypes.c_size_t,  # size_t length
        ctypes.c_int,  # int flags
    ]
    lib.hipFileBufRegister.restype = _HipFileError

    lib.hipFileBufDeregister.argtypes = [ctypes.c_void_p]  # const void *buffer_base
    lib.hipFileBufDeregister.restype = _HipFileError

    lib.hipFileReadAsync.argtypes = [
        ctypes.c_void_p,  # hipFileHandle_t fh
        ctypes.c_void_p,  # void *buffer_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # hoff_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # hoff_t *buffer_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p
        ctypes.c_void_p,  # hipStream_t stream
    ]
    lib.hipFileReadAsync.restype = _HipFileError

    lib.hipFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    lib.hipFileWriteAsync.restype = _HipFileError

    lib.hipFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.hipFileStreamRegister.restype = _HipFileError

    lib.hipFileStreamDeregister.argtypes = [ctypes.c_void_p]
    lib.hipFileStreamDeregister.restype = _HipFileError

    lib.hipFileGetOpErrorString.argtypes = [ctypes.c_int]
    lib.hipFileGetOpErrorString.restype = ctypes.c_char_p


class HipFileBackend(FileGDSBackend):
    """Own the hipfile driver and its registration operations."""

    name = "hipfile"
    _driver = SharedDriver()

    def __init__(self) -> None:
        self._driver_opened = False
        self._init_lock = threading.Lock()
        self._lib_handle: Optional[ctypes.CDLL] = None

    @classmethod
    def is_default(cls) -> bool:
        """Preserve the existing default choice for this PyTorch build."""
        return torch.version.hip is not None

    def validate_environment(self) -> None:
        """Preserve this implementation's existing PyTorch-build requirement."""
        if torch.version.hip is None:
            raise ValueError("hipfile requires a ROCm PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        """Take ownership of fd and register it; close fd on registration failure."""
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def close_driver(self) -> None:
        """Release this backend's driver ownership after its IO has completed.

        Raises:
            RuntimeError: If ``hipFileDriverClose`` reports a non-success status.
        """
        if not self._driver_opened:
            return
        try:
            self._driver.release(self, self._close_driver)
        finally:
            self._driver_opened = False

    def register_handle(self, fd: int) -> int:
        """Register an open fd with hipFile and return the ``hipFileHandle_t``.

        Opens the hipFile driver on first use. The returned handle is the raw
        ``void *`` value (as an int), accepted directly as the first argument of
        ``hipFileReadAsync`` / ``hipFileWriteAsync``.

        Args:
            fd: An open file descriptor for the slab (opened with ``O_DIRECT`` for
                the GDS fast path).

        Returns:
            The registered ``hipFileHandle_t`` as an integer.

        Raises:
            RuntimeError: If ``hipFileHandleRegister`` reports a non-success status.
        """
        self._ensure_driver_open()
        lib = self.library()
        handle = ctypes.c_void_p()
        descr = _HipFileDescr()
        descr.type = _HIPFILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = None
        self.check_error(
            lib.hipFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr)),
            "hipFileHandleRegister",
        )
        return handle.value if handle.value is not None else 0

    def deregister_handle(self, handle: int) -> None:
        """Reverse of :meth:`register_handle` (``hipFileHandleDeregister``).

        Args:
            handle: The ``hipFileHandle_t`` (as an int) from :meth:`register_handle`.
        """
        self.library().hipFileHandleDeregister(ctypes.c_void_p(handle))

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a device tensor with hipFile for GPUDirect Storage DMA.

        Must be called before any ``read_async`` / ``write_async`` whose
        ``buf_base`` falls inside this tensor's allocation. Implicitly opens the
        hipFile driver on first use.

        Args:
            buf: A GPU (HIP device) tensor to register for DMA.

        Raises:
            ValueError: If ``buf`` is not on the GPU.
            RuntimeError: If ``hipFileBufRegister`` reports a non-success status.
        """
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on the GPU")
        self._ensure_driver_open()
        lib = self.library()
        nbytes = buf.numel() * buf.element_size()
        self.check_error(
            lib.hipFileBufRegister(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),
            ),
            "hipFileBufRegister",
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse of :meth:`register_buffer`.

        Args:
            buf: A tensor previously passed to :meth:`register_buffer`.

        Raises:
            RuntimeError: If ``hipFileBufDeregister`` reports a non-success status.
        """
        self.check_error(
            self.library().hipFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "hipFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Register a HIP stream with hipFile.

        ``raw_stream`` is the integer ``hipStream_t`` handle — get it via
        ``torch_dev.current_stream().cuda_stream`` (torch reports the HIP stream
        through the same attribute on ROCm).

        Optional for correctness (``read_async`` / ``write_async`` also take the
        stream per call). Registered with the FIXED_* flags (0x7): hipFile still
        reads the size/offset pointers at stream-execution time -- so their storage
        must stay alive and unchanged until completion (see ``Submission``) -- but
        promising the values are fixed at submission lets hipFile skip per-op setup.

        Args:
            raw_stream: The integer ``hipStream_t`` handle to register.

        Raises:
            RuntimeError: If ``hipFileStreamRegister`` reports a non-success status.
        """
        self._ensure_driver_open()
        self.check_error(
            self.library().hipFileStreamRegister(
                ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
            ),
            "hipFileStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        """Reverse of :meth:`register_stream`.

        Args:
            raw_stream: The ``hipStream_t`` handle passed to :meth:`register_stream`.

        Raises:
            RuntimeError: If ``hipFileStreamDeregister`` reports a non-success status.
        """
        self.check_error(
            self.library().hipFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "hipFileStreamDeregister",
        )

    def library(self) -> ctypes.CDLL:
        """dlopen ``libhipfile.so`` once and declare signatures. Idempotent.

        Thread-safe via double-checked locking: the common case (already loaded)
        returns without taking ``self._init_lock``, so the per-DMA path stays lock-free.

        Returns:
            This backend's cached ``libhipfile.so`` handle.
        """
        if self._lib_handle is not None:
            return self._lib_handle
        with self._init_lock:
            if self._lib_handle is None:
                handle = ctypes.CDLL(_LIBHIPFILE_SONAME)
                _declare_signatures(handle)
                self._lib_handle = handle
        return self._lib_handle

    def check_error(self, err: "_HipFileError", op: str) -> None:
        """Convert a non-zero ``hipFileError_t`` into a Python exception."""
        if err.err != _HIPFILE_SUCCESS:
            raise RuntimeError(
                f"{op} failed: hipFileError(err={err.err} "
                f"[{self._op_error_string(err.err)}], "
                f"hip_drv_err={err.hip_drv_err})"
            )

    def _ensure_driver_open(self) -> None:
        """Idempotently open the hipFile driver (thread-safe).

        Raises:
            RuntimeError: If ``hipFileDriverOpen`` reports a non-success status.
        """
        if self._driver_opened:
            return
        self._driver.acquire(self, self._open_driver)
        self._driver_opened = True

    def _open_driver(self) -> None:
        self.check_error(self.library().hipFileDriverOpen(), "hipFileDriverOpen")

    def _close_driver(self) -> None:
        self.check_error(self.library().hipFileDriverClose(), "hipFileDriverClose")

    def _op_error_string(self, err_code: int) -> str:
        """Return the human-readable name for a ``hipFileOpError_t`` value."""
        raw = self.library().hipFileGetOpErrorString(ctypes.c_int(abs(err_code)))
        return raw.decode() if raw is not None else "unknown"


class AsyncHandle(GDSHandle):
    """An owning hipfile slab handle with stream-ordered IO."""

    _backend: HipFileBackend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``hipFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g. ``buf.data_ptr()``).
        ``buf_offset`` is the byte offset within that registration that the data
        should land at.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        self._backend.check_error(
            self._backend.library().hipFileReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "hipFileReadAsync",
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
        """Enqueue a ``hipFileWriteAsync`` on the stream."""
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        self._backend.check_error(
            self._backend.library().hipFileWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "hipFileWriteAsync",
        )
        return sub
