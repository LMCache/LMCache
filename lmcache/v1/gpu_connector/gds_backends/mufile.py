# SPDX-License-Identifier: Apache-2.0
"""Moore Threads SmartIO muFile implementation of the GDS contracts."""

# Standard
from typing import Any, Optional, cast
import ctypes
import os
import threading

# First Party
from lmcache import torch_device_type
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import Submission

_LIBMUFILE_SONAME = "libmufile.so"
_MUFILE_HANDLE_TYPE_OPAQUE_FD = 1
_STREAM_REGISTER_FLAGS = 0x1
_MUFILE_SUCCESS = 0
_MUFILE_ALIGNMENT = 4096

_OP_ERROR_NAMES: dict[int, str] = {
    0: "MU_FILE_SUCCESS",
    1: "MU_FILE_DRIVER_NOT_INITIALIZED",
    2: "MU_FILE_DRIVER_INVALID_PROPS",
    3: "MU_FILE_DRIVER_UNSUPPORTED_LIMIT",
    4: "MU_FILE_DRIVER_VERSION_MISMATCH",
    5: "MU_FILE_DRIVER_VERSION_READ_ERROR",
    6: "MU_FILE_DRIVER_CLOSING",
    7: "MU_FILE_PLATFORM_NOT_SUPPORTED",
    8: "MU_FILE_IO_NOT_SUPPORTED",
    9: "MU_FILE_DEVICE_NOT_SUPPORTED",
    10: "MU_FILE_MTFS_DRIVER_ERROR",
    11: "MU_FILE_MUSA_DRIVER_ERROR",
    12: "MU_FILE_MUSA_POINTER_INVALID",
    13: "MU_FILE_MUSA_MEMORY_TYPE_INVALID",
    14: "MU_FILE_MUSA_POINTER_RANGE_ERROR",
    15: "MU_FILE_MUSA_CONTEXT_MISMATCH",
    16: "MU_FILE_INVALID_MAPPING_SIZE",
    17: "MU_FILE_INVALID_MAPPING_RANGE",
    18: "MU_FILE_INVALID_FILE_TYPE",
    19: "MU_FILE_INVALID_FILE_OPEN_FLAG",
    20: "MU_FILE_DIO_NOT_SET",
    21: "MU_FILE_INVALID_VALUE",
    22: "MU_FILE_MEMORY_ALREADY_REGISTERED",
    23: "MU_FILE_MEMORY_NOT_REGISTERED",
    24: "MU_FILE_PERMISSION_DENIED",
    25: "MU_FILE_DRIVER_ALREADY_OPEN",
    26: "MU_FILE_HANDLE_NOT_REGISTERED",
    27: "MU_FILE_HANDLE_ALREADY_REGISTERED",
    28: "MU_FILE_DEVICE_NOT_FOUND",
    29: "MU_FILE_INTERNAL_ERROR",
}


class _MUfileError(ctypes.Structure):
    """ctypes mirror of ``MUfileError_t``."""

    _fields_ = [("err", ctypes.c_int)]


class _MUFileHandleUnion(ctypes.Union):
    """ctypes mirror of the POSIX/Win32 handle union."""

    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class _MUFileDescr(ctypes.Structure):
    """ctypes mirror of ``MUFileDescr_t`` on LP64."""

    _fields_ = [("type", ctypes.c_int), ("handle", _MUFileHandleUnion)]


def _op_error_name(err_code: int) -> str:
    """Return a stable name for a muFile operation error."""
    return _OP_ERROR_NAMES.get(abs(err_code), f"MU_FILE_ERROR_{abs(err_code)}")


def _check(err: _MUfileError, op: str) -> None:
    """Raise a Python exception for a non-success muFile status."""
    if err.err != _MUFILE_SUCCESS:
        raise RuntimeError(
            f"{op} failed: muFileError(err={err.err} [{_op_error_name(err.err)}])"
        )


def _check_alignment(
    buf_base: int, size: int, file_offset: int, buf_offset: int
) -> None:
    """Validate the page-aligned operands required by muFile async IO."""
    values = (
        ("buf_base", buf_base),
        ("size", size),
        ("file_offset", file_offset),
        ("buf_offset", buf_offset),
    )
    for name, value in values:
        if value % _MUFILE_ALIGNMENT != 0:
            raise ValueError(f"{name} {value} is not {_MUFILE_ALIGNMENT}-byte aligned")


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Declare the subset of the muFile C ABI used by LMCache."""
    if getattr(lib.muFileReadAsync, "argtypes", None):
        return

    lib.muFileDriverOpen.argtypes = []
    lib.muFileDriverOpen.restype = _MUfileError
    lib.muFileDriverClose.argtypes = []
    lib.muFileDriverClose.restype = _MUfileError

    lib.muFileHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_MUFileDescr),
    ]
    lib.muFileHandleRegister.restype = _MUfileError
    lib.muFileHandleDeregister.argtypes = [ctypes.c_void_p]
    lib.muFileHandleDeregister.restype = _MUfileError

    lib.muFileBufRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.muFileBufRegister.restype = _MUfileError
    lib.muFileBufDeregister.argtypes = [ctypes.c_void_p]
    lib.muFileBufDeregister.restype = _MUfileError

    io_args: list[Any] = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.muFileReadAsync.argtypes = io_args
    lib.muFileReadAsync.restype = ctypes.c_ssize_t
    lib.muFileWriteAsync.argtypes = io_args
    lib.muFileWriteAsync.restype = ctypes.c_ssize_t

    lib.muFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.muFileStreamRegister.restype = _MUfileError
    lib.muFileStreamDeregister.argtypes = [ctypes.c_void_p]
    lib.muFileStreamDeregister.restype = _MUfileError


class Backend(FileGDSBackend):
    """Own the process-shared muFile driver and its registrations."""

    name = "mufile"
    _driver = SharedDriver()

    def __init__(self) -> None:
        super().__init__()
        self._init_lock = threading.Lock()
        self._lib_handle: Optional[ctypes.CDLL] = None
        self._handle_descr_registry: dict[int, _MUFileDescr] = {}
        self._handle_registry_lock = threading.Lock()

    @classmethod
    def is_default(cls) -> bool:
        """Select muFile automatically for a MUSA PyTorch runtime."""
        return torch_device_type == "musa"

    def validate_environment(self) -> None:
        """Reject explicit muFile selection on non-MUSA runtimes."""
        if torch_device_type != "musa":
            raise ValueError("mufile requires a MUSA PyTorch build")

    def open_slab(self, location: str, size: int, direct_io: bool) -> "AsyncHandle":
        """Create a muFile slab, requiring the direct-I/O fast-path contract."""
        if not direct_io:
            raise ValueError("mufile requires --gds-l1-use-direct-io")
        return cast("AsyncHandle", super().open_slab(location, size, direct_io=True))

    def get_raw_stream_handle(self, stream: object) -> int:
        """Extract the native MUSA stream pointer for muFile."""
        for attr in ("musa_stream", "ptr"):
            value = getattr(stream, attr, None)
            if value is not None:
                return int(value)
        raise RuntimeError(
            f"stream of type {type(stream).__name__} does not expose a MUSA handle"
        )

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        """Register an owned descriptor and wrap it in an async handle."""
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> int:
        """Register an fd and retain its descriptor until deregistration."""
        self._ensure_driver_open()
        lib = self.library()
        handle = ctypes.c_void_p()
        descr = _MUFileDescr()
        descr.type = _MUFILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = fd
        _check(
            lib.muFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr)),
            "muFileHandleRegister",
        )
        value = handle.value
        if value is None:
            raise RuntimeError("muFileHandleRegister returned a null handle")
        with self._handle_registry_lock:
            self._handle_descr_registry[value] = descr
        return value

    def deregister_handle(self, handle: int) -> None:
        """Deregister a handle while retaining its descriptor through the call."""
        with self._handle_registry_lock:
            _check(
                self.library().muFileHandleDeregister(ctypes.c_void_p(handle)),
                "muFileHandleDeregister",
            )
            self._handle_descr_registry.pop(handle, None)

    def register_buffer(self, buf: Any) -> None:
        """Register a MUSA tensor for direct DMA."""
        if not getattr(buf, "is_musa", False):
            raise ValueError("register_buffer: tensor must be on the MUSA device")
        self._ensure_driver_open()
        nbytes = buf.numel() * buf.element_size()
        _check(
            self.library().muFileBufRegister(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),
            ),
            "muFileBufRegister",
        )

    def deregister_buffer(self, buf: Any) -> None:
        """Deregister a previously registered MUSA tensor."""
        _check(
            self.library().muFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "muFileBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Register a stream with muFile's fixed-and-aligned flag."""
        self._ensure_driver_open()
        _check(
            self.library().muFileStreamRegister(
                ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
            ),
            "muFileStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        """Deregister a stream after its queued IO has completed."""
        _check(
            self.library().muFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "muFileStreamDeregister",
        )

    def library(self) -> ctypes.CDLL:
        """Load and declare ``libmufile.so`` lazily, once per backend."""
        if self._lib_handle is not None:
            return self._lib_handle
        with self._init_lock:
            if self._lib_handle is None:
                lib = ctypes.CDLL(_LIBMUFILE_SONAME)
                _declare_signatures(lib)
                self._lib_handle = lib
        return self._lib_handle

    def _open_driver(self) -> None:
        """Open the muFile driver after the shared owner is acquired."""
        _check(self.library().muFileDriverOpen(), "muFileDriverOpen")

    def _close_driver(self) -> None:
        """Close the muFile driver after the final backend releases it."""
        _check(self.library().muFileDriverClose(), "muFileDriverClose")


class AsyncHandle(FDGDSHandle):
    """An owning muFile slab handle with stream-ordered IO."""

    _backend: Backend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a muFile read and retain its completion storage."""
        return self._submit(
            self._backend.library().muFileReadAsync,
            "muFileReadAsync",
            buf_base,
            size,
            file_offset,
            buf_offset,
            raw_stream,
        )

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a muFile write and retain its completion storage."""
        return self._submit(
            self._backend.library().muFileWriteAsync,
            "muFileWriteAsync",
            buf_base,
            size,
            file_offset,
            buf_offset,
            raw_stream,
        )

    def _submit(
        self,
        operation: Any,
        operation_name: str,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Marshal one native muFile async operation."""
        _check_alignment(buf_base, size, file_offset, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        # muFile's native async ABI uses size_t* for completion, while the
        # shared backend contract exposes a signed result for deferred errors.
        result = ctypes.cast(ctypes.byref(sub.result), ctypes.POINTER(ctypes.c_size_t))
        ret = operation(
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(buf_base),
            ctypes.byref(sub.size),
            ctypes.byref(sub.file_offset),
            ctypes.byref(sub.buf_offset),
            result,
            ctypes.c_void_p(raw_stream),
        )
        if ret < 0:
            raise RuntimeError(
                f"{operation_name} failed: muFileError(err={-int(ret)} "
                f"[{_op_error_name(int(ret))}])"
            )
        return sub
