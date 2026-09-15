# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper around the uGDS async C API (libugds.so).

Drop-in replacement for :mod:`_cufile_async` that uses uGDS's user-space
NVMe path instead of cuFile. Its common GDS operations are compatible with
the other backends, while :func:`get_device_capacity` is uGDS-specific because
raw devices cannot rely on file allocation to validate the requested slab.

LMCache expects ``libugds.so`` to match the active NVIDIA CUDA or AMD ROCm
platform.

uGDS operates on raw character devices (/dev/ugds_drv*), not filesystem files.
The fd/handle split is the same as _cufile_async: the caller opens the
device (O_RDWR, no O_DIRECT since uGDS IO bypasses the kernel), passes the
fd to register_handle, and wraps the pair via AsyncHandle.from_fd.
"""

# Standard
from typing import Any, Optional
import ctypes
import ctypes.util
import os

# Third Party
import torch

# --- libugds.so lazy loading -----------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    search = ctypes.util.find_library("ugds")
    path = search or "libugds.so"
    _lib = ctypes.CDLL(path)
    _declare_signatures(_lib)
    return _lib


# --- uGDS C types ----------------------------------------------------


class _uGDSError_t(ctypes.Structure):
    _fields_ = [
        ("err", ctypes.c_int),
        ("cu_err", ctypes.c_int),
    ]


class _uGDSDescr_t(ctypes.Structure):
    class _HandleUnion(ctypes.Union):
        _fields_ = [
            ("fd", ctypes.c_int),
            ("handle", ctypes.c_void_p),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
    ]


_UGDS_HANDLE_TYPE_OPAQUE_FD = 1


def _declare_signatures(lib: ctypes.CDLL) -> None:
    lib.uGDSDriverOpen.argtypes = []
    lib.uGDSDriverOpen.restype = _uGDSError_t

    lib.uGDSDriverClose.argtypes = []
    lib.uGDSDriverClose.restype = _uGDSError_t

    lib.uGDSHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # uGDSHandle_t *fh
        ctypes.POINTER(_uGDSDescr_t),  # uGDSDescr_t *descr
    ]
    lib.uGDSHandleRegister.restype = _uGDSError_t

    lib.uGDSHandleDeregister.argtypes = [ctypes.c_void_p]
    lib.uGDSHandleDeregister.restype = None

    try:
        get_device_capacity_fn = lib.uGDSGetDeviceCapacity
    except AttributeError as exc:
        raise RuntimeError(
            "libugds.so does not provide uGDSGetDeviceCapacity; update uGDS "
            "before enabling the LMCache uGDS backend"
        ) from exc
    get_device_capacity_fn.argtypes = [
        ctypes.c_void_p,  # uGDSHandle_t fh
        ctypes.POINTER(ctypes.c_uint64),  # uint64_t *capacity_bytes
    ]
    get_device_capacity_fn.restype = _uGDSError_t

    lib.uGDSBufRegister.argtypes = [
        ctypes.c_void_p,  # const void *bufPtr_base
        ctypes.c_size_t,  # size_t length
        ctypes.c_int,  # int flags
    ]
    lib.uGDSBufRegister.restype = _uGDSError_t

    lib.uGDSBufDeregister.argtypes = [ctypes.c_void_p]
    lib.uGDSBufDeregister.restype = _uGDSError_t

    lib.uGDSReadAsync.argtypes = [
        ctypes.c_void_p,  # uGDSHandle_t fh
        ctypes.c_void_p,  # void *bufPtr_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *bufPtr_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p
        ctypes.c_void_p,  # CUDA or HIP stream handle
    ]
    lib.uGDSReadAsync.restype = _uGDSError_t

    lib.uGDSWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    lib.uGDSWriteAsync.restype = _uGDSError_t

    lib.uGDSStreamRegister.argtypes = [ctypes.c_void_p]
    lib.uGDSStreamRegister.restype = _uGDSError_t

    lib.uGDSStreamDeregister.argtypes = [ctypes.c_void_p]
    lib.uGDSStreamDeregister.restype = _uGDSError_t


# --- Error checking --------------------------------------------------


def _check(err: _uGDSError_t, op: str) -> None:
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: uGDSError(err={err.err}, cu_err={err.cu_err})"
        )


# --- Driver lifecycle ------------------------------------------------

_driver_opened = False


def _ensure_driver_open() -> None:
    global _driver_opened
    if _driver_opened:
        return
    lib = _get_lib()
    _check(lib.uGDSDriverOpen(), "uGDSDriverOpen")
    _driver_opened = True


def close_driver() -> None:
    global _driver_opened
    if not _driver_opened:
        return
    lib = _get_lib()
    try:
        _check(lib.uGDSDriverClose(), "uGDSDriverClose")
    finally:
        _driver_opened = False


# --- Handle registration --------------------------------------------


def register_handle(fd: int) -> int:
    """Register an open uGDS device fd and return the raw uGDSHandle_t.

    Mirrors _cufile_async.register_handle(fd): the caller owns the fd
    (typically an O_RDWR open of /dev/ugds_drvX) and closes it on
    registration failure; wrap the pair via AsyncHandle.from_fd.
    """
    _ensure_driver_open()
    lib = _get_lib()
    handle = ctypes.c_void_p()
    descr = _uGDSDescr_t()
    descr.type = _UGDS_HANDLE_TYPE_OPAQUE_FD
    descr.handle.fd = fd
    _check(
        lib.uGDSHandleRegister(ctypes.byref(handle), ctypes.byref(descr)),
        "uGDSHandleRegister",
    )
    if handle.value is None:
        raise RuntimeError("uGDSHandleRegister returned a null handle")
    return handle.value


def deregister_handle(handle: int) -> None:
    """Reverse of register_handle (uGDSHandleDeregister)."""
    lib = _get_lib()
    lib.uGDSHandleDeregister(ctypes.c_void_p(handle))


def get_device_capacity(fd: int, handle: int) -> int:
    """Return the NVMe namespace capacity associated with a uGDS handle.

    Args:
        fd: Open uGDS character-device descriptor. It is accepted for API
            consistency with the file-based GDS backends and is not inspected.
        handle: Registered ``uGDSHandle_t`` whose namespace capacity to query.

    Returns:
        Usable namespace capacity in bytes.

    Raises:
        RuntimeError: If uGDS cannot query the device or returns zero capacity.
    """
    del fd
    capacity_bytes = ctypes.c_uint64()
    _check(
        _get_lib().uGDSGetDeviceCapacity(
            ctypes.c_void_p(handle), ctypes.byref(capacity_bytes)
        ),
        "uGDSGetDeviceCapacity",
    )
    if capacity_bytes.value == 0:
        raise RuntimeError("uGDSGetDeviceCapacity returned zero capacity")
    return capacity_bytes.value


# --- Buffer / stream registration -----------------------------------

# UGDS_REGISTER_DMABUF from ugds.h requests the AMD HIP dma-buf path.
_UGDS_REGISTER_DMABUF = 0x1


def _buf_register_flags() -> int:
    """Return the ``uGDSBufRegister`` flags matching the PyTorch platform.

    Returns:
        ``_UGDS_REGISTER_DMABUF`` on ROCm, or ``0`` on CUDA.
    """
    return _UGDS_REGISTER_DMABUF if torch.version.hip is not None else 0


def register_buffer(buf: torch.Tensor) -> None:
    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on a CUDA or ROCm GPU")
    _ensure_driver_open()
    lib = _get_lib()
    nbytes = buf.numel() * buf.element_size()
    _check(
        lib.uGDSBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(_buf_register_flags()),
        ),
        "uGDSBufRegister",
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    lib = _get_lib()
    _check(
        lib.uGDSBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "uGDSBufDeregister",
    )


def register_stream(raw_stream: int) -> None:
    _ensure_driver_open()
    lib = _get_lib()
    _check(
        lib.uGDSStreamRegister(ctypes.c_void_p(raw_stream)),
        "uGDSStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    lib = _get_lib()
    _check(
        lib.uGDSStreamDeregister(ctypes.c_void_p(raw_stream)),
        "uGDSStreamDeregister",
    )


# --- Submission + AsyncHandle ----------------------------------------


class Submission:
    """One in-flight uGDSReadAsync / uGDSWriteAsync.

    Mirrors :class:`_cufile_async.Submission`: holds the ctypes storage for the
    size and offset arguments, which uGDS takes by pointer and dereferences
    later from a stream-ordered host callback.

    The instance must therefore stay reachable until the stream has executed
    the operation. Dropping the last reference earlier lets the garbage
    collector free the ctypes storage, and the callback then reads a dangling
    pointer. That surfaces as silently wrong IO (typically all-zero reads)
    rather than an error, so callers must retain submissions until they have
    synchronized the stream. :class:`~gds_context.GDSContext` does this by
    tracking per-stream in-flight submissions.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(self, size: int, file_offset: int, buf_offset: int) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        return self._bytes_done.value


class AsyncHandle:
    """uGDS device handle wrapper, API-compatible with _cufile_async.AsyncHandle."""

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        device_path: str,
        writable: bool = True,
    ) -> None:
        fd = os.open(device_path, os.O_RDWR)
        try:
            handle = register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._handle = handle
        self.path = device_path
        self.writable = writable

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: int,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered uGDS handle.

        Matches _cufile_async.AsyncHandle.from_fd.
        """
        obj = cls.__new__(cls)
        obj._fd = fd
        obj._handle = handle
        obj.path = path
        obj.writable = writable
        return obj

    @property
    def fd(self) -> int:
        return self._fd

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        lib = _get_lib()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.uGDSReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "uGDSReadAsync",
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
        lib = _get_lib()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.uGDSWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "uGDSWriteAsync",
        )
        return sub

    def close(self) -> None:
        if self._fd < 0:
            return
        try:
            deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "AsyncHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
