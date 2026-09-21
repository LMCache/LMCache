# SPDX-License-Identifier: Apache-2.0
"""ugds implementation of the object-based async GDS interface.

Native libraries are loaded lazily. The backend owns driver state; its handles
keep it alive, and the context retains submissions until their DMA completes.
"""

# Standard
from typing import Optional
import ctypes
import ctypes.util
import os
import stat

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector._gds_async import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector._gds_driver import SharedDriver

logger = init_logger(__name__)


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


def _check(err: _uGDSError_t, op: str) -> None:
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: uGDSError(err={err.err}, cu_err={err.cu_err})"
        )


_UGDS_REGISTER_DMABUF = 0x1


def _buf_register_flags() -> int:
    """Return the ``uGDSBufRegister`` flags matching the PyTorch platform.

    Returns:
        ``_UGDS_REGISTER_DMABUF`` on ROCm, or ``0`` on CUDA.
    """
    return _UGDS_REGISTER_DMABUF if torch.version.hip is not None else 0


class UgdsBackend(GDSBackend):
    """Own the ugds driver and its registration operations."""

    name = "ugds"
    _driver = SharedDriver()

    def __init__(self) -> None:
        self._driver_opened = False
        self._lib: Optional[ctypes.CDLL] = None

    def open_slab(self, location: str, size: int, direct_io: bool) -> "AsyncHandle":
        """Validate a raw device and its capacity before allowing slab IO."""
        device_stat = os.stat(location)
        if not stat.S_ISCHR(device_stat.st_mode):
            raise ValueError(f"uGDS path must be a character device: {location}")
        major = os.major(device_stat.st_rdev)
        minor = os.minor(device_stat.st_rdev)
        subsystem = os.path.realpath(f"/sys/dev/char/{major}:{minor}/subsystem")
        if os.path.basename(subsystem) != "ugds_drv":
            raise ValueError(f"uGDS path is not managed by ugds_drv: {location}")
        if direct_io:
            logger.warning("use_direct_io is ignored by uGDS")
        slab = self.open_handle(os.open(location, os.O_RDWR), location)
        try:
            capacity = slab.capacity()
            if size > capacity:
                raise ValueError(
                    f"GDS L1 slab size ({size} bytes) exceeds backing device "
                    f"capacity ({capacity} bytes): {location}"
                )
        except Exception:
            try:
                slab.close()
            except Exception as cleanup_error:
                logger.warning("uGDS handle cleanup failed: %s", cleanup_error)
            raise
        return slab

    def validate_environment(self) -> None:
        """Preserve this implementation's existing PyTorch-build requirement."""
        if torch.version.hip is None and torch.version.cuda is None:
            raise ValueError("ugds requires a ROCm or CUDA PyTorch build")

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        """Take ownership of fd and register it; close fd on registration failure."""
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def close_driver(self) -> None:
        """Release this backend's driver ownership after its IO has completed."""
        if not self._driver_opened:
            return
        try:
            self._driver.release(self, self._close_driver)
        finally:
            self._driver_opened = False

    def register_handle(self, fd: int) -> int:
        """Register an open uGDS device fd and return the raw uGDSHandle_t.

        Mirrors _cufile_async.register_handle(fd): the caller owns the fd
        (typically an O_RDWR open of /dev/ugds_drvX) and closes it on
        registration failure. open_handle() also takes ownership of the fd.
        """
        self._ensure_driver_open()
        lib = self.library()
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

    def deregister_handle(self, handle: int) -> None:
        """Reverse of register_handle (uGDSHandleDeregister)."""
        lib = self.library()
        lib.uGDSHandleDeregister(ctypes.c_void_p(handle))

    def get_device_capacity(self, fd: int, handle: int) -> int:
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
            self.library().uGDSGetDeviceCapacity(
                ctypes.c_void_p(handle), ctypes.byref(capacity_bytes)
            ),
            "uGDSGetDeviceCapacity",
        )
        if capacity_bytes.value == 0:
            raise RuntimeError("uGDSGetDeviceCapacity returned zero capacity")
        return capacity_bytes.value

    def register_buffer(self, buf: torch.Tensor) -> None:
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on a CUDA or ROCm GPU")
        self._ensure_driver_open()
        lib = self.library()
        nbytes = buf.numel() * buf.element_size()
        _check(
            lib.uGDSBufRegister(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(_buf_register_flags()),
            ),
            "uGDSBufRegister",
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        lib = self.library()
        _check(
            lib.uGDSBufDeregister(ctypes.c_void_p(buf.data_ptr())),
            "uGDSBufDeregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        self._ensure_driver_open()
        lib = self.library()
        _check(
            lib.uGDSStreamRegister(ctypes.c_void_p(raw_stream)),
            "uGDSStreamRegister",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        lib = self.library()
        _check(
            lib.uGDSStreamDeregister(ctypes.c_void_p(raw_stream)),
            "uGDSStreamDeregister",
        )

    def library(self) -> ctypes.CDLL:
        if self._lib is not None:
            return self._lib
        search = ctypes.util.find_library("ugds")
        path = search or "libugds.so"
        lib = ctypes.CDLL(path)
        _declare_signatures(lib)
        self._lib = lib
        return lib

    def _ensure_driver_open(self) -> None:
        if self._driver_opened:
            return
        self._driver.acquire(self, self._open_driver)
        self._driver_opened = True

    def _open_driver(self) -> None:
        _check(self.library().uGDSDriverOpen(), "uGDSDriverOpen")

    def _close_driver(self) -> None:
        _check(self.library().uGDSDriverClose(), "uGDSDriverClose")


class AsyncHandle(GDSHandle):
    """An owning ugds slab handle with stream-ordered IO."""

    _backend: UgdsBackend

    def capacity(self) -> int:
        """Return this device's finite namespace capacity in bytes."""
        return self._backend.get_device_capacity(self.fd, self._handle)

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        lib = self._backend.library()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.uGDSReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
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
        lib = self._backend.library()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.uGDSWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "uGDSWriteAsync",
        )
        return sub
