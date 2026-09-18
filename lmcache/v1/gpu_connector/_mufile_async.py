# SPDX-License-Identifier: Apache-2.0
"""Minimal ctypes wrapper around the muFile async C API (Moore Threads MUSA).

MUSA analog of :mod:`lmcache.v1.gpu_connector._cufile_async`. muFile
(``SmartIO/mufile``) is the MUSA GPUDirect-Storage library; its async surface
is similar to cuFile (``muFileReadAsync`` + ``muFileStreamRegister`` + single
``musaStreamSynchronize``), so this module exposes a compatible set of
primitives and Python interface -- :class:`GDSContext` talks to either through
the :mod:`lmcache.v1.gpu_connector._gds_async` dispatch shim.

Surface (similar to ``_hipfile_async`` and ``_cufile_async``):

- :func:`register_buffer` / :func:`deregister_buffer` — wrap
  ``muFileBufRegister`` / ``muFileBufDeregister`` on a torch tensor.
- :func:`register_stream` / :func:`deregister_stream` — wrap
  ``muFileStreamRegister`` / ``muFileStreamDeregister`` on a raw MUSA
  stream handle.
- :func:`register_handle` / :func:`deregister_handle` — wrap
  ``muFileHandleRegister`` / ``muFileHandleDeregister`` on an fd.
- :class:`AsyncHandle` — opens a file with ``O_DIRECT`` (required by
  muFile) and registers the muFile handle. ``read_async`` /
  ``write_async`` enqueue an async IO on a stream and return a
  :class:`Submission`. Callers run ``musaStreamSynchronize`` once to drain a
  batch; :meth:`Submission.bytes_done` returns the actual byte count.

Unlike cuFile (which ships an ``nvidia.cufile`` Python binding), muFile has
no Python package, so every symbol -- driver lifecycle, handle/buffer
registration, and the ``MUfileError_t`` / ``MUFileDescr_t`` structs -- is
bound directly from ``libmufile.so`` via ctypes here. Requires SmartIO >=
the version with the async ABI documented in the LMCache design doc.
"""

# Standard
from typing import Any, Optional
import ctypes
import os
import threading

# Third Party
import torch

# ``libmufile.so`` is dlopened lazily (see ``_lib``) so importing this module
# on a CPU-only / non-MUSA host -- it is transitively pulled in by the GDS
# dispatch shim during CLI command discovery -- does not require the MUSA GPU
# IO driver to be present. This mirrors the lazy ``import cufile`` in the
# cuFile wrapper.

_LIBMUFILE_SONAME = "libmufile.so"

# muFile handle type for a POSIX fd (mufile.h: ``MU_FILE_HANDLE_TYPE_OPAQUE_FD``).
_MUFILE_HANDLE_TYPE_OPAQUE_FD = 1

# muFileStreamRegister flags (mufile.h): only MUFILE_STREAM_FIXED_AND_ALIGNED
# (0x1) is accepted by the library. Other flag values are rejected.
_STREAM_REGISTER_FLAGS = 0x1

# muFileSuccess (mufile.h ``MU_FILE_SUCCESS``).
_MUFILE_SUCCESS = 0

# Alignment requirement for muFile async I/O (4 KiB).
_MUFILE_ALIGNMENT = 4096


# --- Error enum mirror ------------------------------------------------------

# Partial mirror of MUfileOpError from mufile.h for diagnostic messages.
# Full enum is long; we only need the names that appear in common errors.
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
    """ctypes mirror of ``MUfileError_t`` (mufile.h).

    Single int field: ``err`` is the ``MUfileOpError_t`` status
    (0 == success).
    """

    _fields_ = [("err", ctypes.c_int)]


class _MUFileHandleUnion(ctypes.Union):
    """ctypes mirror of the ``MUFileDescr_t.handle`` union (fd or Win32 HANDLE)."""

    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class _MUFileDescr(ctypes.Structure):
    """ctypes mirror of ``MUFileDescr_t`` (mufile.h).

    Layout on LP64: type (4B) + padding (4B) + union (8B) = 16 bytes.
    """

    _fields_ = [
        ("type", ctypes.c_int),
        # 4 bytes padding here on LP64, ctypes handles it.
        ("handle", _MUFileHandleUnion),
    ]


# Guards the one-time dlopen + driver open/close below. Only the init/teardown
# transitions are serialized; the per-DMA fast path (``_lib()`` once loaded)
# never touches the lock.
_init_lock = threading.Lock()
_lib_handle: Optional[ctypes.CDLL] = None


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype on the libmufile symbols. Idempotent."""
    if getattr(lib.muFileReadAsync, "argtypes", None):
        return

    lib.muFileDriverOpen.argtypes = []
    lib.muFileDriverOpen.restype = _MUfileError

    lib.muFileDriverClose.argtypes = []
    lib.muFileDriverClose.restype = _MUfileError

    lib.muFileHandleRegister.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # MUFileHandle_t *fh
        ctypes.POINTER(_MUFileDescr),  # MUFileDescr_t *descr
    ]
    lib.muFileHandleRegister.restype = _MUfileError

    lib.muFileHandleDeregister.argtypes = [ctypes.c_void_p]  # MUFileHandle_t fh
    lib.muFileHandleDeregister.restype = _MUfileError

    lib.muFileBufRegister.argtypes = [
        ctypes.c_void_p,  # const void *buffer_base
        ctypes.c_size_t,  # size_t length
        ctypes.c_int,  # int flags
    ]
    lib.muFileBufRegister.restype = _MUfileError

    lib.muFileBufDeregister.argtypes = [ctypes.c_void_p]  # const void *buffer_base
    lib.muFileBufDeregister.restype = _MUfileError

    # Async I/O: size_t *bytes_*_p for completion, ssize_t return for status.
    lib.muFileReadAsync.argtypes = [
        ctypes.c_void_p,  # MUFileHandle_t fh
        ctypes.c_void_p,  # void *buffer_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *devPtr_offset_p
        ctypes.POINTER(ctypes.c_size_t),  # size_t *bytes_read_p
        ctypes.c_void_p,  # musaStream_t stream
    ]
    lib.muFileReadAsync.restype = ctypes.c_ssize_t

    lib.muFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.muFileWriteAsync.restype = ctypes.c_ssize_t

    lib.muFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.muFileStreamRegister.restype = _MUfileError

    lib.muFileStreamDeregister.argtypes = [ctypes.c_void_p]
    lib.muFileStreamDeregister.restype = _MUfileError


_driver_opened = False


def _lib() -> ctypes.CDLL:
    """dlopen ``libmufile.so`` once and declare signatures. Idempotent.

    Thread-safe via double-checked locking: the common case (already loaded)
    returns without taking ``_init_lock``, so the per-DMA path stays lock-free.

    Returns:
        The process-global ``libmufile.so`` handle.
    """
    global _lib_handle
    if _lib_handle is not None:
        return _lib_handle
    with _init_lock:
        if _lib_handle is None:
            handle = ctypes.CDLL(_LIBMUFILE_SONAME)
            _declare_signatures(handle)
            _lib_handle = handle
    return _lib_handle


def _ensure_driver_open() -> None:
    """Idempotently open the muFile driver (thread-safe).

    Raises:
        RuntimeError: If ``muFileDriverOpen`` reports a non-success status.
    """
    global _driver_opened
    if _driver_opened:
        return
    # ``_lib()`` does its own locking; call it before taking ``_init_lock`` so
    # the (non-reentrant) lock is never acquired twice on the same thread.
    lib = _lib()
    with _init_lock:
        if _driver_opened:
            return
        _check(lib.muFileDriverOpen(), "muFileDriverOpen")
        _driver_opened = True


def close_driver() -> None:
    """Close the muFile driver (thread-safe). Optional — useful in tests.

    Raises:
        RuntimeError: If ``muFileDriverClose`` reports a non-success status.
    """
    global _driver_opened
    if not _driver_opened:
        return
    lib = _lib()
    with _init_lock:
        if not _driver_opened:
            return
        try:
            _check(lib.muFileDriverClose(), "muFileDriverClose")
        finally:
            _driver_opened = False


def _op_error_name(err_code: int) -> str:
    """Return the human-readable name for a ``MUfileOpError_t`` value."""
    return _OP_ERROR_NAMES.get(abs(err_code), f"MU_FILE_ERROR_{abs(err_code)}")


def _check(err: "_MUfileError", op: str) -> None:
    """Convert a non-zero ``MUfileError_t`` into a Python exception."""
    if err.err != _MUFILE_SUCCESS:
        raise RuntimeError(
            f"{op} failed: muFileError(err={err.err} [{_op_error_name(err.err)}])"
        )


# --- Handle registration -------------------------------------------

# muFile's handle points into the caller's MUFileDescr_t, so each descriptor
# must outlive its handle. Registered descriptors are kept here until
# muFileHandleDeregister releases them.
_handle_descr_registry: dict[int, _MUFileDescr] = {}
_handle_registry_lock = threading.Lock()


def register_handle(fd: int) -> int:
    """Register an open fd with muFile and return the ``MUFileHandle_t``.

    Opens the muFile driver on first use. The returned handle is the raw
    ``void *`` value (as an int), accepted directly as the first argument of
    ``muFileReadAsync`` / ``muFileWriteAsync``. The descriptor backing the
    registration is retained by this module until :func:`deregister_handle`.

    Args:
        fd: An open file descriptor for the slab (opened with ``O_DIRECT`` for
            the GDS fast path).

    Returns:
        The registered ``MUFileHandle_t`` as an integer.

    Raises:
        RuntimeError: If ``muFileHandleRegister`` reports a non-success status.
    """
    _ensure_driver_open()
    lib = _lib()
    handle = ctypes.c_void_p()
    descr = _MUFileDescr()
    descr.type = _MUFILE_HANDLE_TYPE_OPAQUE_FD
    descr.handle.fd = fd
    _check(
        lib.muFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr)),
        "muFileHandleRegister",
    )
    value = handle.value if handle.value is not None else 0
    with _handle_registry_lock:
        _handle_descr_registry[value] = descr
    return value


def deregister_handle(handle: int) -> None:
    """Reverse of :func:`register_handle` (``muFileHandleDeregister``).

    Also releases the retained descriptor, which may not outlive the handle.

    Args:
        handle: The ``MUFileHandle_t`` (as an int) from :func:`register_handle`.

    Raises:
        RuntimeError: If ``muFileHandleDeregister`` reports a non-success status.
    """
    with _handle_registry_lock:
        _handle_descr_registry.pop(handle, None)
    _check(
        _lib().muFileHandleDeregister(ctypes.c_void_p(handle)),
        "muFileHandleDeregister",
    )


# --- Buffer / stream registration ----------------------------------


def register_buffer(buf: torch.Tensor) -> None:
    """Register a device tensor with muFile for GPUDirect Storage DMA.

    Must be called before any ``read_async`` / ``write_async`` whose
    ``buf_base`` falls inside this tensor's allocation. Implicitly opens the
    muFile driver on first use.

    Args:
        buf: A GPU (MUSA device) tensor to register for DMA.

    Raises:
        ValueError: If ``buf`` is not on the GPU.
        RuntimeError: If ``muFileBufRegister`` reports a non-success status.
    """
    if not getattr(buf, "is_musa", False):
        raise ValueError("register_buffer: tensor must be on the MUSA device")
    _ensure_driver_open()
    lib = _lib()
    nbytes = buf.numel() * buf.element_size()
    _check(
        lib.muFileBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(0),
        ),
        "muFileBufRegister",
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`.

    Args:
        buf: A tensor previously passed to :func:`register_buffer`.

    Raises:
        RuntimeError: If ``muFileBufDeregister`` reports a non-success status.
    """
    _check(
        _lib().muFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "muFileBufDeregister",
    )


def register_stream(raw_stream: int) -> None:
    """Register a MUSA stream with muFile.

    ``raw_stream`` is the integer ``musaStream_t`` handle — get it via
    ``stream.musa_stream`` or ``stream.ptr`` on MUSA platforms.

    Optional for correctness (``read_async`` / ``write_async`` also take the
    stream per call). Registered with the FIXED_AND_ALIGNED flag (0x1): muFile
    still reads the size/offset pointers at stream-execution time -- so their
    storage must stay alive and unchanged until completion (see ``Submission``)
    -- but promising the values are fixed at submission lets muFile skip
    per-op setup.

    Args:
        raw_stream: The integer ``musaStream_t`` handle to register.

    Raises:
        RuntimeError: If ``muFileStreamRegister`` reports a non-success status.
        ValueError: If flags other than 0x1 are requested (muFile only accepts
            MUFILE_STREAM_FIXED_AND_ALIGNED).
    """
    _ensure_driver_open()
    _check(
        _lib().muFileStreamRegister(
            ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
        ),
        "muFileStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream`.

    Args:
        raw_stream: The ``musaStream_t`` handle passed to :func:`register_stream`.

    Raises:
        RuntimeError: If ``muFileStreamDeregister`` reports a non-success status.
    """
    _check(
        _lib().muFileStreamDeregister(ctypes.c_void_p(raw_stream)),
        "muFileStreamDeregister",
    )


# --- AsyncHandle + Submission --------------------------------------


class Submission:
    """One in-flight ``muFileReadAsync`` / ``muFileWriteAsync``.

    Holds the host-side ``size_p`` / ``file_offset_p`` / ``buf_offset_p`` /
    ``bytes_done_p`` storage that muFile writes into asynchronously. These
    ctypes objects MUST stay alive until the stream actually executes the op —
    keep the :class:`Submission` reference (or stash it in a list) until after
    the stream sync.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(
        self,
        size: int,
        file_offset: int,
        buf_offset: int,
    ) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_size_t(0)

    @property
    def bytes_done(self) -> int:
        """Bytes actually transferred. Valid only AFTER the stream sync."""
        return self._bytes_done.value


def _check_alignment(
    buf_base: int, size: int, file_offset: int, buf_offset: int
) -> None:
    """Raise ValueError if any async I/O operand is not 4 KiB aligned.

    muFile's async path requires page-aligned base, size, file offset, and
    device offset. This wrapper enforces that contract before calling the
    library.
    """
    if buf_base % _MUFILE_ALIGNMENT != 0:
        raise ValueError(
            f"buf_base 0x{buf_base:x} is not {_MUFILE_ALIGNMENT}-byte aligned"
        )
    if size % _MUFILE_ALIGNMENT != 0:
        raise ValueError(f"size {size} is not {_MUFILE_ALIGNMENT}-byte aligned")
    if file_offset % _MUFILE_ALIGNMENT != 0:
        raise ValueError(
            f"file_offset {file_offset} is not {_MUFILE_ALIGNMENT}-byte aligned"
        )
    if buf_offset % _MUFILE_ALIGNMENT != 0:
        raise ValueError(
            f"buf_offset {buf_offset} is not {_MUFILE_ALIGNMENT}-byte aligned"
        )


class AsyncHandle:
    """Open file + muFile handle wrapper.

    Opens with ``O_DIRECT`` (required for muFile's GPUDirect Storage fast
    path). Optionally pre-allocates the file via ``posix_fallocate``.
    """

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        path: str,
        writable: bool = False,
        fallocate_size: Optional[int] = None,
        mode: int = 0o644,
    ) -> None:
        flags = os.O_DIRECT
        if writable:
            flags |= os.O_CREAT | os.O_RDWR
        else:
            flags |= os.O_RDONLY
        self.path = path
        self.writable = writable
        self._fd = os.open(path, flags, mode)
        try:
            if fallocate_size is not None and writable:
                os.posix_fallocate(self._fd, 0, fallocate_size)
            self._handle = register_handle(self._fd)
        except Exception:
            os.close(self._fd)
            raise

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: int,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered muFile handle.

        For callers that open + register the file themselves (e.g. a slab that
        must be created, truncated, and ``posix_fallocate``d before
        ``muFileHandleRegister``) and just need an ``AsyncHandle`` around the
        result. The descriptor is NOT retained; the caller must keep the fd
        open until ``close()``.
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
        """Enqueue a ``muFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g. ``buf.data_ptr()``).
        ``buf_offset`` is the byte offset within that registration that the data
        should land at.
        """
        _check_alignment(buf_base, size, file_offset, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        ret = _lib().muFileReadAsync(
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(buf_base),
            ctypes.byref(sub._size),
            ctypes.byref(sub._file_offset),
            ctypes.byref(sub._buf_offset),
            ctypes.byref(sub._bytes_done),
            ctypes.c_void_p(raw_stream),
        )
        if ret < 0:
            raise RuntimeError(f"muFileReadAsync failed: {_op_error_name(int(ret))}")
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``muFileWriteAsync`` on the stream."""
        _check_alignment(buf_base, size, file_offset, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        ret = _lib().muFileWriteAsync(
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(buf_base),
            ctypes.byref(sub._size),
            ctypes.byref(sub._file_offset),
            ctypes.byref(sub._buf_offset),
            ctypes.byref(sub._bytes_done),
            ctypes.c_void_p(raw_stream),
        )
        if ret < 0:
            raise RuntimeError(f"muFileWriteAsync failed: {_op_error_name(int(ret))}")
        return sub

    def close(self) -> None:
        """Deregister the muFile handle and close the fd."""
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

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()
