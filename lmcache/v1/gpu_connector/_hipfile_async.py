# SPDX-License-Identifier: Apache-2.0
"""Minimal ctypes wrapper around the hipFile async C API (AMD ROCm).

AMD analog of :mod:`lmcache.v1.gpu_connector._cufile_async`. hipFile
(``ROCm/hipFile``, "AMD Infinity Storage") is the ROCm GPUDirect-Storage
library; its async surface mirrors cuFile 1:1 (``hipFileReadAsync`` +
``hipFileStreamRegister`` + batched submit + single ``hipStreamSynchronize``),
so this module exposes the same primitives and the **same Python surface** as
the cuFile wrapper -- :class:`GDSContext` talks to either through the
:mod:`lmcache.v1.gpu_connector._gds_async` dispatch shim.

Surface (identical to ``_cufile_async``):

- :func:`register_buffer` / :func:`deregister_buffer` — wrap
  ``hipFileBufRegister`` / ``hipFileBufDeregister`` on a torch tensor.
- :func:`register_stream` / :func:`deregister_stream` — wrap
  ``hipFileStreamRegister`` / ``hipFileStreamDeregister`` on a raw HIP
  stream handle.
- :func:`register_handle` / :func:`deregister_handle` — wrap
  ``hipFileHandleRegister`` / ``hipFileHandleDeregister`` on an fd.
- :class:`AsyncHandle` — opens a file with ``O_DIRECT`` (required by
  hipFile) and registers the hipFile handle. ``read_async`` /
  ``write_async`` enqueue an async IO on a stream and return a
  :class:`Submission`. Callers run ``hipStreamSynchronize`` once to drain a
  batch; :meth:`Submission.bytes_done` returns the actual byte count.

Unlike cuFile (which ships an ``nvidia.cufile`` Python binding), hipFile has
no Python package, so every symbol -- driver lifecycle, handle/buffer
registration, and the ``hipFileError_t`` / ``hipFileDescr_t`` structs -- is
bound directly from ``libhipfile.so`` via ctypes here. Requires ROCm >= 7.2.0.
"""

# Standard
from typing import Any, Optional
import ctypes
import os
import threading

# Third Party
import torch

# ``libhipfile.so`` is dlopened lazily (see ``_lib``) so importing this module
# on a CPU-only / NVIDIA host -- it is transitively pulled in by the GDS
# dispatch shim during CLI command discovery -- does not require the ROCm GPU
# IO driver to be present. This mirrors the lazy ``import cufile`` in the cuFile
# wrapper.

_LIBHIPFILE_SONAME = "libhipfile.so"

# hipFile handle type for a POSIX fd (hipfile.h: ``hipFileHandleTypeOpaqueFD``).
_HIPFILE_HANDLE_TYPE_OPAQUE_FD = 1

# hipFileStreamRegister flags (hipfile.h): buffer offset, file offset, and size
# are all fixed at submission time (HIPFILE_STREAM_FIXED_* = 0x1|0x2|0x4). Same
# 0x7 the cuFile wrapper uses. PAGE_ALIGNED_INPUTS (0x8) is omitted -- transfer
# sizes are not always 4 KiB.
_STREAM_REGISTER_FLAGS = 0x7

# hipFileSuccess (hipfile.h ``hipFileOpError``).
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


# Guards the one-time dlopen + driver open/close below. Only the init/teardown
# transitions are serialized; the per-DMA fast path (``_lib()`` once loaded)
# never touches the lock.
_init_lock = threading.Lock()
_lib_handle: Optional[ctypes.CDLL] = None


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


_driver_opened = False


def _lib() -> ctypes.CDLL:
    """dlopen ``libhipfile.so`` once and declare signatures. Idempotent.

    Thread-safe via double-checked locking: the common case (already loaded)
    returns without taking ``_init_lock``, so the per-DMA path stays lock-free.

    Returns:
        The process-global ``libhipfile.so`` handle.
    """
    global _lib_handle
    if _lib_handle is not None:
        return _lib_handle
    with _init_lock:
        if _lib_handle is None:
            handle = ctypes.CDLL(_LIBHIPFILE_SONAME)
            _declare_signatures(handle)
            _lib_handle = handle
    return _lib_handle


def _ensure_driver_open() -> None:
    """Idempotently open the hipFile driver (thread-safe).

    Raises:
        RuntimeError: If ``hipFileDriverOpen`` reports a non-success status.
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
        _check(lib.hipFileDriverOpen(), "hipFileDriverOpen")
        _driver_opened = True


def close_driver() -> None:
    """Close the hipFile driver (thread-safe). Optional — useful in tests.

    Raises:
        RuntimeError: If ``hipFileDriverClose`` reports a non-success status.
    """
    global _driver_opened
    if not _driver_opened:
        return
    lib = _lib()
    with _init_lock:
        if not _driver_opened:
            return
        try:
            _check(lib.hipFileDriverClose(), "hipFileDriverClose")
        finally:
            _driver_opened = False


def _op_error_string(err_code: int) -> str:
    """Return the human-readable name for a ``hipFileOpError_t`` value."""
    raw = _lib().hipFileGetOpErrorString(ctypes.c_int(abs(err_code)))
    return raw.decode() if raw is not None else "unknown"


def _check(err: "_HipFileError", op: str) -> None:
    """Convert a non-zero ``hipFileError_t`` into a Python exception."""
    if err.err != _HIPFILE_SUCCESS:
        raise RuntimeError(
            f"{op} failed: hipFileError(err={err.err} [{_op_error_string(err.err)}], "
            f"hip_drv_err={err.hip_drv_err})"
        )


# --- Handle registration -------------------------------------------


def register_handle(fd: int) -> int:
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
    _ensure_driver_open()
    lib = _lib()
    handle = ctypes.c_void_p()
    descr = _HipFileDescr()
    descr.type = _HIPFILE_HANDLE_TYPE_OPAQUE_FD
    descr.handle.fd = fd
    descr.fs_ops = None
    _check(
        lib.hipFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr)),
        "hipFileHandleRegister",
    )
    return handle.value if handle.value is not None else 0


def deregister_handle(handle: int) -> None:
    """Reverse of :func:`register_handle` (``hipFileHandleDeregister``).

    Args:
        handle: The ``hipFileHandle_t`` (as an int) from :func:`register_handle`.
    """
    _lib().hipFileHandleDeregister(ctypes.c_void_p(handle))


# --- Buffer / stream registration ----------------------------------


def register_buffer(buf: torch.Tensor) -> None:
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
    _ensure_driver_open()
    lib = _lib()
    nbytes = buf.numel() * buf.element_size()
    _check(
        lib.hipFileBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(0),
        ),
        "hipFileBufRegister",
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`.

    Args:
        buf: A tensor previously passed to :func:`register_buffer`.

    Raises:
        RuntimeError: If ``hipFileBufDeregister`` reports a non-success status.
    """
    _check(
        _lib().hipFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "hipFileBufDeregister",
    )


def register_stream(raw_stream: int) -> None:
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
    _ensure_driver_open()
    _check(
        _lib().hipFileStreamRegister(
            ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
        ),
        "hipFileStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream`.

    Args:
        raw_stream: The ``hipStream_t`` handle passed to :func:`register_stream`.

    Raises:
        RuntimeError: If ``hipFileStreamDeregister`` reports a non-success status.
    """
    _check(
        _lib().hipFileStreamDeregister(ctypes.c_void_p(raw_stream)),
        "hipFileStreamDeregister",
    )


# --- AsyncHandle + Submission --------------------------------------


class Submission:
    """One in-flight ``hipFileReadAsync`` / ``hipFileWriteAsync``.

    Holds the host-side ``size_p`` / ``file_offset_p`` / ``buf_offset_p`` /
    ``bytes_done_p`` storage that hipFile writes into asynchronously. These
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
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        """Bytes actually transferred. Valid only AFTER the stream sync."""
        return self._bytes_done.value


class AsyncHandle:
    """Open file + hipFile handle wrapper.

    Opens with ``O_DIRECT`` (required for hipFile's GPUDirect Storage fast
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
        handle: Any,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered hipFile handle.

        For callers that open + register the file themselves (e.g. a slab that
        must be created, truncated, and ``posix_fallocate``d before
        ``hipFileHandleRegister``) and just need an ``AsyncHandle`` around the
        result.
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
        """Enqueue a ``hipFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g. ``buf.data_ptr()``).
        ``buf_offset`` is the byte offset within that registration that the data
        should land at.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            _lib().hipFileReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
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
        _check(
            _lib().hipFileWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "hipFileWriteAsync",
        )
        return sub

    def close(self) -> None:
        """Deregister the hipFile handle and close the fd."""
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
