# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper around the Phoenix frozen file-IO ABI (``libphxfile.so``).

Backend for the GDS L1 tier's async dispatch shim
(:mod:`lmcache.v1.gpu_connector._gds_async`). ``libphxfile.so`` is a thin
frozen-ABI layer maintained in the Phoenix tree
(``phoenix/adapters/lmcache/phxfile``): the ``phxFile*`` symbols are frozen
— names, signatures and semantics never change — so libphoenix evolution
(e.g. dropping the device parameter from ``phxfs_regmem``) is absorbed
inside that library. This wrapper only ever needs the library reinstalled,
never re-coded.

Naming and semantics deliberately mirror AMD hipFile
(:mod:`lmcache.v1.gpu_connector._hipfile_async`) so the four GDS backend
wrappers (cuFile / hipFile / uGDS / phx) stay line-by-line analogous:

- :func:`register_handle` wraps ``phxFileHandleRegister`` — currently an
  identity boxing of the POSIX fd (phxfs reads/writes plain fds).
- :func:`register_buffer` wraps ``phxFileBufRegister`` — the frozen,
  device-free entry point. The shim resolves the buffer's device itself
  (probe-based, like libphoenix's own stream path resolves buffers at IO
  time); page-size alignment and the registration bookkeeping also live
  inside the shim.
- :func:`register_stream` / :func:`deregister_stream` wrap
  ``phxFileStreamRegister`` / ``phxFileStreamDeregister`` — frozen no-ops
  today (every phxfs submission carries the stream), reserved for future
  per-stream resource pre-claiming.

Execution semantics (stream-ordered, cuFile-compatible): submissions
enqueue a DMA that is ordered on the caller's CUDA/ROCm stream -- after
everything the caller enqueued before the submission, and before
everything enqueued after it. The submission returns immediately; the
transfer outcome lands in :attr:`Submission.bytes_done` once the stream
is synchronized past it. The :class:`Submission`'s ctypes storage is
handed to the C API by reference (late-binding) and must stay alive until
then (the caller -- :mod:`lmcache.v1.gpu_connector.gds_context` -- keeps
submissions behind a GPU event checkpoint).

``libphxfile.so`` (resolving via ``ldconfig`` / ``LD_LIBRARY_PATH``, after
``bash phoenix/adapters/lmcache/phxfile/install.sh``) must expose the
frozen ``phxFile*`` surface: loading fails fast otherwise.
"""

# Standard
from typing import Any, Optional
import ctypes
import ctypes.util
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# --- libphxfile.so lazy loading ------------------------------------------

_lib: Optional[ctypes.CDLL] = None


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


def _get_lib() -> ctypes.CDLL:
    """Load ``libphxfile.so`` on first use and declare the frozen ABI."""
    global _lib
    if _lib is not None:
        return _lib
    search = ctypes.util.find_library("phxfile")
    path = search or "libphxfile.so"
    lib = ctypes.CDLL(path)
    _declare_signatures(lib, path)
    _lib = lib
    return lib


# --- Error checking ---------------------------------------------------


def _check(rc: int, op: str) -> None:
    """Convert a negative phxFile return code into a Python exception."""
    if rc < 0:
        try:
            why = os.strerror(-rc)
        except ValueError:
            why = "unknown error"
        raise RuntimeError(f"{op} failed: phxFileError(rc={rc} [{why}])")


# --- Backend surface (contract of _gds_async) ----------------------------


def register_handle(fd: int) -> int:
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
    lib = _get_lib()
    fh = ctypes.c_void_p()
    _check(
        int(lib.phxFileHandleRegister(ctypes.byref(fh), ctypes.c_int(fd))),
        "phxFileHandleRegister",
    )
    if fh.value is None:
        raise RuntimeError("phxFileHandleRegister returned a null handle")
    return fh.value


def deregister_handle(handle: int) -> None:
    """Reverse of :func:`register_handle` (``phxFileHandleDeregister``)."""
    _get_lib().phxFileHandleDeregister(ctypes.c_void_p(handle))


def register_buffer(buf: torch.Tensor) -> None:
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
        _get_lib().phxFileBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
        ),
        "phxFileBufRegister",
    )
    logger.debug(
        "_phx_async: registered 0x%x (%d bytes) via libphxfile probe",
        buf.data_ptr(),
        nbytes,
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`.

    Wraps ``phxFileBufDeregister``: only the base address is passed; the
    aligned registration length and phxfs device are played back from the
    shim's bookkeeping. Unregistered buffers are silently tolerated
    inside the shim (matching the tolerance of the cuFile path teardown).

    Args:
        buf: The tensor previously passed to :func:`register_buffer`.

    Raises:
        RuntimeError: If ``phxFileBufDeregister`` fails.
    """
    _check(
        _get_lib().phxFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "phxFileBufDeregister",
    )


def register_stream(raw_stream: int) -> None:
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
        _get_lib().phxFileStreamRegister(ctypes.c_void_p(raw_stream)),
        "phxFileStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream` (``phxFileStreamDeregister``)."""
    _check(
        _get_lib().phxFileStreamDeregister(ctypes.c_void_p(raw_stream)),
        "phxFileStreamDeregister",
    )


def close_driver() -> None:
    """Release every shim-side registration and close all opened devices.

    Wraps ``phxFileDriverClose``: the shim sweeps any buffer registration
    still in its table, closes every phxfs device it opened, and resets
    its caches. Individual cleanup failures are reported by the shim on
    stderr and do not raise.
    """
    _check(_get_lib().phxFileDriverClose(), "phxFileDriverClose")


# --- Submission + AsyncHandle --------------------------------------------


class Submission:
    """One in-flight (stream-ordered) phx IO.

    Mirrors :class:`_cufile_async.Submission`: holds the transfer
    parameters and the ``bytes_done`` result storage. The ctypes fields
    are handed to the frozen C API by reference (late-binding), so the
    keep-alive-until-stream-sync contract documented there applies
    unchanged (the GDS context's event checkpoint owns that lifetime).
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
    """Slab-file handle wrapper, API-compatible with _cufile_async.AsyncHandle."""

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
        """Wrap an already-opened fd (``handle`` is the fd itself for phx)."""
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
            _get_lib().phxFileReadAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
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
            _get_lib().phxFileWriteAsync(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "phxFileWriteAsync",
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
