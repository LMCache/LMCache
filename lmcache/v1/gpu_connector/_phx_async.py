# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper around the Phoenix phxfs C API (``libphoenix.so``).

Backend for the GDS L1 tier's async dispatch shim
(:mod:`lmcache.v1.gpu_connector._gds_async`). Phoenix (phxfs) DMAs NVMe
data straight into registered GPU buffers like cuFile/hipFile/uGDS, so the
common backend surface maps naturally:

- :func:`register_buffer` wraps ``phxfs_regmem`` (lazily opens the phxfs
  device and registers the buffer 64 KiB-aligned).
- :func:`register_handle` is a passthrough: phxfs operates on plain POSIX
  fds, so the "handle" is the fd itself.
- :func:`register_stream` / :func:`deregister_stream` are no-ops: phxfs
  has no stream registration (every submission carries the stream); the
  functions exist for the shared backend surface. IO submissions are
  stream-ordered via ``phxfs_read_stream`` / ``phxfs_write_stream``.

Execution semantics (stream-ordered, cuFile-compatible): submissions
enqueue a DMA that is ordered on the caller's CUDA/ROCm stream -- after
everything the caller enqueued before the submission, and before
everything enqueued after it. The submission returns immediately; the
transfer outcome lands in :attr:`Submission.bytes_done` once the stream
is synchronized past it. The :class:`Submission`'s ctypes storage is
handed to the C API by reference and must stay alive until then (the
caller -- :mod:`lmcache.v1.gpu_connector.gds_context` -- keeps
submissions behind a GPU event checkpoint).

``libphoenix.so`` (resolving via ``ldconfig`` / ``LD_LIBRARY_PATH``) must
expose the stream-ordered API (``phxfs_read_stream`` /
``phxfs_write_stream``): loading fails fast otherwise (there is no
synchronous fallback).
"""

# Standard
from typing import Any, Optional
import bisect
import ctypes
import ctypes.util
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# --- libphoenix.so lazy loading -----------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _declare_signatures(lib: ctypes.CDLL, path_hint: str) -> None:
    """Set argtypes/restype on the phxfs symbols used by this module."""
    # Stream-ordered API is required (no synchronous fallback): check for
    # the symbols before touching anything else, so a pre-stream
    # libphoenix fails fast with a clear message.
    missing = [
        sym
        for sym in ("phxfs_read_stream", "phxfs_write_stream")
        if not hasattr(lib, sym)
    ]
    if missing:
        raise RuntimeError(
            f"libphoenix at {path_hint} lacks the stream-ordered API "
            f"({', '.join(missing)}); build libphoenix with stream support"
        )

    lib.phxfs_find_dev.argtypes = [ctypes.c_int]
    lib.phxfs_find_dev.restype = ctypes.c_int

    lib.phxfs_open.argtypes = [ctypes.c_int]
    lib.phxfs_open.restype = ctypes.c_int

    lib.phxfs_close.argtypes = [ctypes.c_int]
    lib.phxfs_close.restype = ctypes.c_int

    lib.phxfs_get_page_size.argtypes = []
    lib.phxfs_get_page_size.restype = ctypes.c_uint64

    lib.phxfs_regmem.argtypes = [
        ctypes.c_int,  # int device_id (phxfs index)
        ctypes.c_void_p,  # const void *addr (device address)
        ctypes.c_size_t,  # size_t len
        ctypes.POINTER(ctypes.c_void_p),  # void **target_addr (out)
    ]
    lib.phxfs_regmem.restype = ctypes.c_int

    lib.phxfs_deregmem.argtypes = [
        ctypes.c_int,  # int device_id (phxfs index)
        ctypes.c_void_p,  # const void *addr
        ctypes.c_size_t,  # size_t len
    ]
    lib.phxfs_deregmem.restype = ctypes.c_int

    stream_params: list[type] = [
        ctypes.c_int,  # int fd
        ctypes.c_void_p,  # void *buf
        ctypes.POINTER(ctypes.c_size_t),  # size_t *nbytes
        ctypes.POINTER(ctypes.c_int64),  # off_t *buf_offset
        ctypes.POINTER(ctypes.c_int64),  # off_t *f_offset
        ctypes.POINTER(ctypes.c_ssize_t),  # ssize_t *bytes_done
        ctypes.c_void_p,  # void *stream (vendor-opaque)
    ]
    lib.phxfs_read_stream.argtypes = stream_params
    lib.phxfs_read_stream.restype = ctypes.c_int
    lib.phxfs_write_stream.argtypes = stream_params
    lib.phxfs_write_stream.restype = ctypes.c_int


def _get_lib() -> ctypes.CDLL:
    """Load ``libphoenix.so`` on first use and declare the phxfs ABI."""
    global _lib
    if _lib is not None:
        return _lib
    search = ctypes.util.find_library("phoenix")
    path = search or "libphoenix.so"
    lib = ctypes.CDLL(path)
    _declare_signatures(lib, path)
    _lib = lib
    return _lib


# --- Error checking ---------------------------------------------------


def _check(rc: int, op: str) -> None:
    """Convert a negative phxfs return code into a Python exception."""
    if rc < 0:
        try:
            why = os.strerror(-rc)
        except ValueError:
            why = "unknown error"
        raise RuntimeError(f"{op} failed: phxfsError(rc={rc} [{why}])")


# --- Device + buffer registration state ----------------------------------
#
# phxfs devices are opened once per CUDA/HIP ordinal (find_dev + open) and
# cached. Registered buffers are tracked in a sorted table so that
# deregistration passes the same (aligned) length that was registered and
# so duplicate registrations are detected. (The IO path resolves the
# buffer device inside libphoenix — this table is not consulted for IO.)

_state_lock = threading.Lock()
_devices: dict[int, int] = {}
"""CUDA/HIP ordinal -> opened phxfs device index."""
_reg_bases: list[int] = []
"""Sorted registered buffer base pointers (parallel to ``_reg_entries``)."""
_reg_entries: list[tuple[int, int]] = []
"""(aligned_length, phxfs_device_index) per entry in ``_reg_bases``."""

_page_size: int = 0
"""Cached device page size in bytes (64 KiB on NVIDIA); 0 = not queried."""

_MAP_MODE_NAMES = {0: "FULL", 1: "STAGING"}


def _align_up(size: int, alignment: int) -> int:
    return (size + alignment - 1) & ~(alignment - 1)


def _get_page_size_locked(lib: ctypes.CDLL) -> int:
    """Return the (cached) device page size. Caller holds ``_state_lock``."""
    global _page_size
    if _page_size == 0:
        _page_size = int(lib.phxfs_get_page_size())
        if _page_size <= 0:
            raise RuntimeError(
                f"phxfs_get_page_size returned invalid value {_page_size}"
            )
    return _page_size


def _ensure_device_open_locked(lib: ctypes.CDLL, cuda_ordinal: int) -> int:
    """Open the phxfs device for ``cuda_ordinal`` once. Caller holds the lock.

    Returns the phxfs device index used for regmem / deregmem.

    Raises:
        RuntimeError: If ``phxfs_find_dev`` or ``phxfs_open`` fails.
    """
    phxfs_dev = _devices.get(cuda_ordinal)
    if phxfs_dev is not None:
        return phxfs_dev
    phxfs_dev = int(lib.phxfs_find_dev(cuda_ordinal))
    _check(phxfs_dev, f"phxfs_find_dev({cuda_ordinal})")
    _check(int(lib.phxfs_open(phxfs_dev)), f"phxfs_open({phxfs_dev})")
    _devices[cuda_ordinal] = phxfs_dev
    try:
        map_mode = int(lib.phxfs_get_map_mode(phxfs_dev))
        mode_name = _MAP_MODE_NAMES.get(map_mode, f"unknown({map_mode})")
    except AttributeError:
        mode_name = "unknown"
    logger.info(
        "_phx_async: phxfs device %d opened for GPU %d (map_mode=%s, page_size=%d KiB)",
        phxfs_dev,
        cuda_ordinal,
        mode_name,
        _get_page_size_locked(lib) // 1024,
    )
    return phxfs_dev


# --- Backend surface (contract of _gds_async) ----------------------------


def register_handle(fd: int) -> int:
    """Accept an open fd for phxfs IO and return the "handle".

    phxfs reads and writes plain POSIX fds (no library-side handle
    registration), so the fd itself is the handle; :class:`AsyncHandle`
    round-trips it and closes the fd on ``close()``. Loads ``libphoenix``
    eagerly so a missing library fails at slab setup, not at first DMA.

    Args:
        fd: Open slab-file descriptor (as created by
            :meth:`gds_context.GDSContext.initialize`).

    Returns:
        ``fd`` unchanged.
    """
    _get_lib()
    return fd


def deregister_handle(handle: int) -> None:
    """Reverse of :func:`register_handle`: nothing to do for phxfs."""
    return


def register_buffer(buf: torch.Tensor) -> None:
    """Register a device tensor with phxfs for GDS DMA.

    Opens the phxfs device for the tensor's GPU on first use, then
    registers the buffer with ``phxfs_regmem``. phxfs requires the
    registration length to be device-page aligned (64 KiB on NVIDIA), so
    the length is rounded up; the IO size is independent of the
    registration length and never exceeds the buffer.

    Args:
        buf: Contiguous GPU tensor (a <=16 MiB slice of a staging buffer,
            as passed by :meth:`gds_context.GDSContext.register_gpu_buffer`).

    Raises:
        ValueError: If ``buf`` is not a GPU tensor or is empty.
        RuntimeError: If the phxfs device cannot be opened or the
            registration is rejected.
    """
    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on a CUDA or ROCm GPU")
    nbytes = buf.numel() * buf.element_size()
    if nbytes == 0:
        raise ValueError("register_buffer: tensor is empty")
    base = buf.data_ptr()
    cuda_ordinal = buf.device.index
    if cuda_ordinal is None:
        cuda_ordinal = torch.cuda.current_device()
    lib = _get_lib()
    with _state_lock:
        phxfs_dev = _ensure_device_open_locked(lib, cuda_ordinal)
        page_size = _get_page_size_locked(lib)
        aligned_len = _align_up(nbytes, page_size)
        target_addr = ctypes.c_void_p()
        _check(
            int(
                lib.phxfs_regmem(
                    phxfs_dev, base, aligned_len, ctypes.byref(target_addr)
                )
            ),
            "phxfs_regmem",
        )
        # NOTE: target_addr is an internal host-mapped handle; IO must use
        # the original device address, so it is deliberately not recorded.
        # Duplicate/overlap detection is entirely the library's job
        # (phxfs_regmem reference-counts exact duplicates and rejects
        # overlapping ranges); every successful call is recorded so
        # register/deregister stay symmetric.
        idx = bisect.bisect_left(_reg_bases, base)
        _reg_bases.insert(idx, base)
        _reg_entries.insert(idx, (aligned_len, phxfs_dev))
    logger.debug(
        "_phx_async: registered 0x%x..0x%x on phxfs device %d",
        base,
        base + aligned_len,
        phxfs_dev,
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`.

    Deregisters the buffer's registration with ``phxfs_deregmem`` using
    the same aligned length that was registered. Unregistered buffers are
    ignored (matching the tolerance of the cuFile path teardown).

    Args:
        buf: The tensor previously passed to :func:`register_buffer`.

    Raises:
        RuntimeError: If ``phxfs_deregmem`` fails.
    """
    lib = _get_lib()
    base = buf.data_ptr()
    with _state_lock:
        idx = bisect.bisect_left(_reg_bases, base)
        if idx >= len(_reg_bases) or _reg_bases[idx] != base:
            return
        aligned_len, phxfs_dev = _reg_entries[idx]
        _check(
            int(lib.phxfs_deregmem(phxfs_dev, base, aligned_len)),
            "phxfs_deregmem",
        )
        del _reg_bases[idx]
        del _reg_entries[idx]


def register_stream(raw_stream: int) -> None:
    """No-op: phxfs needs no stream registration.

    Every phxfs submission carries the stream handle (unlike cuFile's
    optional cuFileStreamRegister hint). Kept only because the shared
    backend surface (``_gds_async`` re-exports it and ``gds_context``
    calls it uniformly).

    Args:
        raw_stream: Raw CUDA/ROCm stream handle (ignored).
    """
    del raw_stream


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream`: nothing to do for phxfs."""
    del raw_stream


def close_driver() -> None:
    """Release every phxfs registration and close all opened devices.

    Sweeps any buffer registration still left in the table (the normal
    teardown path deregisters each via :func:`deregister_buffer`), then
    closes every opened phxfs device.
    """
    lib = _get_lib()
    with _state_lock:
        for base, (aligned_len, phxfs_dev) in zip(
            _reg_bases, _reg_entries, strict=True
        ):
            try:
                rc = int(lib.phxfs_deregmem(phxfs_dev, base, aligned_len))
                if rc < 0:
                    logger.warning(
                        "close_driver: phxfs_deregmem(0x%x) failed with %d",
                        base,
                        rc,
                    )
            except Exception as e:  # noqa: BLE001 - teardown sweep
                logger.warning("close_driver: phxfs_deregmem raised %s", e)
        _reg_bases.clear()
        _reg_entries.clear()
        for phxfs_dev in sorted(set(_devices.values())):
            rc = int(lib.phxfs_close(phxfs_dev))
            if rc < 0:
                logger.warning(
                    "close_driver: phxfs_close(%d) failed with %d",
                    phxfs_dev,
                    rc,
                )
        _devices.clear()


# --- IO submission --------------------------------------------------------


# --- Submission + AsyncHandle --------------------------------------------


class Submission:
    """One in-flight (stream-ordered) phxfs IO.

    Mirrors :class:`_cufile_async.Submission`: holds the transfer
    parameters and the ``bytes_done`` result storage. The ctypes fields
    are handed to the stream-ordered C API by reference, so the
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
        """Wrap an already-opened fd (``handle`` is the fd itself for phxfs)."""
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
            _get_lib().phxfs_read_stream(
                self._fd,
                buf_base,
                ctypes.byref(sub._size),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "phxfs_read_stream",
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
            _get_lib().phxfs_write_stream(
                self._fd,
                buf_base,
                ctypes.byref(sub._size),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "phxfs_write_stream",
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
