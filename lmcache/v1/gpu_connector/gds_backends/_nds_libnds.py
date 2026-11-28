# SPDX-License-Identifier: Apache-2.0
"""ctypes binding and stream-ordered I/O adapter for openEuler ``libnds.so``.

This module wraps the C ABI of ``libnds.so`` (kernel ``nds_p2p`` on local NVMe):
it defines the ``ctypes`` struct layouts and function signatures, pins NPU
buffers via ``nds_register_mem``, manages one ``io_ctx`` per NPU stream, and
executes stream-ordered ``read_async`` / ``write_async`` on the slab file by
enqueuing ``nds_io_submit`` + ``nds_io_getevents`` onto the NPU stream via
``aclrtLaunchHostFunc``.

It is separated from ``nds.py`` to isolate the C driver bindings from the
GDS backend. ``nds.py`` owns the slab file and ``Submission`` tracking and
passes the file descriptor and logical offset through to this module.

Source: openEuler kernel ``tools/lib/nds``
([openEuler-26.09](https://gitcode.com/openeuler/kernel/tree/openEuler-26.09)).
User API: ``nds_api.h``; device node ``/dev/p2p_device`` (``drivers/nds``).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from functools import lru_cache
from typing import Any
import ctypes
import errno
import itertools
import os

# Third Party
import torch

# First Party
from lmcache.v1.common.logging import init_logger

logger = init_logger(__name__)

_ACL_HOST_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# void* cookie is 64-bit (c_void_p(1 << 64) is NULL); fold the counter and skip
# 0 and ids still queued. Python's count itself does not wrap.
_CALLBACK_ID_SPACE = 1 << 64
_callback_ids = itertools.count(1)
_pending_callbacks: dict[int, Callable[[], None]] = {}


def _allocate_callback_id() -> int:
    while True:
        callback_id = next(_callback_ids) % _CALLBACK_ID_SPACE
        if callback_id and callback_id not in _pending_callbacks:
            return callback_id


def _exception_to_submission_result(exc: BaseException) -> int:
    """Map an exception to a negative ``Submission.result`` for deferred I/O."""
    if isinstance(exc, OSError) and exc.errno is not None:
        return -int(exc.errno)
    return -1


@_ACL_HOST_FUNC
def _host_trampoline(callback_id: int) -> None:
    _pending_callbacks.pop(callback_id)()


def _npu_stream_handle(npu: Any) -> int | None:
    handle = getattr(npu.current_stream(), "npu_stream", None)
    return handle if isinstance(handle, int) else None


@lru_cache(maxsize=1)
def _aclrt_launch_host_func() -> Callable[..., int]:
    fn = ctypes.CDLL("libascendcl.so").aclrtLaunchHostFunc
    fn.argtypes = [ctypes.c_void_p, _ACL_HOST_FUNC, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _launch_npu_stream_host_callback(
    raw_stream: int, callback: Callable[[], None]
) -> None:
    """Enqueue ``callback`` on the current NPU stream after prior stream work."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("NDS stream-ordered I/O requires torch.npu")
    # Reading npu_stream calls NPUStream::stream(), which drains the torch
    # task queue. Launch on this pointer, not one saved at register_stream.
    # The caller already set this thread's device; it has one context.
    handle = _npu_stream_handle(npu)
    if handle != raw_stream:
        current = handle if isinstance(handle, int) else 0
        raise RuntimeError(
            "NDS I/O must run on the current NPU stream "
            f"(registered 0x{raw_stream:x}, current 0x{current:x})"
        )
    callback_id = _allocate_callback_id()
    _pending_callbacks[callback_id] = callback
    err = int(
        _aclrt_launch_host_func()(
            ctypes.c_void_p(handle),
            _host_trampoline,
            ctypes.c_void_p(callback_id),
        )
    )
    if err != 0:
        _pending_callbacks.pop(callback_id, None)
        raise RuntimeError(f"aclrtLaunchHostFunc failed with error {err}")


_DEFAULT_NDS_LIBRARY_PATH = "/usr/local/lib/libnds.so"
# (major << 8) | minor. The ctypes layouts in this file match this UAPI only.
_NDS_UAPI_VERSION = 0x0001

NDS_IO_OP_PREAD = 0
NDS_IO_OP_PWRITE = 1
NDS_IO_F_REGISTERED_MEM = 1 << 0
# One callback blocks in getevents, so a stream never has more than one
# submission outstanding. The depth still leaves room for a future batch.
_IO_CTX_DEPTH = 64


class _NdsInitParam(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint64 * 4),
    ]


class _NdsFsDesc(ctypes.Structure):
    _fields_ = [
        ("fs_fd", ctypes.POINTER(ctypes.c_int32)),
        ("fs_fd_cnt", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NdsMemRange(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _NdsMemDesc(ctypes.Structure):
    _fields_ = [
        ("ranges", ctypes.POINTER(_NdsMemRange)),
        ("nr", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _NdsIoCtxParam(ctypes.Structure):
    _fields_ = [
        ("max_io_cnt", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


class _NdsIoObj(ctypes.Structure):
    class _Data(ctypes.Union):
        _fields_ = [("_data", ctypes.c_uint64), ("fd", ctypes.c_int32)]

    _fields_ = [("reserved", ctypes.c_uint64), ("u", _Data)]


class _NdsIoVec(ctypes.Structure):
    _fields_ = [
        ("buf_addr", ctypes.c_uint64),
        ("buf_len", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NdsIoCb(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint32),
        ("rw_flags", ctypes.c_uint32),
        ("obj", _NdsIoObj),
        ("offset", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("iov", ctypes.POINTER(_NdsIoVec)),
        ("iov_cnt", ctypes.c_uint32),
        ("host_pid", ctypes.c_int32),
        ("reserved", ctypes.c_uint64 * 2),
    ]


class _NdsIoEvent(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _NdsIoCtx(ctypes.Structure):
    pass


# Every symbol returns ``c_int``, which is the ctypes default.
_LIBNDS_SYMBOLS: tuple[tuple[str, list[Any]], ...] = (
    ("nds_init", [ctypes.POINTER(_NdsInitParam)]),
    ("nds_exit", []),
    ("nds_register_fs", [ctypes.POINTER(_NdsFsDesc)]),
    ("nds_unregister_fs", [ctypes.POINTER(_NdsFsDesc)]),
    ("nds_register_mem", [ctypes.POINTER(_NdsMemDesc)]),
    ("nds_unregister_mem", [ctypes.POINTER(_NdsMemDesc)]),
    (
        "nds_io_new_ctx",
        [
            ctypes.POINTER(_NdsIoCtxParam),
            ctypes.POINTER(ctypes.POINTER(_NdsIoCtx)),
        ],
    ),
    ("nds_io_destroy_ctx", [ctypes.POINTER(_NdsIoCtx)]),
    (
        "nds_io_submit",
        [ctypes.POINTER(_NdsIoCtx), ctypes.c_int, ctypes.POINTER(_NdsIoCb)],
    ),
    (
        "nds_io_getevents",
        [
            ctypes.POINTER(_NdsIoCtx),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_NdsIoEvent),
            ctypes.c_void_p,
        ],
    ),
)


def _posix_errno(ret: int, operation: str) -> None:
    if ret < 0:
        raise OSError(-ret, operation)


def _mem_desc(addr: int, size: int) -> tuple[_NdsMemRange, _NdsMemDesc]:
    """One-range descriptor. The range must stay alive for the ioctl."""
    rng = _NdsMemRange(addr=addr, size=size)
    return rng, _NdsMemDesc(ranges=ctypes.pointer(rng), nr=1, flags=0)


class OpenEulerNDSLibrary:
    """Process-local owner of openEuler ``libnds.so`` (P2P NVMe ↔ NPU HBM)."""

    def __init__(self, library_path: str | None = None) -> None:
        if library_path is None:
            library_path = os.environ.get("NDS_LIBRARY_PATH")
        if library_path is not None:
            if not os.path.isfile(library_path):
                raise FileNotFoundError(
                    f"openEuler NDS library not found at {library_path}"
                )
        elif os.path.isfile(_DEFAULT_NDS_LIBRARY_PATH):
            library_path = _DEFAULT_NDS_LIBRARY_PATH
        else:
            # Let the dynamic loader search ldconfig and LD_LIBRARY_PATH.
            library_path = "libnds.so"
        self._lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_LOCAL)
        self._declare_signatures(self._lib)
        self._driver_open = False
        # Filled by register_stream and read by that stream's host callback.
        # GDSContext serializes registration, so this map has no lock.
        self._stream_ctxs: dict[int, ctypes._Pointer[_NdsIoCtx]] = {}

    def _declare_signatures(self, lib: ctypes.CDLL) -> None:
        for name, argtypes in _LIBNDS_SYMBOLS:
            try:
                fn = getattr(lib, name)
            except AttributeError:
                raise AttributeError(
                    f"openEuler libnds missing required symbol {name!r}"
                ) from None
            fn.argtypes = argtypes

    def driver_open(self) -> None:
        param = _NdsInitParam()
        ret = self._lib.nds_init(ctypes.byref(param))
        _posix_errno(ret, "nds_init")
        # nds_init writes the driver UAPI. An older library leaves this zero.
        if param.version != _NDS_UAPI_VERSION:
            # The .so already opened the device. Exit so a later open can retry.
            self._lib.nds_exit()
            got = f"v{param.version >> 8}.{param.version & 0xFF}"
            need = f"v{_NDS_UAPI_VERSION >> 8}.{_NDS_UAPI_VERSION & 0xFF}"
            raise RuntimeError(f"openEuler libnds UAPI {got}, need {need}")
        self._driver_open = True

    def driver_close(self) -> None:
        if not self._driver_open:
            self._stream_ctxs.clear()
            return
        contexts = list(self._stream_ctxs.values())
        self._stream_ctxs.clear()
        for ctx in contexts:
            self._destroy_io_ctx(ctx)
        ret = self._lib.nds_exit()
        _posix_errno(ret, "nds_exit")
        self._driver_open = False

    def register_stream(self, raw_stream: int) -> None:
        """Create the ``io_ctx`` used by every callback on ``raw_stream``."""
        if raw_stream in self._stream_ctxs:
            return
        if not self._driver_open:
            raise RuntimeError("NDS driver is not open")
        self._stream_ctxs[raw_stream] = self._new_io_ctx()

    def deregister_stream(self, raw_stream: int) -> None:
        """Destroy the ``io_ctx`` for ``raw_stream``. A missing stream is a no-op."""
        ctx = self._stream_ctxs.pop(raw_stream, None)
        if ctx is not None:
            self._destroy_io_ctx(ctx)

    def _new_io_ctx(self) -> ctypes._Pointer[_NdsIoCtx]:
        ctx = ctypes.POINTER(_NdsIoCtx)()
        ctx_param = _NdsIoCtxParam(max_io_cnt=_IO_CTX_DEPTH, flags=0, reserved=0)
        ret = self._lib.nds_io_new_ctx(ctypes.byref(ctx_param), ctypes.byref(ctx))
        _posix_errno(ret, "nds_io_new_ctx")
        return ctx

    def _destroy_io_ctx(self, ctx: ctypes._Pointer[_NdsIoCtx]) -> None:
        ret = self._lib.nds_io_destroy_ctx(ctx)
        if ret < 0:
            logger.warning("nds_io_destroy_ctx failed: %d", ret)

    def _stream_ctx(self, raw_stream: int) -> ctypes._Pointer[_NdsIoCtx]:
        try:
            return self._stream_ctxs[raw_stream]
        except KeyError:
            raise RuntimeError(f"NDS stream {raw_stream} has no io_ctx") from None

    def file_register(self, fd: int) -> int:
        fd_arr = (ctypes.c_int32 * 1)(fd)
        desc = _NdsFsDesc(fd_arr, 1, 0)
        ret = self._lib.nds_register_fs(ctypes.byref(desc))
        _posix_errno(ret, "nds_register_fs")
        return fd

    def file_deregister(self, native_handle: int) -> None:
        fd_arr = (ctypes.c_int32 * 1)(native_handle)
        desc = _NdsFsDesc(fd_arr, 1, 0)
        ret = self._lib.nds_unregister_fs(ctypes.byref(desc))
        if ret == -2:  # ENOENT if the topology is already gone
            return
        _posix_errno(ret, "nds_unregister_fs")

    def register_memory(self, addr: int, size: int) -> None:
        """Pin one VA range through the per-fd memory table."""
        _rng, desc = _mem_desc(addr, size)
        ret = self._lib.nds_register_mem(ctypes.byref(desc))
        _posix_errno(ret, "nds_register_mem")

    def unregister_memory(self, addr: int, size: int) -> None:
        _rng, desc = _mem_desc(addr, size)
        ret = self._lib.nds_unregister_mem(ctypes.byref(desc))
        if ret == -2:  # ENOENT if already gone
            return
        _posix_errno(ret, "nds_unregister_mem")

    def submit_async(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        result: ctypes.c_int64,
        raw_stream: int,
        *,
        is_write: bool,
    ) -> None:
        """Enqueue one stream-ordered read or write of the slab file.

        An unregistered stream fails here, before the host callback is queued.
        """
        self._stream_ctx(raw_stream)

        def host_work() -> None:
            try:
                done = self._io(
                    self._stream_ctx(raw_stream),
                    file_fd,
                    ptr_base,
                    size,
                    file_offset,
                    buf_offset,
                    is_write,
                )
                result.value = done
            except BaseException as exc:
                logger.error(
                    "NDS async I/O failed (fd=%d, offset=%d, size=%d): %s",
                    file_fd,
                    file_offset,
                    size,
                    exc,
                )
                result.value = _exception_to_submission_result(exc)

        _launch_npu_stream_host_callback(raw_stream, host_work)

    def _io(
        self,
        ctx: ctypes._Pointer[_NdsIoCtx],
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        is_write: bool,
    ) -> int:
        """Submit one op and block in ``nds_io_getevents`` on ``ctx``.

        ``ctx`` belongs to one stream, and ACL runs that stream's callbacks
        one at a time, so this call cannot observe another submission.
        """
        # libnds packs buf_len in uint32 and the driver rejects above 2 GiB.
        # A larger Python int would wrap in the ctypes field.
        if size <= 0 or size > (2 << 30):
            raise ValueError("I/O size must be positive and at most 2 GiB")
        iov = _NdsIoVec(buf_addr=ptr_base + buf_offset, buf_len=size)
        cb = _NdsIoCb(
            opcode=NDS_IO_OP_PWRITE if is_write else NDS_IO_OP_PREAD,
            rw_flags=NDS_IO_F_REGISTERED_MEM,
            offset=file_offset,
            iov=ctypes.pointer(iov),
            iov_cnt=1,
        )
        cb.obj.u.fd = file_fd
        ret = self._lib.nds_io_submit(ctx, 1, ctypes.byref(cb))
        if ret != 1:
            _posix_errno(ret if ret < 0 else -1, "nds_io_submit")
        events = (_NdsIoEvent * 1)()
        # NULL timeout waits until this context has a completion.
        # EINTR abandons a still-running DMA in the queue; retry that wait.
        ret = self._lib.nds_io_getevents(ctx, 1, 1, events, None)
        while ret == -errno.EINTR:
            ret = self._lib.nds_io_getevents(ctx, 1, 1, events, None)
        if ret != 1:
            _posix_errno(ret if ret < 0 else -5, "nds_io_getevents")
        if events[0].res != 0:
            raise OSError(-int(events[0].res), "nds_io completion")
        return size


@lru_cache(maxsize=1)
def get_openeuler_nds_library() -> OpenEulerNDSLibrary:
    return OpenEulerNDSLibrary()
