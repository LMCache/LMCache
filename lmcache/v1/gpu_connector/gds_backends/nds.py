# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS backend for LMCache GDS L1 (Ascend NPU direct storage)."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import Submission

# Shared GDS page alignment (cuFile / hipFile and GDSContext slab rounding).
# libnds itself only requires 512-byte sector alignment.
_GDS_ALIGNMENT = 4096


@dataclass(eq=False)
class _RegisteredRegion:
    """One NPU staging region for bounds checks."""

    tensor: torch.Tensor
    base_ptr: int
    nbytes: int


class Backend(FileGDSBackend):
    """NDS-backed file slab (RFC ``NDSBackend``); discovery name is ``Backend``."""

    name = "nds"
    _driver = SharedDriver()

    def __init__(self) -> None:
        super().__init__()
        self._registered_streams: set[int] = set()
        self._regions: dict[int, _RegisteredRegion] = {}

    @classmethod
    def is_default(cls) -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and npu.is_available()

    def validate_environment(self) -> None:
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise ValueError("nds backend requires torch.npu")
        if not npu.is_available():
            raise ValueError("nds backend requires an available NPU")

    def open_handle(self, fd: int, path: str) -> "NDSHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return NDSHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> Any:
        self._ensure_driver_open()
        return self._library().file_register(fd)

    def deregister_handle(self, handle: Any) -> None:
        self._library().file_deregister(handle)

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Pin one contiguous NPU staging region."""
        if buf.device.type != "npu":
            raise ValueError("register_buffer: tensor must be on NPU")
        if not buf.is_contiguous():
            raise ValueError("register_buffer: tensor must be contiguous")
        view = buf.view(torch.uint8)
        nbytes = view.numel()
        if nbytes <= 0:
            raise ValueError("register_buffer: tensor is empty")
        base = view.data_ptr()
        if base in self._regions:
            raise ValueError(f"NDS buffer base 0x{base:x} is already registered")
        self._ensure_driver_open()
        self._library().register_memory(base, nbytes)
        self._regions[base] = _RegisteredRegion(
            tensor=view, base_ptr=base, nbytes=nbytes
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse :meth:`register_buffer` for ``buf``."""
        base = buf.view(torch.uint8).data_ptr()
        reg = self._regions.pop(base, None)
        if reg is not None:
            self._library().unregister_memory(reg.base_ptr, reg.nbytes)

    def register_stream(self, raw_stream: int) -> None:
        if raw_stream in self._registered_streams:
            return
        self._ensure_driver_open()
        self._registered_streams.add(raw_stream)
        try:
            self._library().register_stream(raw_stream)
        except BaseException:
            self._registered_streams.discard(raw_stream)
            raise

    def deregister_stream(self, raw_stream: int) -> None:
        if raw_stream not in self._registered_streams:
            return
        try:
            self._library().deregister_stream(raw_stream)
        finally:
            self._registered_streams.discard(raw_stream)

    def _validate_stream(self, raw_stream: int) -> None:
        if raw_stream not in self._registered_streams:
            raise RuntimeError(f"NDS stream {raw_stream} is not registered")

    def _validate_io_bounds(
        self,
        file_offset: int,
        size: int,
        buf_base: int,
        buf_offset: int,
        alignment: int = _GDS_ALIGNMENT,
    ) -> None:
        """Check GDS alignment and the registered NPU region before I/O."""
        if size <= 0:
            raise ValueError("I/O size must be positive")
        if file_offset < 0 or buf_offset < 0:
            raise ValueError("I/O offsets must be non-negative")
        if file_offset % alignment or size % alignment or buf_offset % alignment:
            raise ValueError(
                f"NDS I/O requires {alignment}-byte alignment for offset and size"
            )
        region = self._regions.get(buf_base)
        if region is None:
            raise RuntimeError(f"NDS buffer base 0x{buf_base:x} is not registered")
        if buf_offset + size > region.nbytes:
            raise RuntimeError("buffer I/O exceeds registered region")

    def _library(self):
        # First Party
        from lmcache.v1.gpu_connector.gds_backends._nds_libnds import (
            get_openeuler_nds_library,
        )

        return get_openeuler_nds_library()

    def _open_driver(self) -> None:
        self._library().driver_open()

    def _close_driver(self) -> None:
        self._library().driver_close()


class NDSHandle(FDGDSHandle):
    """Owning NDS slab-file handle. Reads and writes use the file descriptor."""

    def __init__(self, backend: Backend, fd: int, handle: Any, path: str) -> None:
        super().__init__(backend, fd, handle, path)
        self._backend: Backend = backend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """GDS ``read_async`` contract; libnds load runs in a stream host callback.

        ``bytes_done`` is zero until the callback completes the ``PREAD``.
        """
        return self._transfer_async(
            buf_base, size, file_offset, buf_offset, raw_stream, is_write=False
        )

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """GDS ``write_async`` contract; libnds store runs in a stream host callback.

        ``bytes_done`` is zero until the callback completes the ``PWRITE``.
        """
        return self._transfer_async(
            buf_base, size, file_offset, buf_offset, raw_stream, is_write=True
        )

    def _transfer_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
        *,
        is_write: bool,
    ) -> Submission:
        if self._fd < 0:
            raise RuntimeError("NDS slab file is not open on this handle")
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        lib = self._backend._library()
        op = lib.write_async if is_write else lib.read_async
        op(
            self._handle,
            buf_base,
            size,
            file_offset,
            buf_offset,
            sub.result,
            raw_stream,
        )
        return sub
