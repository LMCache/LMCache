# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS backend for LMCache GDS L1 (Ascend NPU direct storage)."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import Submission

# hal_kernel_p2p_get_pages rejects a VA that does not start on a 2 MiB boundary.
_NDS_PAGE = 2 * 1024 * 1024


class Backend(FileGDSBackend):
    """NDS-backed file slab (RFC ``NDSBackend``); discovery name is ``Backend``."""

    name = "nds"
    _driver = SharedDriver()

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
        if base % _NDS_PAGE:
            raise ValueError(
                f"register_buffer: address 0x{base:x} is not {_NDS_PAGE}-byte aligned"
            )
        self._ensure_driver_open()
        self._library().register_memory(base, nbytes)

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse :meth:`register_buffer` for ``buf``."""
        view = buf.view(torch.uint8)
        self._library().unregister_memory(view.data_ptr(), view.numel())

    def register_stream(self, raw_stream: int) -> None:
        self._ensure_driver_open()
        self._library().register_stream(raw_stream)

    def deregister_stream(self, raw_stream: int) -> None:
        self._library().deregister_stream(raw_stream)

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

    _backend: Backend

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
        return self._submit(
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
        return self._submit(
            buf_base, size, file_offset, buf_offset, raw_stream, is_write=True
        )

    def _submit(
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
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        self._backend._library().submit_async(
            self._fd,
            buf_base,
            size,
            file_offset,
            buf_offset,
            sub.result,
            raw_stream,
            is_write=is_write,
        )
        return sub
