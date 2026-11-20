# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS backend for LMCache GDS L1 (Ascend NPU direct storage)."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends._nds_fiemap import FileExtentMap
from lmcache.v1.gpu_connector.gds_backends._nds_libnds import _IoSegment
from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle, Submission

logger = init_logger(__name__)

# Shared GDS page alignment (cuFile / hipFile and GDSContext slab rounding).
# libnds itself only requires 512-byte sector alignment.
_GDS_ALIGNMENT = 4096
# Shared GDS slice size (cuFile's default direct I/O cap). libnds accepts a
# larger registration; GDSContext already splits transfers here.
_MAX_GDS_REGION = 16 * 1024 * 1024


@dataclass(eq=False)
class _RegisteredRegion:
    """One <=16 MiB NPU staging slice for bounds checks and optional mem register."""

    tensor: torch.Tensor
    base_ptr: int
    nbytes: int


def _block_device_path_for_file(path: str) -> str:
    """Return ``/dev/...`` for the block device that hosts a file (``st_dev`` match)."""
    dev = os.stat(path).st_dev
    major, minor = os.major(dev), os.minor(dev)
    try:
        name = Path(f"/sys/dev/block/{major}:{minor}").resolve(strict=True).name
    except OSError:
        raise ValueError(
            f"NDS slab file is not on a local block device: {path} "
            f"(dev {major}:{minor})"
        ) from None

    device_path = Path("/dev") / name
    if not device_path.exists():
        raise ValueError(
            f"NDS block device node not found: {device_path} "
            f"for {path} (dev {major}:{minor})"
        )
    return str(device_path)


class Backend(FileGDSBackend):
    """NDS-backed file slab (RFC ``NDSBackend``); discovery name is ``Backend``."""

    name = "nds"
    _driver = SharedDriver()

    def __init__(self) -> None:
        super().__init__()
        self._registered_streams: set[int] = set()
        self._regions: dict[int, _RegisteredRegion] = {}
        self._slab_size_bytes: int = 0

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

    def open_slab(self, location: str, size: int, direct_io: bool) -> GDSHandle:
        """Preallocate the slab file as a block lease and attach ``/dev/<bdev>``.

        Records ``size`` on the backend, delegates file creation to
        :class:`FileGDSBackend`, and calls :meth:`NDSHandle.attach_block_device`
        so both reads and writes use FIEMAP physical offsets on the backing
        block device. A failure while attaching the block device closes the
        slab file handle before the exception propagates.
        """
        self._slab_size_bytes = size
        handle = super().open_slab(location, size, direct_io)
        try:
            if not isinstance(handle, NDSHandle):
                raise TypeError("NDS open_slab must return NDSHandle")
            handle.attach_block_device(self, size)
        except Exception:
            try:
                handle.close()
            except Exception:
                logger.exception(
                    "Failed to close NDS slab handle after open_slab failure"
                )
            raise
        return handle

    def open_handle(self, fd: int, path: str) -> "NDSHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return NDSHandle(
            self,
            fd,
            handle,
            path,
            slab_size=self._slab_size_bytes,
        )

    def register_handle(self, fd: int) -> Any:
        self._ensure_driver_open()
        return self._library().file_register(fd)

    def deregister_handle(self, handle: Any) -> None:
        self._library().file_deregister(handle)

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register NPU staging memory in <=16 MiB regions and pin each one."""
        if buf.device.type != "npu":
            raise ValueError("register_buffer: tensor must be on NPU")
        if not buf.is_contiguous():
            raise ValueError("register_buffer: tensor must be contiguous")
        self._ensure_driver_open()
        view = buf.view(torch.uint8)
        pinned: list[_RegisteredRegion] = []
        try:
            for start in range(0, view.numel(), _MAX_GDS_REGION):
                region = view[start : min(start + _MAX_GDS_REGION, view.numel())]
                base = region.data_ptr()
                nbytes = region.numel()
                if base in self._regions:
                    raise ValueError(
                        f"NDS buffer base 0x{base:x} is already registered"
                    )
                self._library().register_memory(base, nbytes)
                reg = _RegisteredRegion(
                    tensor=region,
                    base_ptr=base,
                    nbytes=nbytes,
                )
                self._regions[base] = reg
                pinned.append(reg)
        except BaseException:
            for reg in reversed(pinned):
                self._regions.pop(reg.base_ptr, None)
                try:
                    self._library().unregister_memory(reg.base_ptr, reg.nbytes)
                except Exception:
                    logger.exception(
                        "Failed to unregister NDS region 0x%x during rollback",
                        reg.base_ptr,
                    )
            raise

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse :meth:`register_buffer` for each 16 MiB region base in ``buf``."""
        view = buf.view(torch.uint8)
        for start in range(0, view.numel(), _MAX_GDS_REGION):
            region = view[start : min(start + _MAX_GDS_REGION, view.numel())]
            reg = self._regions.pop(region.data_ptr(), None)
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
        """Check GDS alignment, slab span, and registered NPU region before I/O."""
        if size <= 0:
            raise ValueError("I/O size must be positive")
        if file_offset < 0 or buf_offset < 0:
            raise ValueError("I/O offsets must be non-negative")
        if file_offset % alignment or size % alignment or buf_offset % alignment:
            raise ValueError(
                f"NDS I/O requires {alignment}-byte alignment for offset and size"
            )
        if self._slab_size_bytes and file_offset + size > self._slab_size_bytes:
            raise ValueError("file I/O exceeds slab size")
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
    """Owning NDS slab handle with symmetric block-device read/write."""

    def __init__(
        self,
        backend: Backend,
        fd: int,
        handle: Any,
        path: str,
        *,
        slab_size: int,
    ) -> None:
        super().__init__(backend, fd, handle, path)
        self._backend: Backend = backend
        self._slab_size = slab_size
        self._extents: FileExtentMap | None = None
        self._dev_fd = -1
        self._dev_handle: Any = None
        self._dev_path = ""

    def attach_block_device(self, backend: Backend, slab_size: int) -> None:
        """Map the slab's FIEMAP extents once and open ``/dev/<bdev>`` for I/O.

        The slab **file** fd remains open until :meth:`close` to pin its disk
        allocation. Both load and store issue ``PREAD`` / ``PWRITE`` on the
        backing **block device** fd at each extent's physical byte offset
        (translated via :meth:`_io_segments`). On ``register_handle`` failure
        the block fd is closed; any failure propagates to
        :meth:`Backend.open_slab`, which closes the slab file handle.
        """
        self._slab_size = slab_size
        self._extents = FileExtentMap.load(self._fd, slab_size)
        self._dev_path = _block_device_path_for_file(self.path)
        # Open failure propagates; Backend.open_slab closes the slab file.
        dev_fd = os.open(self._dev_path, os.O_RDWR)
        try:
            dev_handle = backend.register_handle(dev_fd)
        except Exception:
            os.close(dev_fd)
            raise
        self._dev_fd = dev_fd
        self._dev_handle = dev_handle

    def close(self) -> None:
        if self._dev_fd >= 0:
            dev_fd = self._dev_fd
            dev_handle = self._dev_handle
            self._dev_fd = -1
            self._dev_handle = None
            try:
                self._backend.deregister_handle(dev_handle)
            finally:
                os.close(dev_fd)
        super().close()

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """GDS ``read_async`` contract; libnds load runs in a stream host callback.

        Logical ``file_offset`` is mapped to physical device offsets on the caller.
        ``bytes_done`` is zero until the callback completes all extent ``PREAD``s.
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

        Logical ``file_offset`` is mapped to physical device offsets on the caller.
        ``bytes_done`` is zero until the callback completes all extent ``PWRITE``s.
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
        if self._dev_fd < 0:
            raise RuntimeError("NDS block device is not open on this handle")
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        segments = self._io_segments(file_offset, size, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        lib = self._backend._library()
        op = lib.write_async if is_write else lib.read_async
        op(
            self._dev_handle,
            buf_base,
            segments,
            sub.result,
            raw_stream,
        )
        return sub

    def _loaded_extents(self) -> FileExtentMap:
        extents = self._extents
        if extents is None:
            raise RuntimeError("NDS slab extents are not loaded")
        return extents

    def _io_segments(
        self, file_offset: int, size: int, buf_offset: int
    ) -> list[_IoSegment]:
        """Translate ``[file_offset, file_offset + size)`` to device DMA segments."""
        extents = self._loaded_extents()
        segments: list[_IoSegment] = []
        end = file_offset + size
        pos = file_offset
        while pos < end:
            ext = extents.extent_at(pos)
            seg_end = min(end, ext.logical + ext.length)
            segments.append(
                _IoSegment(
                    device_offset=ext.physical + (pos - ext.logical),
                    length=seg_end - pos,
                    buf_offset=buf_offset + (pos - file_offset),
                )
            )
            pos = seg_end
        return segments
