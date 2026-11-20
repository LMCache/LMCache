# SPDX-License-Identifier: Apache-2.0
"""NDS backend tests without native libraries or raw block devices."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import ctypes
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector._gds_backends import available_backends, create_backend
from lmcache.v1.gpu_connector.gds_backends import _nds_fiemap as fiemap_mod
from lmcache.v1.gpu_connector.gds_backends import nds as na
from lmcache.v1.gpu_connector.gds_backends._nds_fiemap import (
    FIEMAP_EXTENT_UNWRITTEN,
    FileExtent,
    FileExtentMap,
)
from lmcache.v1.gpu_connector.gds_backends._nds_libnds import _IoSegment
from lmcache.v1.gpu_connector.gds_backends.base import Submission
from lmcache.v1.gpu_connector.gds_context import GDSContext


class _FakeNDSLibrary:
    """Stand-in for :class:`OpenEulerNDSLibrary` recording all calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.deferred_callbacks: list[Callable[[], None]] = []
        self.defer_io = False
        self.fail_on_register_mem_call: int | None = None
        self._register_mem_count = 0

    def driver_open(self) -> None:
        self.calls.append(("driver_open",))

    def driver_close(self) -> None:
        self.calls.append(("driver_close",))

    def file_register(self, fd: int) -> int:
        self.calls.append(("file_register", fd))
        return fd + 100

    def file_deregister(self, handle: int) -> None:
        self.calls.append(("file_deregister", handle))

    def register_memory(self, addr: int, size: int) -> None:
        self._register_mem_count += 1
        if self.fail_on_register_mem_call == self._register_mem_count:
            raise OSError(12, "pin failed")
        self.calls.append(("register_memory", addr, size))

    def unregister_memory(self, addr: int, size: int) -> None:
        self.calls.append(("unregister_memory", addr, size))

    def register_stream(self, raw_stream: int) -> None:
        self.calls.append(("register_stream", raw_stream))

    def deregister_stream(self, raw_stream: int) -> None:
        self.calls.append(("deregister_stream", raw_stream))

    def read_async(
        self,
        dev_fd: int,
        ptr_base: int,
        segments: list[_IoSegment],
        result: ctypes.c_int64,
        raw_stream: int,
    ) -> None:
        self.calls.append(("read_async", dev_fd, ptr_base, list(segments), raw_stream))
        total = sum(seg.length for seg in segments)
        if self.defer_io:
            self.deferred_callbacks.append(lambda: setattr(result, "value", total))
        else:
            result.value = total

    def write_async(
        self,
        dev_fd: int,
        ptr_base: int,
        segments: list[_IoSegment],
        result: ctypes.c_int64,
        raw_stream: int,
    ) -> None:
        self.calls.append(("write_async", dev_fd, ptr_base, list(segments), raw_stream))
        total = sum(seg.length for seg in segments)
        if self.defer_io:
            self.deferred_callbacks.append(lambda: setattr(result, "value", total))
        else:
            result.value = total


class _AmbiguousTensorComparison:
    """Mimics ``torch.Tensor == torch.Tensor`` raising on truth-value testing."""

    def __bool__(self) -> bool:
        raise RuntimeError(
            "Boolean value of Tensor with more than one value is ambiguous"
        )


class _FakeNpuView:
    """Sliceable byte-view stand-in for an NPU ``torch.Tensor``."""

    def __init__(self, ptr: int, nbytes: int, device_index: int = 0) -> None:
        self._ptr = ptr
        self._nbytes = nbytes
        self.device = SimpleNamespace(type="npu", index=device_index)

    def __eq__(self, other: object) -> Any:
        if isinstance(other, _FakeNpuView) and self._nbytes > 1:
            return _AmbiguousTensorComparison()
        return NotImplemented

    def is_contiguous(self) -> bool:
        return True

    def view(self, _dtype: Any) -> "_FakeNpuView":
        return self

    def numel(self) -> int:
        return self._nbytes

    def data_ptr(self) -> int:
        return self._ptr

    def __getitem__(self, item: slice) -> "_FakeNpuView":
        start = 0 if item.start is None else item.start
        stop = self._nbytes if item.stop is None else item.stop
        return _FakeNpuView(self._ptr + start, stop - start, self.device.index)


def _fake_npu_tensor(
    ptr: int = 0x100000, nbytes: int = 8192, device_index: int = 0
) -> Any:
    return _FakeNpuView(ptr=ptr, nbytes=nbytes, device_index=device_index)


@pytest.fixture
def fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeNDSLibrary:
    lib = _FakeNDSLibrary()
    monkeypatch.setattr(na.Backend, "_library", lambda self: lib)
    return lib


@pytest.fixture
def backend(fake_lib: _FakeNDSLibrary) -> Iterator[na.Backend]:
    del fake_lib
    value = na.Backend()
    yield value
    value.close_driver()


def _mock_slab_fs(
    monkeypatch: pytest.MonkeyPatch,
    extents: tuple[FileExtent, ...],
    slab_size: int,
) -> list[int]:
    """Stub slab file creation, sysfs block resolution, and FIEMAP query."""
    closed_fds: list[int] = []
    next_fd = iter((10, 11, 12))

    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)
    monkeypatch.setattr(os, "posix_fallocate", lambda *a: None, raising=False)
    monkeypatch.setattr(os, "open", lambda *_a, **_kw: next(next_fd))
    monkeypatch.setattr(os, "close", closed_fds.append)
    monkeypatch.setattr(
        na,
        "_block_device_path_for_file",
        lambda _path: "/dev/nvme0n1p2",
    )
    monkeypatch.setattr(
        FileExtentMap,
        "load",
        classmethod(lambda cls, _fd, size: cls(extents, slab_size)),
    )
    return closed_fds


class TestDiscoveryAndEnvironment:
    def test_nds_listed_in_available_backends(self) -> None:
        names = available_backends()
        assert "nds" in names
        assert "fiemap" not in names

    def test_create_backend_nds_requires_npu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch, "npu", None, raising=False)
        with pytest.raises(ValueError, match="torch.npu"):
            create_backend("nds")

    def test_nds_is_default_when_npu_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(is_available=lambda: True),
            raising=False,
        )
        assert na.Backend.is_default()

    def test_nds_is_not_default_without_npu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch, "npu", None, raising=False)
        assert not na.Backend.is_default()


class TestDriverLifecycle:
    def test_overlapping_backends_share_driver(
        self, backend: na.Backend, fake_lib: _FakeNDSLibrary
    ) -> None:
        other = na.Backend()
        try:
            backend.register_stream(7)
            other.register_stream(9)
            assert fake_lib.calls.count(("driver_open",)) == 1
            backend.deregister_stream(7)
            backend.close_driver()
            assert ("driver_close",) not in fake_lib.calls
            other.register_buffer(_fake_npu_tensor())
            other.deregister_buffer(_fake_npu_tensor())
            other.deregister_stream(9)
        finally:
            backend.close_driver()
            other.close_driver()
        assert fake_lib.calls.count(("driver_close",)) == 1

    def test_driver_reopens_after_last_owner_closes(
        self, backend: na.Backend, fake_lib: _FakeNDSLibrary
    ) -> None:
        backend.register_stream(7)
        backend.deregister_stream(7)
        backend.close_driver()
        backend.close_driver()
        assert fake_lib.calls.count(("driver_open",)) == 1
        assert fake_lib.calls.count(("driver_close",)) == 1
        backend.register_stream(7)
        assert fake_lib.calls.count(("driver_open",)) == 2
        backend.deregister_stream(7)
        backend.close_driver()

    def test_failed_driver_open_can_retry(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            fake_lib,
            "driver_open",
            lambda: (_ for _ in ()).throw(RuntimeError("open failed")),
        )
        with pytest.raises(RuntimeError, match="open failed"):
            backend.register_stream(7)
        backend.close_driver()
        assert ("driver_close",) not in fake_lib.calls

        monkeypatch.setattr(
            fake_lib,
            "driver_open",
            lambda: fake_lib.calls.append(("driver_open",)),
        )
        backend.register_stream(7)
        assert fake_lib.calls.count(("driver_open",)) == 1
        backend.close_driver()
        assert fake_lib.calls.count(("driver_close",)) == 1

    def test_failed_driver_close_resets_state(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend.register_stream(7)
        backend.deregister_stream(7)

        def fail_close() -> None:
            fake_lib.calls.append(("driver_close",))
            raise RuntimeError("close failed")

        monkeypatch.setattr(fake_lib, "driver_close", fail_close)
        with pytest.raises(RuntimeError, match="close failed"):
            backend.close_driver()
        backend.close_driver()
        assert fake_lib.calls.count(("driver_close",)) == 1

        monkeypatch.setattr(
            fake_lib,
            "driver_close",
            lambda: fake_lib.calls.append(("driver_close",)),
        )
        backend.register_stream(7)
        assert fake_lib.calls.count(("driver_open",)) == 2
        backend.deregister_stream(7)
        backend.close_driver()
        assert fake_lib.calls.count(("driver_close",)) == 2

    def test_failed_context_setup_keeps_peer_driver_alive(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        other = na.Backend()
        monkeypatch.setattr(other, "validate_environment", lambda: None)

        def fail_slab(_location: str, _size: int, _direct_io: bool) -> None:
            handle = other.register_handle(8)
            other.deregister_handle(handle)
            raise RuntimeError("slab setup failed")

        monkeypatch.setattr(other, "open_slab", fail_slab)
        backend.register_stream(7)
        try:
            context = GDSContext(other)
            with pytest.raises(RuntimeError, match="slab setup failed"):
                context.initialize(GdsL1Config("/slab", 4096))
            assert ("driver_close",) not in fake_lib.calls
            backend.register_stream(9)
            assert fake_lib.calls.count(("driver_open",)) == 1
            backend.deregister_stream(7)
            backend.deregister_stream(9)
        finally:
            other.close_driver()
            backend.close_driver()
        assert fake_lib.calls.count(("driver_close",)) == 1


class TestBlockDeviceAndExtentMap:
    def test_block_device_path_resolves_device_number_through_sysfs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        device = os.makedev(259, 2)
        monkeypatch.setattr(
            na.os,
            "stat",
            lambda _path: SimpleNamespace(st_dev=device),
        )

        def resolve(path_self: Path, *, strict: bool = False) -> Path:
            assert str(path_self) == "/sys/dev/block/259:2"
            assert strict
            return Path("/sys/devices/pci0000:00/nvme0n1/nvme0n1p2")

        monkeypatch.setattr(na.Path, "resolve", resolve)
        monkeypatch.setattr(
            na.Path,
            "exists",
            lambda path_self: str(path_self) == "/dev/nvme0n1p2",
        )
        assert na._block_device_path_for_file("/slab/cache") == "/dev/nvme0n1p2"

    def test_file_extent_map_accepts_unwritten_and_looks_up_with_bisect(self) -> None:
        e0 = FileExtent(logical=0, physical=0x10000, length=4096, flags=0)
        e1 = FileExtent(
            logical=4096,
            physical=0x20000,
            length=8192,
            flags=FIEMAP_EXTENT_UNWRITTEN,
        )
        emap = FileExtentMap((e0, e1), 12288)
        assert emap.extent_at(0) == e0
        assert emap.extent_at(4095) == e0
        assert emap.extent_at(4096) == e1
        assert emap.extent_at(12287) == e1
        with pytest.raises(RuntimeError, match="not covered by FIEMAP"):
            emap.extent_at(-1)
        with pytest.raises(RuntimeError, match="not covered by FIEMAP"):
            emap.extent_at(12288)


class TestSlabAndHandleLifecycle:
    def test_open_handle_closes_fd_on_register_failure(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed: list[int] = []
        monkeypatch.setattr(os, "close", closed.append)
        monkeypatch.setattr(
            fake_lib,
            "file_register",
            lambda _fd: (_ for _ in ()).throw(RuntimeError("register")),
        )
        with pytest.raises(RuntimeError, match="register"):
            backend.open_handle(42, "/tmp/slab")
        assert closed == [42]

    def test_open_slab_closes_file_handle_when_block_device_attach_fails(
        self,
        backend: na.Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_fds = _mock_slab_fs(
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=4096, flags=0),),
            4096,
        )

        def open_fail(path: str, _flags: int, *_args: object) -> int:
            if path == "/dev/nvme0n1p2":
                raise OSError(13, "Permission denied")
            return 11

        monkeypatch.setattr(os, "open", open_fail)
        with pytest.raises(OSError, match="Permission denied"):
            backend.open_slab("/slabs", 4096, direct_io=False)
        # Initial fallocate fd (11) and registered slab file fd (11) both closed.
        assert len(closed_fds) == 2

    def test_handle_close_is_idempotent(
        self,
        backend: na.Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_fds = _mock_slab_fs(
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=4096, flags=0),),
            4096,
        )
        handle = backend.open_slab("/slabs", 4096, direct_io=False)
        closed_fds.clear()
        handle.close()
        handle.close()
        assert closed_fds == [12, 11]


class TestBufferAndStreamRegistration:
    def test_register_buffer_rejects_non_npu_tensor(self, backend: na.Backend) -> None:
        cpu = torch.empty(4096, dtype=torch.uint8)
        with pytest.raises(ValueError, match="NPU"):
            backend.register_buffer(cpu)

    def test_register_buffer_non_monotonic_addresses_io_and_deregister(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        b_high = _fake_npu_tensor(ptr=0x200000, nbytes=8192, device_index=1)
        b_low = _fake_npu_tensor(ptr=0x100000, nbytes=4096, device_index=0)
        backend.register_buffer(b_high)
        backend.register_buffer(b_low)
        assert fake_lib.calls.count(("driver_open",)) == 1
        assert ("register_memory", 0x200000, 8192) in fake_lib.calls
        assert ("register_memory", 0x100000, 4096) in fake_lib.calls

        _mock_slab_fs(
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=8192, flags=0),),
            8192,
        )
        handle = backend.open_slab("/slabs", 8192, direct_io=False)
        backend.register_stream(7)

        assert handle.read_async(0x200000, 4096, 0, 4096, 7).bytes_done == 4096
        assert handle.write_async(0x100000, 4096, 0, 0, 7).bytes_done == 4096
        with pytest.raises(RuntimeError, match="exceeds registered region"):
            handle.read_async(0x100000, 4096, 0, 4096, 7)

        fake_lib.calls.clear()
        backend.deregister_buffer(b_low)
        assert fake_lib.calls == [("unregister_memory", 0x100000, 4096)]
        with pytest.raises(RuntimeError, match="not registered"):
            handle.read_async(0x100000, 4096, 0, 0, 7)
        assert handle.read_async(0x200000, 4096, 0, 4096, 7).bytes_done == 4096

        fake_lib.calls.clear()
        backend.deregister_buffer(b_high)
        assert fake_lib.calls == [("unregister_memory", 0x200000, 8192)]
        with pytest.raises(RuntimeError, match="not registered"):
            handle.read_async(0x200000, 4096, 0, 0, 7)

    def test_register_buffer_rejects_duplicate_base_without_corrupting_existing(
        self,
        backend: na.Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_slab_fs(
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=4096, flags=0),),
            4096,
        )
        handle = backend.open_slab("/slabs", 4096, direct_io=False)
        backend.register_stream(7)
        buf = _fake_npu_tensor(ptr=0x100000, nbytes=4096)
        backend.register_buffer(buf)
        with pytest.raises(ValueError, match="already registered"):
            backend.register_buffer(buf)
        assert handle.read_async(0x100000, 4096, 0, 0, 7).bytes_done == 4096

    def test_register_buffer_unregisters_pinned_regions_on_failure(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        region_size = 16 * 1024 * 1024
        fake_lib.fail_on_register_mem_call = 2
        buf = _fake_npu_tensor(ptr=0x100000, nbytes=region_size * 2)
        with pytest.raises(OSError, match="pin failed"):
            backend.register_buffer(buf)
        assert ("unregister_memory", 0x100000, region_size) in fake_lib.calls
        # Rolled-back region must not remain registered for subsequent I/O.
        _mock_slab_fs(
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=4096, flags=0),),
            4096,
        )
        handle = backend.open_slab("/slabs", 4096, direct_io=False)
        backend.register_stream(7)
        with pytest.raises(RuntimeError, match="not registered"):
            handle.read_async(0x100000, 4096, 0, 0, 7)

    def test_stream_registration_is_idempotent_and_rolls_back_on_error(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend.register_stream(7)
        backend.register_stream(7)
        assert fake_lib.calls.count(("register_stream", 7)) == 1
        backend.deregister_stream(7)
        backend.deregister_stream(7)
        assert fake_lib.calls.count(("deregister_stream", 7)) == 1

        monkeypatch.setattr(
            fake_lib,
            "register_stream",
            lambda _s: (_ for _ in ()).throw(OSError("ctx")),
        )
        with pytest.raises(OSError, match="ctx"):
            backend.register_stream(9)


class TestNDSHandleIO:
    def _open_test_slab(
        self,
        backend: na.Backend,
        monkeypatch: pytest.MonkeyPatch,
        extents: tuple[FileExtent, ...],
        slab_size: int,
    ) -> na.NDSHandle:
        _mock_slab_fs(monkeypatch, extents, slab_size)
        slab = backend.open_slab("/slabs", slab_size, direct_io=False)
        assert isinstance(slab, na.NDSHandle)
        return slab

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_requires_attached_block_device(
        self, backend: na.Backend, operation: str
    ) -> None:
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor())
        handle = backend.open_handle(5, "/slab")
        submit = handle.read_async if operation == "read" else handle.write_async
        with pytest.raises(RuntimeError, match="block device is not open"):
            submit(0x100000, 4096, 0, 0, 7)

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_splits_multi_extent_segments_symmetrically(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
    ) -> None:
        handle = self._open_test_slab(
            backend,
            monkeypatch,
            (
                FileExtent(logical=0, physical=0xA000, length=4096, flags=0),
                FileExtent(logical=4096, physical=0xB000, length=4096, flags=0),
            ),
            8192,
        )
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=16384))
        submit = handle.read_async if operation == "read" else handle.write_async
        sub = submit(
            buf_base=0x100000,
            size=8192,
            file_offset=0,
            buf_offset=4096,
            raw_stream=7,
        )
        assert isinstance(sub, Submission)
        assert sub.bytes_done == 8192
        assert fake_lib.calls[-1] == (
            f"{operation}_async",
            112,
            0x100000,
            [
                _IoSegment(0xA000, 4096, 4096),
                _IoSegment(0xB000, 4096, 8192),
            ],
            7,
        )

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_segments_use_buffer_offset_not_file_offset(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
    ) -> None:
        handle = self._open_test_slab(
            backend,
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=16384, flags=0),),
            16384,
        )
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=8192))
        submit = handle.read_async if operation == "read" else handle.write_async
        submit(0x100000, 4096, 8192, 0, 7)
        assert fake_lib.calls[-1][3] == [_IoSegment(0xA000 + 8192, 4096, 0)]

    def test_async_io_defers_completion_until_callback(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_lib.defer_io = True
        handle = self._open_test_slab(
            backend,
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=4096, flags=0),),
            4096,
        )
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=4096))
        sub = handle.read_async(0x100000, 4096, 0, 0, 7)
        assert sub.bytes_done == 0
        fake_lib.deferred_callbacks[0]()
        assert sub.bytes_done == 4096

    def test_async_io_rejects_unaligned_and_out_of_bounds_operands(
        self,
        backend: na.Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handle = self._open_test_slab(
            backend,
            monkeypatch,
            (FileExtent(logical=0, physical=0xA000, length=8192, flags=0),),
            8192,
        )
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=8192))
        with pytest.raises(ValueError, match="4096"):
            handle.write_async(0x100000, 4096, 512, 0, 7)
        with pytest.raises(ValueError, match="slab size"):
            handle.read_async(0x100000, 4096, 8192, 0, 7)
        with pytest.raises(RuntimeError, match="not registered"):
            handle.read_async(0x999000, 4096, 0, 0, 7)


class TestStructLayout:
    """Guard the FIEMAP ctypes structures against ``<linux/fiemap.h>``."""

    def test_fiemap_extent_layout(self) -> None:
        assert ctypes.sizeof(fiemap_mod._FiemapExtent) == 56
        assert fiemap_mod._FiemapExtent.fe_logical.offset == 0
        assert fiemap_mod._FiemapExtent.fe_physical.offset == 8
        assert fiemap_mod._FiemapExtent.fe_length.offset == 16
        assert fiemap_mod._FiemapExtent.fe_flags.offset == 40

    def test_fiemap_header_layout(self) -> None:
        assert fiemap_mod._Fiemap.fm_extents.offset == 32
