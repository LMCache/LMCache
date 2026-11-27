# SPDX-License-Identifier: Apache-2.0
"""NDS backend tests without native libraries."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
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
from lmcache.v1.gpu_connector.gds_backends import nds as na
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
        return fd

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
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        result: ctypes.c_int64,
        raw_stream: int,
    ) -> None:
        self.calls.append(
            ("read_async", file_fd, ptr_base, size, file_offset, buf_offset, raw_stream)
        )
        if self.defer_io:
            self.deferred_callbacks.append(lambda: setattr(result, "value", size))
        else:
            result.value = size

    def write_async(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        result: ctypes.c_int64,
        raw_stream: int,
    ) -> None:
        self.calls.append(
            (
                "write_async",
                file_fd,
                ptr_base,
                size,
                file_offset,
                buf_offset,
                raw_stream,
            )
        )
        if self.defer_io:
            self.deferred_callbacks.append(lambda: setattr(result, "value", size))
        else:
            result.value = size


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


def _mock_slab_fs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub slab file creation. Returns the fds ``os.close`` was given."""
    closed_fds: list[int] = []
    next_fd = iter((10, 11, 12))
    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)
    monkeypatch.setattr(os, "posix_fallocate", lambda *a: None, raising=False)
    monkeypatch.setattr(os, "open", lambda *_a, **_kw: next(next_fd))
    monkeypatch.setattr(os, "close", closed_fds.append)
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

    def test_open_slab_closes_file_when_register_fails(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_fds = _mock_slab_fs(monkeypatch)
        monkeypatch.setattr(
            fake_lib,
            "file_register",
            lambda _fd: (_ for _ in ()).throw(RuntimeError("register")),
        )
        with pytest.raises(RuntimeError, match="register"):
            backend.open_slab("/slabs", 4096, direct_io=False)
        # fallocate fd, then the reopened slab fd after registration fails.
        assert closed_fds == [10, 11]

    def test_handle_close_deregisters_before_closing_the_fd(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_fds = _mock_slab_fs(monkeypatch)
        handle = backend.open_slab("/slabs", 4096, direct_io=False)
        closed_fds.clear()
        handle.close()
        handle.close()
        assert ("file_deregister", 11) in fake_lib.calls
        assert fake_lib.calls.count(("file_deregister", 11)) == 1
        assert closed_fds == [11]
        deregister_at = fake_lib.calls.index(("file_deregister", 11))
        assert all(
            call[0] != "file_deregister" or i == deregister_at
            for i, call in enumerate(fake_lib.calls)
        )


class TestBufferAndStreamRegistration:
    def test_register_buffer_rejects_non_npu_tensor(self, backend: na.Backend) -> None:
        cpu = torch.empty(4096, dtype=torch.uint8)
        with pytest.raises(ValueError, match="NPU"):
            backend.register_buffer(cpu)

    def test_register_buffer_rejects_empty(self, backend: na.Backend) -> None:
        with pytest.raises(ValueError, match="empty"):
            backend.register_buffer(_fake_npu_tensor(nbytes=0))

    def test_register_buffer_pins_the_whole_region_once(
        self, backend: na.Backend, fake_lib: _FakeNDSLibrary
    ) -> None:
        region = 16 * 1024 * 1024
        buf = _fake_npu_tensor(ptr=0x100000, nbytes=region * 2)
        backend.register_buffer(buf)
        # 0x100000 is not 2 MiB aligned, so the pin starts at the previous boundary.
        assert ("register_memory", 0, 0x100000 + region * 2) in fake_lib.calls
        assert fake_lib.calls.count(("register_memory", 0, 0x100000 + region * 2)) == 1

    def test_misaligned_gds_slices_share_one_pin(
        self, backend: na.Backend, fake_lib: _FakeNDSLibrary
    ) -> None:
        """16 MiB slices of one misaligned buffer must not overlap in the driver."""
        slice_bytes = 16 * 1024 * 1024
        base = 0x108000
        first = _fake_npu_tensor(ptr=base, nbytes=slice_bytes)
        second = _fake_npu_tensor(ptr=base + slice_bytes, nbytes=slice_bytes)
        backend.register_buffer(first)
        backend.register_buffer(second)
        assert fake_lib.calls == [
            ("driver_open",),
            ("register_memory", 0, base + slice_bytes),
            ("unregister_memory", 0, base + slice_bytes),
            ("register_memory", 0, base + 2 * slice_bytes),
        ]
        fake_lib.calls.clear()
        backend.deregister_buffer(second)
        assert fake_lib.calls == [
            ("unregister_memory", 0, base + 2 * slice_bytes),
            ("register_memory", 0, base + slice_bytes),
        ]
        fake_lib.calls.clear()
        backend.deregister_buffer(first)
        assert fake_lib.calls == [("unregister_memory", 0, base + slice_bytes)]

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
        assert ("register_memory", 0, 0x101000) in fake_lib.calls

        _mock_slab_fs(monkeypatch)
        handle = backend.open_slab("/slabs", 8192, direct_io=False)
        backend.register_stream(7)

        assert handle.read_async(0x200000, 4096, 0, 4096, 7).bytes_done == 4096
        assert handle.write_async(0x100000, 4096, 0, 0, 7).bytes_done == 4096
        with pytest.raises(RuntimeError, match="exceeds registered region"):
            handle.read_async(0x100000, 4096, 0, 4096, 7)

        fake_lib.calls.clear()
        backend.deregister_buffer(b_low)
        assert fake_lib.calls == [("unregister_memory", 0, 0x101000)]
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
        _mock_slab_fs(monkeypatch)
        handle = backend.open_slab("/slabs", 4096, direct_io=False)
        backend.register_stream(7)
        buf = _fake_npu_tensor(ptr=0x100000, nbytes=4096)
        backend.register_buffer(buf)
        with pytest.raises(ValueError, match="already registered"):
            backend.register_buffer(buf)
        assert handle.read_async(0x100000, 4096, 0, 0, 7).bytes_done == 4096

    def test_failed_registration_leaves_the_buffer_unregistered(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_lib.fail_on_register_mem_call = 1
        buf = _fake_npu_tensor(ptr=0x100000, nbytes=4096)
        with pytest.raises(OSError, match="pin failed"):
            backend.register_buffer(buf)
        assert not any(call[0] == "unregister_memory" for call in fake_lib.calls)
        _mock_slab_fs(monkeypatch)
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
        self, backend: na.Backend, monkeypatch: pytest.MonkeyPatch, slab_size: int
    ) -> na.NDSHandle:
        _mock_slab_fs(monkeypatch)
        slab = backend.open_slab("/slabs", slab_size, direct_io=False)
        assert isinstance(slab, na.NDSHandle)
        return slab

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_requires_an_open_slab_file(
        self,
        backend: na.Backend,
        operation: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(os, "close", lambda _fd: None)
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor())
        handle = backend.open_handle(5, "/slab")
        handle.close()
        submit = handle.read_async if operation == "read" else handle.write_async
        with pytest.raises(RuntimeError, match="not open"):
            submit(0x100000, 4096, 0, 0, 7)

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_submits_the_slab_fd_and_logical_offset(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
    ) -> None:
        handle = self._open_test_slab(backend, monkeypatch, 16384)
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=16384))
        submit = handle.read_async if operation == "read" else handle.write_async
        sub = submit(
            buf_base=0x100000,
            size=8192,
            file_offset=4096,
            buf_offset=4096,
            raw_stream=7,
        )
        assert isinstance(sub, Submission)
        assert sub.bytes_done == 8192
        assert fake_lib.calls[-1] == (
            f"{operation}_async",
            11,
            0x100000,
            8192,
            4096,
            4096,
            7,
        )

    def test_async_io_defers_completion_until_callback(
        self,
        backend: na.Backend,
        fake_lib: _FakeNDSLibrary,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_lib.defer_io = True
        handle = self._open_test_slab(backend, monkeypatch, 4096)
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
        handle = self._open_test_slab(backend, monkeypatch, 8192)
        backend.register_stream(7)
        backend.register_buffer(_fake_npu_tensor(ptr=0x100000, nbytes=8192))
        with pytest.raises(ValueError, match="4096"):
            handle.write_async(0x100000, 4096, 512, 0, 7)
        with pytest.raises(ValueError, match="slab size"):
            handle.read_async(0x100000, 4096, 8192, 0, 7)
        with pytest.raises(RuntimeError, match="not registered"):
            handle.read_async(0x999000, 4096, 0, 0, 7)
