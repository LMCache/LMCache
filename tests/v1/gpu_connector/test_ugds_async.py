# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the uGDS async wrapper (``_ugds_async``).

These tests are pure: ``libugds.so`` is never loaded and no raw device is
opened. A fake library exercises the Python ctypes wrapper, including driver
lifecycle, handle descriptors, buffer and stream registration, async argument
marshalling, error propagation, resource cleanup, and C structure layout.

The real ctypes ABI and end-to-end DMA path are covered by the opt-in
``test_ugds_context_roundtrip`` hardware test in ``test_gds_context.py``.
"""

# Standard
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
import ctypes
import os
import stat

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import _ugds_async as ua


def _ok() -> ua._uGDSError_t:
    """Return a successful ``uGDSError_t``."""
    return ua._uGDSError_t(err=0, cu_err=0)


def _err(code: int = 5001, cuda_code: int = 0) -> ua._uGDSError_t:
    """Return a failed ``uGDSError_t``."""
    return ua._uGDSError_t(err=code, cu_err=cuda_code)


class _FakeLib:
    """Stand-in for ``libugds.so`` that records all symbol calls."""

    uGDSDriverOpen: Callable[..., ua._uGDSError_t]
    uGDSDriverClose: Callable[..., ua._uGDSError_t]
    uGDSHandleRegister: Callable[..., ua._uGDSError_t]
    uGDSGetDeviceCapacity: Callable[..., ua._uGDSError_t]
    uGDSBufRegister: Callable[..., ua._uGDSError_t]
    uGDSStreamRegister: Callable[..., ua._uGDSError_t]
    uGDSReadAsync: Callable[..., ua._uGDSError_t]
    uGDSWriteAsync: Callable[..., ua._uGDSError_t]

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}
        self.device_capacity = 2 << 40

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any) -> ua._uGDSError_t:
            self.calls.setdefault(name, []).append(args)
            if name == "uGDSHandleRegister":
                args[0]._obj.value = 0xDEADBEEF
            elif name == "uGDSGetDeviceCapacity":
                args[1]._obj.value = self.device_capacity
            return _ok()

        return _record


@pytest.fixture
def backend() -> ua.UgdsBackend:
    return ua.UgdsBackend()


@pytest.fixture(autouse=True)
def _fake_lib(backend: ua.UgdsBackend, monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the lazy-loaded CDLL and reset process-global driver state."""
    lib = _FakeLib()
    monkeypatch.setattr(backend, "library", lambda: lib)
    return lib


def _fake_gpu_tensor(ptr: int = 0x1000, nbytes: int = 4096) -> SimpleNamespace:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class TestCheck:
    def test_success_is_noop(self) -> None:
        ua._check(_ok(), "op")

    def test_error_includes_operation_and_codes(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            ua._check(_err(5036, cuda_code=700), "uGDSBufRegister")
        message = str(exc_info.value)
        assert "uGDSBufRegister" in message
        assert "5036" in message
        assert "700" in message


class TestDriverLifecycle:
    def test_ensure_open_calls_driver_once(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0)
        backend.register_stream(0)
        assert len(_fake_lib.calls["uGDSDriverOpen"]) == 1

    def test_open_failure_does_not_mark_driver_open(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSDriverOpen = lambda: _err(5001)
        with pytest.raises(RuntimeError, match="uGDSDriverOpen"):
            backend.register_stream(0)

    def test_close_driver_when_open(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0)
        backend.close_driver()
        assert len(_fake_lib.calls["uGDSDriverClose"]) == 1

    def test_close_driver_noop_when_closed(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.close_driver()
        assert "uGDSDriverClose" not in _fake_lib.calls

    def test_close_failure_still_resets_state(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0)
        _fake_lib.uGDSDriverClose = lambda: _err(5007)
        with pytest.raises(RuntimeError, match="uGDSDriverClose"):
            backend.close_driver()


class TestHandleRegistration:
    def test_builds_opaque_fd_descriptor(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        captured: dict[str, int] = {}

        def _register(handle_ref: Any, descriptor_ref: Any) -> ua._uGDSError_t:
            descriptor = descriptor_ref._obj
            captured["type"] = descriptor.type
            captured["fd"] = descriptor.handle.fd
            handle_ref._obj.value = 0xBEEF
            return _ok()

        _fake_lib.uGDSHandleRegister = _register
        handle = backend.register_handle(42)
        assert handle == 0xBEEF
        assert captured == {
            "type": ua._UGDS_HANDLE_TYPE_OPAQUE_FD,
            "fd": 42,
        }

    def test_register_handle_opens_driver(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_handle(7)
        assert "uGDSDriverOpen" in _fake_lib.calls

    def test_register_handle_rejects_null_handle(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSHandleRegister = lambda *args: _ok()
        with pytest.raises(RuntimeError, match="null handle"):
            backend.register_handle(7)

    def test_register_handle_propagates_error(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSHandleRegister = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSHandleRegister"):
            backend.register_handle(7)

    def test_deregister_handle_dispatches(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.deregister_handle(0x1234)
        (handle,) = _fake_lib.calls["uGDSHandleDeregister"][0]
        assert handle.value == 0x1234

    def test_get_device_capacity_returns_namespace_capacity(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        assert backend.get_device_capacity(42, 0x1234) == _fake_lib.device_capacity
        handle, capacity_ref = _fake_lib.calls["uGDSGetDeviceCapacity"][0]
        assert handle.value == 0x1234
        assert capacity_ref._obj.value == _fake_lib.device_capacity

    def test_get_device_capacity_propagates_error(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSGetDeviceCapacity = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSGetDeviceCapacity"):
            backend.get_device_capacity(42, 0x1234)

    def test_get_device_capacity_rejects_zero_capacity(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.device_capacity = 0
        with pytest.raises(RuntimeError, match="zero capacity"):
            backend.get_device_capacity(42, 0x1234)


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self, backend: ua.UgdsBackend) -> None:
        with pytest.raises(ValueError, match="CUDA or ROCm"):
            backend.register_buffer(SimpleNamespace(is_cuda=False))

    @pytest.mark.parametrize(
        ("hip_version", "expected_flags"),
        [(None, 0), ("6.3", ua._UGDS_REGISTER_DMABUF)],
    )
    def test_register_buffer_passes_pointer_size_and_flags(
        self,
        backend: ua.UgdsBackend,
        _fake_lib: _FakeLib,
        monkeypatch: pytest.MonkeyPatch,
        hip_version: str | None,
        expected_flags: int,
    ) -> None:
        monkeypatch.setattr(torch.version, "hip", hip_version)
        backend.register_buffer(_fake_gpu_tensor(ptr=0x2000, nbytes=8192))
        pointer, length, flags = _fake_lib.calls["uGDSBufRegister"][0]
        assert pointer.value == 0x2000
        assert length.value == 8192
        assert flags.value == expected_flags

    def test_register_buffer_propagates_error(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSBufRegister = lambda *args: _err(5036)
        with pytest.raises(RuntimeError, match="uGDSBufRegister"):
            backend.register_buffer(_fake_gpu_tensor())

    def test_deregister_buffer_dispatches(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.deregister_buffer(_fake_gpu_tensor(ptr=0x2000))
        (pointer,) = _fake_lib.calls["uGDSBufDeregister"][0]
        assert pointer.value == 0x2000


class TestStreamRegistration:
    def test_register_stream_dispatches(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0xABC)
        (stream,) = _fake_lib.calls["uGDSStreamRegister"][0]
        assert stream.value == 0xABC

    def test_register_stream_propagates_error(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSStreamRegister = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSStreamRegister"):
            backend.register_stream(0xABC)

    def test_deregister_stream_dispatches(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        backend.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["uGDSStreamDeregister"][0]
        assert stream.value == 0xABC


class TestSubmission:
    def test_stores_async_arguments_and_driver_result(self) -> None:
        submission = ua.Submission(size=4096, file_offset=8192, buf_offset=512)
        assert submission.size.value == 4096
        assert submission.file_offset.value == 8192
        assert submission.buf_offset.value == 512
        assert submission.bytes_done == 0
        submission.result.value = 4096
        assert submission.bytes_done == 4096


class TestAsyncHandleIO:
    def _handle(self, backend: ua.UgdsBackend) -> ua.AsyncHandle:
        return ua.AsyncHandle(
            backend=backend,
            fd=5,
            handle=0xFEED,
            path="/dev/ugds_drv0",
        )

    def test_read_async_marshals_all_arguments(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        captured: dict[str, int] = {}

        def _read(
            handle: Any,
            buffer: Any,
            size_ref: Any,
            file_offset_ref: Any,
            buffer_offset_ref: Any,
            bytes_ref: Any,
            stream: Any,
        ) -> ua._uGDSError_t:
            captured.update(
                handle=handle.value,
                buffer=buffer.value,
                size=size_ref._obj.value,
                file_offset=file_offset_ref._obj.value,
                buffer_offset=buffer_offset_ref._obj.value,
                stream=stream.value,
            )
            bytes_ref._obj.value = size_ref._obj.value
            return _ok()

        _fake_lib.uGDSReadAsync = _read
        submission = self._handle(backend).read_async(
            buf_base=0x3000,
            size=4096,
            file_offset=8192,
            buf_offset=512,
            raw_stream=0x9,
        )
        assert captured == {
            "handle": 0xFEED,
            "buffer": 0x3000,
            "size": 4096,
            "file_offset": 8192,
            "buffer_offset": 512,
            "stream": 0x9,
        }
        assert submission.bytes_done == 4096

    def test_write_async_dispatches(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        submission = self._handle(backend).write_async(
            buf_base=0x3000,
            size=2048,
            file_offset=512,
            buf_offset=128,
            raw_stream=0x9,
        )
        assert "uGDSWriteAsync" in _fake_lib.calls
        assert isinstance(submission, ua.Submission)

    def test_io_error_raises(
        self, backend: ua.UgdsBackend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.uGDSWriteAsync = lambda *args: _err(5023)
        with pytest.raises(RuntimeError, match="uGDSWriteAsync"):
            self._handle(backend).write_async(
                buf_base=0x3000,
                size=2048,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )


class TestAsyncHandleLifecycle:
    def test_close_deregisters_handle_and_closes_fd_once(
        self, backend: ua.UgdsBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deregistered: list[int] = []
        closed: list[int] = []
        monkeypatch.setattr(backend, "deregister_handle", deregistered.append)
        monkeypatch.setattr(ua.os, "close", closed.append)
        handle = ua.AsyncHandle(
            backend=backend,
            fd=5,
            handle=0xFEED,
            path="/dev/ugds_drv0",
        )
        handle.close()
        handle.close()
        assert deregistered == [0xFEED]
        assert closed == [5]
        assert handle.fd == -1


class TestStructLayout:
    """Guard the ctypes structures against ``ugds.h`` on LP64 platforms."""

    def test_error_struct_layout(self) -> None:
        assert ctypes.sizeof(ua._uGDSError_t) == 8
        assert ua._uGDSError_t.err.offset == 0
        assert ua._uGDSError_t.cu_err.offset == 4

    def test_descriptor_struct_layout(self) -> None:
        assert ctypes.sizeof(ua._uGDSDescr_t) == 16
        assert ua._uGDSDescr_t.type.offset == 0
        assert ua._uGDSDescr_t.handle.offset == 8


def _mock_device(monkeypatch: pytest.MonkeyPatch, mode: int, subsystem: str) -> None:
    real_stat = os.stat
    realpath = os.path.realpath
    monkeypatch.setattr(
        os,
        "stat",
        lambda path, *a, **kw: (
            SimpleNamespace(st_mode=mode, st_rdev=os.makedev(511, 0))
            if path == "/dev/ugds_drv7"
            else real_stat(path, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda path, *a, **kw: (
            f"/sys/class/{subsystem}"
            if path.startswith("/sys/dev/char/")
            else realpath(path, *a, **kw)
        ),
    )


@pytest.mark.parametrize(
    ("mode", "subsystem", "error"),
    [
        (stat.S_IFREG, "ugds_drv", "character device"),
        (stat.S_IFBLK, "ugds_drv", "character device"),
        (stat.S_IFCHR, "nvidia", "ugds_drv"),
    ],
)
def test_slab_rejects_invalid_device_before_open(
    backend: ua.UgdsBackend,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    subsystem: str,
    error: str,
) -> None:
    _mock_device(monkeypatch, mode, subsystem)
    monkeypatch.setattr(os, "open", lambda *a: pytest.fail("must not open device"))
    with pytest.raises(ValueError, match=error):
        backend.open_slab("/dev/ugds_drv7", 4096, True)


def test_raw_slab_never_creates_or_truncates_files(
    backend: ua.UgdsBackend,
    monkeypatch: pytest.MonkeyPatch,
    _fake_lib: _FakeLib,
) -> None:
    _mock_device(monkeypatch, stat.S_IFCHR, "ugds_drv")
    opened: list[object] = []
    closed: list[int] = []

    def open_device(*args: object) -> int:
        opened.append(args)
        return 33

    monkeypatch.setattr(os, "open", open_device)
    monkeypatch.setattr(os, "close", closed.append)
    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: pytest.fail("mkdir"))
    monkeypatch.setattr(
        os, "posix_fallocate", lambda *a: pytest.fail("allocate"), raising=False
    )
    slab = backend.open_slab("/dev/ugds_drv7", 4096, True)
    assert opened == [("/dev/ugds_drv7", os.O_RDWR)]
    assert slab.path == "/dev/ugds_drv7"
    assert slab.capacity() == _fake_lib.device_capacity
    slab.close()
    assert closed == [33]
    assert len(_fake_lib.calls["uGDSHandleDeregister"]) == 1


@pytest.mark.parametrize("capacity", [0, 2048])
def test_capacity_failure_releases_handle_and_descriptor(
    backend: ua.UgdsBackend,
    monkeypatch: pytest.MonkeyPatch,
    _fake_lib: _FakeLib,
    capacity: int,
) -> None:
    _mock_device(monkeypatch, stat.S_IFCHR, "ugds_drv")
    _fake_lib.device_capacity = capacity
    closed: list[int] = []
    monkeypatch.setattr(os, "open", lambda *a: 33)
    monkeypatch.setattr(os, "close", closed.append)
    with pytest.raises((RuntimeError, ValueError), match="capacity"):
        backend.open_slab("/dev/ugds_drv7", 4096, False)
    assert len(_fake_lib.calls["uGDSHandleDeregister"]) == 1
    assert closed == [33]
