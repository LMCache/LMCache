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


@pytest.fixture(autouse=True)
def _fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the lazy-loaded CDLL and reset process-global driver state."""
    lib = _FakeLib()
    monkeypatch.setattr(ua, "_lib", lib)
    monkeypatch.setattr(ua, "_driver_opened", False)
    return lib


def _fake_gpu_tensor(ptr: int = 0x1000, nbytes: int = 4096) -> SimpleNamespace:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class TestApiSurface:
    def test_exports_required_names(self) -> None:
        required = (
            "AsyncHandle",
            "Submission",
            "close_driver",
            "register_handle",
            "deregister_handle",
            "get_device_capacity",
            "register_buffer",
            "deregister_buffer",
            "register_stream",
            "deregister_stream",
        )
        for name in required:
            assert hasattr(ua, name), f"_ugds_async missing {name!r}"


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
    def test_ensure_open_calls_driver_once(self, _fake_lib: _FakeLib) -> None:
        ua._ensure_driver_open()
        ua._ensure_driver_open()
        assert len(_fake_lib.calls["uGDSDriverOpen"]) == 1
        assert ua._driver_opened is True

    def test_open_failure_does_not_mark_driver_open(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSDriverOpen = lambda: _err(5001)
        with pytest.raises(RuntimeError, match="uGDSDriverOpen"):
            ua._ensure_driver_open()
        assert ua._driver_opened is False

    def test_close_driver_when_open(self, _fake_lib: _FakeLib) -> None:
        ua._ensure_driver_open()
        ua.close_driver()
        assert len(_fake_lib.calls["uGDSDriverClose"]) == 1
        assert ua._driver_opened is False

    def test_close_driver_noop_when_closed(self, _fake_lib: _FakeLib) -> None:
        ua.close_driver()
        assert "uGDSDriverClose" not in _fake_lib.calls

    def test_close_failure_still_resets_state(self, _fake_lib: _FakeLib) -> None:
        ua._ensure_driver_open()
        _fake_lib.uGDSDriverClose = lambda: _err(5007)
        with pytest.raises(RuntimeError, match="uGDSDriverClose"):
            ua.close_driver()
        assert ua._driver_opened is False


class TestHandleRegistration:
    def test_builds_opaque_fd_descriptor(self, _fake_lib: _FakeLib) -> None:
        captured: dict[str, int] = {}

        def _register(handle_ref: Any, descriptor_ref: Any) -> ua._uGDSError_t:
            descriptor = descriptor_ref._obj
            captured["type"] = descriptor.type
            captured["fd"] = descriptor.handle.fd
            handle_ref._obj.value = 0xBEEF
            return _ok()

        _fake_lib.uGDSHandleRegister = _register
        handle = ua.register_handle(42)
        assert handle == 0xBEEF
        assert captured == {
            "type": ua._UGDS_HANDLE_TYPE_OPAQUE_FD,
            "fd": 42,
        }

    def test_register_handle_opens_driver(self, _fake_lib: _FakeLib) -> None:
        ua.register_handle(7)
        assert "uGDSDriverOpen" in _fake_lib.calls

    def test_register_handle_rejects_null_handle(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSHandleRegister = lambda *args: _ok()
        with pytest.raises(RuntimeError, match="null handle"):
            ua.register_handle(7)

    def test_register_handle_propagates_error(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSHandleRegister = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSHandleRegister"):
            ua.register_handle(7)

    def test_deregister_handle_dispatches(self, _fake_lib: _FakeLib) -> None:
        ua.deregister_handle(0x1234)
        (handle,) = _fake_lib.calls["uGDSHandleDeregister"][0]
        assert handle.value == 0x1234

    def test_get_device_capacity_returns_namespace_capacity(
        self, _fake_lib: _FakeLib
    ) -> None:
        assert ua.get_device_capacity(42, 0x1234) == _fake_lib.device_capacity
        handle, capacity_ref = _fake_lib.calls["uGDSGetDeviceCapacity"][0]
        assert handle.value == 0x1234
        assert capacity_ref._obj.value == _fake_lib.device_capacity

    def test_get_device_capacity_propagates_error(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSGetDeviceCapacity = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSGetDeviceCapacity"):
            ua.get_device_capacity(42, 0x1234)

    def test_get_device_capacity_rejects_zero_capacity(
        self, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.device_capacity = 0
        with pytest.raises(RuntimeError, match="zero capacity"):
            ua.get_device_capacity(42, 0x1234)


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self) -> None:
        with pytest.raises(ValueError, match="CUDA or ROCm"):
            ua.register_buffer(SimpleNamespace(is_cuda=False))

    @pytest.mark.parametrize(
        ("hip_version", "expected_flags"),
        [(None, 0), ("6.3", ua._UGDS_REGISTER_DMABUF)],
    )
    def test_register_buffer_passes_pointer_size_and_flags(
        self,
        _fake_lib: _FakeLib,
        monkeypatch: pytest.MonkeyPatch,
        hip_version: str | None,
        expected_flags: int,
    ) -> None:
        monkeypatch.setattr(torch.version, "hip", hip_version)
        ua.register_buffer(_fake_gpu_tensor(ptr=0x2000, nbytes=8192))
        pointer, length, flags = _fake_lib.calls["uGDSBufRegister"][0]
        assert pointer.value == 0x2000
        assert length.value == 8192
        assert flags.value == expected_flags

    def test_register_buffer_propagates_error(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSBufRegister = lambda *args: _err(5036)
        with pytest.raises(RuntimeError, match="uGDSBufRegister"):
            ua.register_buffer(_fake_gpu_tensor())

    def test_deregister_buffer_dispatches(self, _fake_lib: _FakeLib) -> None:
        ua.deregister_buffer(_fake_gpu_tensor(ptr=0x2000))
        (pointer,) = _fake_lib.calls["uGDSBufDeregister"][0]
        assert pointer.value == 0x2000


class TestStreamRegistration:
    def test_register_stream_dispatches(self, _fake_lib: _FakeLib) -> None:
        ua.register_stream(0xABC)
        (stream,) = _fake_lib.calls["uGDSStreamRegister"][0]
        assert stream.value == 0xABC

    def test_register_stream_propagates_error(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSStreamRegister = lambda *args: _err(5008)
        with pytest.raises(RuntimeError, match="uGDSStreamRegister"):
            ua.register_stream(0xABC)

    def test_deregister_stream_dispatches(self, _fake_lib: _FakeLib) -> None:
        ua.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["uGDSStreamDeregister"][0]
        assert stream.value == 0xABC


class TestSubmission:
    def test_stores_async_arguments_and_driver_result(self) -> None:
        submission = ua.Submission(size=4096, file_offset=8192, buf_offset=512)
        assert submission._size.value == 4096
        assert submission._file_offset.value == 8192
        assert submission._buf_offset.value == 512
        assert submission.bytes_done == 0
        submission._bytes_done.value = 4096
        assert submission.bytes_done == 4096


class TestAsyncHandleIO:
    def _handle(self) -> ua.AsyncHandle:
        return ua.AsyncHandle.from_fd(
            fd=5,
            handle=0xFEED,
            path="/dev/ugds_drv0",
            writable=True,
        )

    def test_read_async_marshals_all_arguments(self, _fake_lib: _FakeLib) -> None:
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
        submission = self._handle().read_async(
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

    def test_write_async_dispatches(self, _fake_lib: _FakeLib) -> None:
        submission = self._handle().write_async(
            buf_base=0x3000,
            size=2048,
            file_offset=512,
            buf_offset=128,
            raw_stream=0x9,
        )
        assert "uGDSWriteAsync" in _fake_lib.calls
        assert isinstance(submission, ua.Submission)

    def test_io_error_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.uGDSWriteAsync = lambda *args: _err(5023)
        with pytest.raises(RuntimeError, match="uGDSWriteAsync"):
            self._handle().write_async(
                buf_base=0x3000,
                size=2048,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )


class TestAsyncHandleLifecycle:
    def test_constructor_opens_and_registers_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[tuple[str, int]] = []
        registered: list[int] = []

        def open_device(path: str, flags: int) -> int:
            opened.append((path, flags))
            return 33

        def register_device(fd: int) -> int:
            registered.append(fd)
            return 0xBEEF

        monkeypatch.setattr(
            ua.os,
            "open",
            open_device,
        )
        monkeypatch.setattr(
            ua,
            "register_handle",
            register_device,
        )
        handle = ua.AsyncHandle("/dev/ugds_drv0")
        assert opened == [("/dev/ugds_drv0", ua.os.O_RDWR)]
        assert registered == [33]
        assert handle.fd == 33
        assert handle.path == "/dev/ugds_drv0"
        assert handle.writable is True

    def test_constructor_closes_fd_on_registration_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[int] = []
        monkeypatch.setattr(ua.os, "open", lambda path, flags: 33)
        monkeypatch.setattr(
            ua,
            "register_handle",
            lambda fd: (_ for _ in ()).throw(RuntimeError("register failed")),
        )
        monkeypatch.setattr(ua.os, "close", closed.append)
        with pytest.raises(RuntimeError, match="register failed"):
            ua.AsyncHandle("/dev/ugds_drv0")
        assert closed == [33]

    def test_close_deregisters_handle_and_closes_fd_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deregistered: list[int] = []
        closed: list[int] = []
        monkeypatch.setattr(ua, "deregister_handle", deregistered.append)
        monkeypatch.setattr(ua.os, "close", closed.append)
        handle = ua.AsyncHandle.from_fd(
            fd=5,
            handle=0xFEED,
            path="/dev/ugds_drv0",
            writable=True,
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
