# SPDX-License-Identifier: Apache-2.0
"""hipFile tests with a fake native library."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import ctypes

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.gds_backends import hipfile as ha


def _ok() -> ha._HipFileError:
    return ha._HipFileError(err=ha._HIPFILE_SUCCESS, hip_drv_err=0)


def _err(code: int = 5001) -> ha._HipFileError:
    return ha._HipFileError(err=code, hip_drv_err=0)


class _FakeLib:
    """Record native calls and return success unless overridden."""

    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def __getattr__(self, name: str):
        def _record(*args):
            self.calls[name] = args
            if name == "hipFileGetOpErrorString":
                return b"hipFileFakeError"
            return _ok()

        return _record


@pytest.fixture
def backend() -> ha.Backend:
    return ha.Backend()


@pytest.fixture(autouse=True)
def _fake_lib(backend: ha.Backend, monkeypatch) -> _FakeLib:
    """Replace the backend's native library with a fresh fake."""
    lib = _FakeLib()
    monkeypatch.setattr(backend, "library", lambda: lib)
    return lib


def _fake_gpu_tensor(ptr: int = 0x1000, nbytes: int = 4096):
    """A duck-typed stand-in for a CUDA ``torch.Tensor`` (no GPU needed)."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class TestDriverLifecycle:
    def test_concurrent_open_and_close_call_driver_once(
        self,
        backend: ha.Backend,
        _fake_lib: _FakeLib,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        opened, closed = Mock(return_value=_ok()), Mock(return_value=_ok())
        monkeypatch.setattr(_fake_lib, "hipFileDriverOpen", opened)
        monkeypatch.setattr(_fake_lib, "hipFileDriverClose", closed)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(backend.register_stream, range(16)))
            opened.assert_called_once()
            list(pool.map(lambda _: backend.close_driver(), range(16)))
            closed.assert_called_once()


class TestRegisterHandle:
    def test_builds_opaque_fd_descr_and_returns_handle(
        self, backend: ha.Backend, _fake_lib
    ):
        captured = {}

        def _register(fh_ref, descr_ref):
            descr = descr_ref._obj
            captured["type"] = descr.type
            captured["fd"] = descr.handle.fd
            # Populate the out-param like the real driver would.
            fh_ref._obj.value = 0xDEADBEEF
            return _ok()

        _fake_lib.hipFileHandleRegister = _register
        handle = backend.register_handle(42)
        assert handle == 0xDEADBEEF
        assert captured["type"] == ha._HIPFILE_HANDLE_TYPE_OPAQUE_FD
        assert captured["fd"] == 42
        backend.deregister_handle(handle)
        (native_handle,) = _fake_lib.calls["hipFileHandleDeregister"]
        assert native_handle.value == handle


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self, backend: ha.Backend):
        cpu = SimpleNamespace(is_cuda=False)
        with pytest.raises(ValueError):
            backend.register_buffer(cpu)

    def test_register_buffer_passes_size(self, backend: ha.Backend, _fake_lib):
        backend.register_buffer(_fake_gpu_tensor(ptr=0x2000, nbytes=8192))
        base, length, flags = _fake_lib.calls["hipFileBufRegister"]
        assert base.value == 0x2000
        assert length.value == 8192
        backend.deregister_buffer(_fake_gpu_tensor(ptr=0x2000))
        (base,) = _fake_lib.calls["hipFileBufDeregister"]
        assert base.value == 0x2000


class TestStreamRegistration:
    def test_register_stream_uses_fixed_flags(self, backend: ha.Backend, _fake_lib):
        backend.register_stream(0xABC)
        stream, flags = _fake_lib.calls["hipFileStreamRegister"]
        assert stream.value == 0xABC
        # FIXED_BUF_OFFSET | FIXED_FILE_OFFSET | FIXED_FILE_SIZE.
        assert flags == 0x7
        backend.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["hipFileStreamDeregister"]
        assert stream.value == 0xABC


class TestAsyncHandleIO:
    def _handle(self, backend: ha.Backend) -> ha.AsyncHandle:
        return ha.AsyncHandle(backend=backend, fd=5, handle=0xFEED, path="/slab")

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_marshals_arguments(
        self, backend: ha.Backend, _fake_lib, monkeypatch, operation: str
    ):
        native = Mock(return_value=_ok())
        monkeypatch.setattr(_fake_lib, f"hipFile{operation.title()}Async", native)
        h = self._handle(backend)
        submit = h.read_async if operation == "read" else h.write_async
        sub = submit(
            buf_base=0x3000, size=4096, file_offset=8192, buf_offset=512, raw_stream=0x9
        )
        fh, buf, size, offset, buf_offset, result, stream = native.call_args.args
        assert fh.value == 0xFEED
        assert buf.value == 0x3000
        assert stream.value == 0x9
        assert size._obj is sub.size and size._obj.value == 4096
        assert offset._obj is sub.file_offset and offset._obj.value == 8192
        assert buf_offset._obj is sub.buf_offset and buf_offset._obj.value == 512
        assert result._obj is sub.result
        result._obj.value = 4096
        assert sub.bytes_done == 4096

    def test_io_error_raises(self, backend: ha.Backend, _fake_lib):
        _fake_lib.hipFileWriteAsync = lambda *a: _err(5023)
        h = self._handle(backend)
        with pytest.raises(RuntimeError) as exc:
            h.write_async(
                buf_base=0x3000, size=2048, file_offset=0, buf_offset=0, raw_stream=0x9
            )
        assert "hipFileWriteAsync" in str(exc.value)
        assert "5023" in str(exc.value)
        assert "hipFileFakeError" in str(exc.value)


class TestStructLayout:
    """Guard the ctypes structs against the hipfile.h C ABI (LP64)."""

    def test_error_struct_size(self):
        assert ctypes.sizeof(ha._HipFileError) == 8

    def test_descr_struct_layout(self):
        assert ctypes.sizeof(ha._HipFileDescr) == 24
        assert ha._HipFileDescr.type.offset == 0
        assert ha._HipFileDescr.handle.offset == 8
        assert ha._HipFileDescr.fs_ops.offset == 16
