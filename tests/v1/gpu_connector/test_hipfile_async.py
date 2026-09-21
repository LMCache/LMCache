# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hipFile async backend.

These are pure: ``libhipfile.so`` is never dlopened. A fake lib (see
:class:`_FakeLib`) is substituted at the ``_lib`` seam, so the tests exercise
the Python logic of the ctypes wrapper -- ``hipFileDescr_t`` construction, the
stream-register flag value, error decoding, :class:`Submission` lifetime, and
which symbol each call dispatches to -- without a GPU or the ROCm GPU IO
driver. The ctypes ABI itself (struct field offsets, argtype marshalling) is
covered by the on-hardware roundtrip tests in ``test_gds_context.py``.
"""

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
    """A success ``hipFileError_t`` (err == hipFileSuccess)."""
    return ha._HipFileError(err=ha._HIPFILE_SUCCESS, hip_drv_err=0)


def _err(code: int = 5001) -> ha._HipFileError:
    """A failure ``hipFileError_t``."""
    return ha._HipFileError(err=code, hip_drv_err=0)


class _FakeLib:
    """Stand-in for the ``libhipfile.so`` CDLL.

    Each hipFile symbol records its positional args in :attr:`calls` and returns
    a success ``hipFileError_t`` by default. Tests override individual symbols
    (e.g. to populate an out-param or force an error) by assigning attributes.
    """

    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def __getattr__(self, name: str):
        # Any hipFile* symbol not explicitly overridden records its args and
        # succeeds. ``__getattr__`` only fires for names not set in __dict__.
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


class TestCheck:
    def test_success_is_noop(self, backend: ha.Backend):
        backend.check_error(_ok(), "op")

    def test_nonzero_raises_with_code_and_name(self, backend: ha.Backend, _fake_lib):
        with pytest.raises(RuntimeError) as exc:
            backend.check_error(_err(5002), "hipFileDriverOpen")
        msg = str(exc.value)
        assert "hipFileDriverOpen" in msg
        assert "5002" in msg
        # The op-error string is resolved through the lib.
        assert "hipFileFakeError" in msg


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

    def test_ensure_open_calls_driver_open_once(self, backend: ha.Backend, _fake_lib):
        backend.register_stream(0)
        backend.register_stream(0)
        # Open-state tracking is inherited from the base backend.
        assert "hipFileDriverOpen" in _fake_lib.calls

    def test_close_driver_when_open(self, backend: ha.Backend, _fake_lib):
        backend.register_stream(0)
        backend.close_driver()
        assert "hipFileDriverClose" in _fake_lib.calls

    def test_close_driver_noop_when_closed(self, backend: ha.Backend, _fake_lib):
        backend.close_driver()
        assert "hipFileDriverClose" not in _fake_lib.calls


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

    def test_register_handle_opens_driver(self, backend: ha.Backend, _fake_lib):
        backend.register_handle(7)
        assert "hipFileDriverOpen" in _fake_lib.calls

    def test_deregister_handle_dispatches(self, backend: ha.Backend, _fake_lib):
        backend.deregister_handle(0x1234)
        assert "hipFileHandleDeregister" in _fake_lib.calls


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

    def test_deregister_buffer_dispatches(self, backend: ha.Backend, _fake_lib):
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

    def test_deregister_stream_dispatches(self, backend: ha.Backend, _fake_lib):
        backend.register_stream(0xABC)
        backend.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["hipFileStreamDeregister"]
        assert stream.value == 0xABC


class TestSubmission:
    def test_bytes_done_defaults_zero_then_reflects_driver(self):
        sub = ha.Submission(size=4096, file_offset=0, buf_offset=0)
        assert sub.bytes_done == 0
        sub.result.value = 4096
        assert sub.bytes_done == 4096


class TestAsyncHandleIO:
    def _handle(self, backend: ha.Backend) -> ha.AsyncHandle:
        return ha.AsyncHandle(backend=backend, fd=5, handle=0xFEED, path="/slab")

    def test_read_async_dispatches_and_returns_submission(
        self, backend: ha.Backend, _fake_lib
    ):
        def _read(fh, buf, size_p, foff_p, boff_p, bytes_p, stream):
            # Driver reports the byte count into the caller's storage.
            bytes_p._obj.value = 4096
            _fake_lib.calls["hipFileReadAsync"] = (fh, buf, stream)
            return _ok()

        _fake_lib.hipFileReadAsync = _read
        h = self._handle(backend)
        sub = h.read_async(
            buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
        )
        fh, buf, stream = _fake_lib.calls["hipFileReadAsync"]
        assert fh.value == 0xFEED
        assert buf.value == 0x3000
        assert stream.value == 0x9
        assert sub.bytes_done == 4096

    def test_write_async_dispatches(self, backend: ha.Backend, _fake_lib):
        h = self._handle(backend)
        sub = h.write_async(
            buf_base=0x3000, size=2048, file_offset=512, buf_offset=0, raw_stream=0x9
        )
        assert "hipFileWriteAsync" in _fake_lib.calls
        assert isinstance(sub, ha.Submission)

    def test_io_error_raises(self, backend: ha.Backend, _fake_lib):
        _fake_lib.hipFileWriteAsync = lambda *a: _err(5023)
        h = self._handle(backend)
        with pytest.raises(RuntimeError) as exc:
            h.write_async(
                buf_base=0x3000, size=2048, file_offset=0, buf_offset=0, raw_stream=0x9
            )
        assert "hipFileWriteAsync" in str(exc.value)


class TestStructLayout:
    """Guard the ctypes structs against the hipfile.h C ABI (LP64)."""

    def test_error_struct_size(self):
        assert ctypes.sizeof(ha._HipFileError) == 8

    def test_descr_struct_layout(self):
        assert ctypes.sizeof(ha._HipFileDescr) == 24
        assert ha._HipFileDescr.type.offset == 0
        assert ha._HipFileDescr.handle.offset == 8
        assert ha._HipFileDescr.fs_ops.offset == 16
