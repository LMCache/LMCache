# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the muFile async wrapper (``_mufile_async``).

These are pure: ``libmufile.so`` is never dlopened. A fake lib (see
:class:`_FakeLib`) is substituted at the ``_lib`` seam, so the tests exercise
the Python logic of the ctypes wrapper -- ``MUFileDescr_t`` construction and
ownership, the stream-register flag value, error decoding, :class:`Submission`
lifetime, 4 KiB submit alignment enforcement, and which symbol each call
dispatches to -- without a GPU or the SmartIO GPU IO driver. The ctypes ABI
itself (struct field offsets, argtype marshalling) is covered by the
on-hardware roundtrip tests in ``test_gds_context.py``.
"""

# Standard
from types import SimpleNamespace
import ctypes

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector import _mufile_async as ma


def _ok() -> ma._MUfileError:
    """A success ``MUfileError_t`` (err == MU_FILE_SUCCESS)."""
    return ma._MUfileError(err=ma._MUFILE_SUCCESS)


def _err(code: int = 2) -> ma._MUfileError:
    """A failure ``MUfileError_t``."""
    return ma._MUfileError(err=code)


class _FakeLib:
    """Stand-in for the ``libmufile.so`` CDLL.

    Each muFile symbol records its positional args in :attr:`calls` and returns
    a success ``MUfileError_t`` by default. Tests override individual symbols
    (e.g. to populate an out-param or force an error) by assigning attributes.
    """

    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def __getattr__(self, name: str):
        # Any muFile* symbol not explicitly overridden records its args and
        # succeeds. ``__getattr__`` only fires for names not set in __dict__.
        # Async IO returns a plain ``ssize_t`` submit status per the ABI; every
        # other symbol returns a ``MUfileError_t`` struct.
        def _record(*args):
            self.calls[name] = args
            if name in ("muFileReadAsync", "muFileWriteAsync"):
                return 0
            return _ok()

        return _record


@pytest.fixture(autouse=True)
def _fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the ``_lib`` seam with a fake and reset module driver state."""
    lib = _FakeLib()
    monkeypatch.setattr(ma, "_lib", lambda: lib)
    monkeypatch.setattr(ma, "_driver_opened", False)
    return lib


def _fake_gpu_tensor(ptr: int = 0x1000, nbytes: int = 4096):
    """A duck-typed stand-in for a MUSA ``torch.Tensor`` (no GPU needed)."""
    return SimpleNamespace(
        is_musa=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class TestCheck:
    def test_success_is_noop(self):
        ma._check(_ok(), "op")

    def test_nonzero_raises_with_code(self, _fake_lib):
        with pytest.raises(RuntimeError) as exc:
            ma._check(_err(5), "muFileDriverOpen")
        msg = str(exc.value)
        assert "muFileDriverOpen" in msg
        assert "5" in msg


class TestDriverLifecycle:
    def test_ensure_open_calls_driver_open_once(self, _fake_lib):
        ma._ensure_driver_open()
        ma._ensure_driver_open()
        assert "muFileDriverOpen" in _fake_lib.calls
        assert ma._driver_opened is True

    def test_close_driver_when_open(self, _fake_lib):
        ma._ensure_driver_open()
        ma.close_driver()
        assert "muFileDriverClose" in _fake_lib.calls
        assert ma._driver_opened is False

    def test_close_driver_noop_when_closed(self, _fake_lib):
        ma.close_driver()
        assert "muFileDriverClose" not in _fake_lib.calls


class TestRegisterHandle:
    def test_builds_opaque_fd_descr_and_returns_handle_and_keeps_descr(self, _fake_lib):
        captured = {}

        def _register(fh_ref, descr_ref):
            descr = descr_ref._obj
            captured["type"] = descr.type
            captured["fd"] = descr.handle.fd
            # Populate the out-param like the real driver would.
            fh_ref._obj.value = 0xDEADBEEF
            return _ok()

        _fake_lib.muFileHandleRegister = _register
        handle = ma.register_handle(42)
        assert handle == 0xDEADBEEF
        assert captured["type"] == ma._MUFILE_HANDLE_TYPE_OPAQUE_FD
        assert captured["fd"] == 42
        # The descriptor stays module-owned until deregister_handle.
        assert 0xDEADBEEF in ma._handle_descr_registry

    def test_register_handle_opens_driver(self, _fake_lib):
        ma.register_handle(7)
        assert "muFileDriverOpen" in _fake_lib.calls

    def test_deregister_handle_dispatches_and_releases_descr(self, _fake_lib):
        # Set up a register that actually populates the handle out-param.
        def _register(fh_ref, descr_ref):
            fh_ref._obj.value = 0x1234
            return _ok()

        _fake_lib.muFileHandleRegister = _register
        handle = ma.register_handle(99)
        assert handle == 0x1234
        ma.deregister_handle(handle)
        assert "muFileHandleDeregister" in _fake_lib.calls
        (fh,) = _fake_lib.calls["muFileHandleDeregister"]
        assert fh.value == 0x1234
        assert handle not in ma._handle_descr_registry


class TestBufferRegistration:
    def test_rejects_non_musa_tensor(self):
        cpu = SimpleNamespace(is_musa=False)
        with pytest.raises(ValueError):
            ma.register_buffer(cpu)

    def test_register_buffer_passes_size(self, _fake_lib):
        ma.register_buffer(_fake_gpu_tensor(ptr=0x2000, nbytes=8192))
        base, length, flags = _fake_lib.calls["muFileBufRegister"]
        assert base.value == 0x2000
        assert length.value == 8192
        assert flags.value == 0

    def test_deregister_buffer_dispatches(self, _fake_lib):
        ma.deregister_buffer(_fake_gpu_tensor(ptr=0x2000))
        (base,) = _fake_lib.calls["muFileBufDeregister"]
        assert base.value == 0x2000


class TestStreamRegistration:
    def test_register_stream_uses_fixed_and_aligned_flag(self, _fake_lib):
        ma.register_stream(0xABC)
        stream, flags = _fake_lib.calls["muFileStreamRegister"]
        assert stream.value == 0xABC
        # MUFILE_STREAM_FIXED_AND_ALIGNED: the only flag muFile accepts.
        assert flags == 0x1

    def test_deregister_stream_dispatches(self, _fake_lib):
        ma.register_stream(0xABC)
        ma.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["muFileStreamDeregister"]
        assert stream.value == 0xABC


class TestSubmission:
    def test_bytes_done_defaults_zero_then_reflects_driver(self):
        sub = ma.Submission(size=4096, file_offset=0, buf_offset=0)
        assert sub.bytes_done == 0
        sub._bytes_done.value = 4096
        assert sub.bytes_done == 4096

    def test_completion_is_size_t(self):
        sub = ma.Submission(size=4096, file_offset=0, buf_offset=0)
        assert isinstance(sub._bytes_done, ctypes.c_size_t)


class TestAsyncHandleIO:
    def _handle(self) -> ma.AsyncHandle:
        return ma.AsyncHandle.from_fd(fd=5, handle=0xFEED, path="/slab", writable=True)

    def test_read_async_dispatches_and_returns_submission(self, _fake_lib):
        def _read(fh, buf, size_p, foff_p, boff_p, bytes_p, stream):
            # Driver reports the byte count into the caller's storage.
            bytes_p._obj.value = 4096
            _fake_lib.calls["muFileReadAsync"] = (fh, buf, stream)
            return 0

        _fake_lib.muFileReadAsync = _read
        h = self._handle()
        sub = h.read_async(
            buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
        )
        fh, buf, stream = _fake_lib.calls["muFileReadAsync"]
        assert fh.value == 0xFEED
        assert buf.value == 0x3000
        assert stream.value == 0x9
        assert sub.bytes_done == 4096

    def test_write_async_dispatches(self, _fake_lib):
        h = self._handle()
        sub = h.write_async(
            buf_base=0x3000, size=4096, file_offset=4096, buf_offset=0, raw_stream=0x9
        )
        assert "muFileWriteAsync" in _fake_lib.calls
        assert isinstance(sub, ma.Submission)

    def test_submit_error_raises(self, _fake_lib):
        _fake_lib.muFileWriteAsync = lambda *a: -5  # -MU_FILE_DRIVER_NOT_INITIALIZED
        h = self._handle()
        with pytest.raises(RuntimeError) as exc:
            h.write_async(
                buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
            )
        assert "muFileWriteAsync" in str(exc.value)

    def test_unaligned_submit_raises_before_calling_lib(self, _fake_lib):
        h = self._handle()
        with pytest.raises(ValueError, match="4096"):
            h.write_async(
                buf_base=0x3001,
                size=4096,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )
        assert "muFileWriteAsync" not in _fake_lib.calls

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"buf_base": 0x3001, "size": 4096, "file_offset": 0, "buf_offset": 0},
            {"buf_base": 0x3000, "size": 4097, "file_offset": 0, "buf_offset": 0},
            {"buf_base": 0x3000, "size": 4096, "file_offset": 512, "buf_offset": 0},
            {"buf_base": 0x3000, "size": 4096, "file_offset": 0, "buf_offset": 256},
        ],
    )
    def test_misaligned_any_operand_rejects(self, _fake_lib, kwargs):
        h = self._handle()
        with pytest.raises(ValueError):
            h.write_async(raw_stream=0x9, **kwargs)
        assert "muFileWriteAsync" not in _fake_lib.calls


class TestCloseDriverIdempotency:
    def test_close_driver_twice_is_safe(self, _fake_lib):
        ma._ensure_driver_open()
        ma.close_driver()
        calls_after_first = dict(_fake_lib.calls)
        ma.close_driver()
        assert _fake_lib.calls == calls_after_first


class TestStructLayout:
    """Guard the ctypes structs against the mufile.h C ABI (LP64)."""

    def test_error_struct_size(self):
        assert ctypes.sizeof(ma._MUfileError) == 4

    def test_descr_struct_layout(self):
        assert ctypes.sizeof(ma._MUFileDescr) == 16
        assert ma._MUFileDescr.type.offset == 0
        assert ma._MUFileDescr.handle.offset == 8


class TestErrorMapping:
    def test_op_error_names_are_stable(self):
        # Spot-check the MUfileOpError enum mirror: success and two errors.
        assert ma._MUFILE_SUCCESS == 0
        assert ma._OP_ERROR_NAMES[0] == "MU_FILE_SUCCESS"
        assert ma._OP_ERROR_NAMES[26] == "MU_FILE_HANDLE_NOT_REGISTERED"
        assert ma._OP_ERROR_NAMES[29] == "MU_FILE_INTERNAL_ERROR"


class TestPartialCompletion:
    def test_submit_ok_but_bytes_zero_is_reported_not_faked(self, _fake_lib):
        """Completion reflects the driver-written value, never fabricated."""
        _fake_lib.muFileReadAsync = lambda *a: 0  # submit OK, completion stays 0
        h = ma.AsyncHandle.from_fd(fd=5, handle=0xFEED, path="/slab", writable=True)
        sub = h.read_async(
            buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
        )
        assert sub.bytes_done == 0


class TestBackendDispatch:
    """``mufile`` selection and platform validation in the ``_gds_async`` shim.

    These are host-independent (no CUDA gate): they monkeypatch the platform
    detection the shim consults, never dlopen any library.
    """

    def test_mufile_rejected_off_musa(self, monkeypatch: pytest.MonkeyPatch):
        # First Party
        from lmcache.v1.gpu_connector import _gds_async

        monkeypatch.setattr(_gds_async, "_torch_device_type", lambda: "cuda")
        with pytest.raises(ValueError, match="MUSA"):
            _gds_async.select_backend("mufile")

    def test_mufile_accepted_on_musa(self, monkeypatch: pytest.MonkeyPatch):
        # First Party
        from lmcache.v1.gpu_connector import _gds_async

        monkeypatch.setattr(_gds_async, "_torch_device_type", lambda: "musa")
        assert _gds_async.select_backend("mufile") == "mufile"
        # The re-bound surface must be the mufile wrapper.
        assert _gds_async.AsyncHandle is ma.AsyncHandle

    def test_backend_capability_queries(self):
        # First Party
        from lmcache.v1.gpu_connector import _gds_async

        assert _gds_async.get_max_registered_region_bytes() == 16 * 1024 * 1024
        assert _gds_async.get_io_alignment() == 4096


class TestGetRawStreamHandle:
    """Platform stream-handle extraction (gds_context.get_raw_stream_handle)."""

    def test_prefers_musa_stream(self):
        # First Party
        from lmcache.v1.gpu_connector.gds_context import get_raw_stream_handle

        stream = SimpleNamespace(musa_stream=0xA1, cuda_stream=0xB2)
        assert get_raw_stream_handle(stream) == 0xA1

    def test_cuda_stream_fallback(self):
        # First Party
        from lmcache.v1.gpu_connector.gds_context import get_raw_stream_handle

        stream = SimpleNamespace(cuda_stream=0xB2)
        assert get_raw_stream_handle(stream) == 0xB2

    def test_ptr_fallback(self):
        # First Party
        from lmcache.v1.gpu_connector.gds_context import get_raw_stream_handle

        stream = SimpleNamespace(ptr=0xC3)
        assert get_raw_stream_handle(stream) == 0xC3

    def test_raises_when_no_handle_attribute(self):
        # First Party
        from lmcache.v1.gpu_connector.gds_context import get_raw_stream_handle

        with pytest.raises(RuntimeError, match="stream handle"):
            get_raw_stream_handle(SimpleNamespace())
