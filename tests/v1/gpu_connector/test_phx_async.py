# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Phoenix frozen-ABI wrapper (``_phx_async``).

These tests are pure: neither ``libphxfile.so`` nor ``libphoenix.so`` is
loaded and no phxfs device is opened. A fake library exercises the Python
ctypes wrapper over the frozen ``phxFile*`` surface: library loading
(including the fail-fast taken when the shim predates the async symbols),
the device-free buffer registration (raw length passed through — device
probing, alignment and the registration table live inside the shim, not
here), address-only deregistration, the symbol-backed stream-registration
surface, handle boxing, stream-ordered IO argument marshalling (hipFile
parameter order: file_offset before buf_offset; byref ctypes storage +
raw stream handle; late-binding error propagation via ``bytes_done``),
handle lifecycle, and ``close_driver``.

The real phxfs ABI and end-to-end DMA path are exercised on Phoenix
hardware via the GDS L1 tier (``--gds-l1-backend phx``).
"""

# Standard
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector import _phx_async as pa


class _FakeLib:
    """Stand-in for ``libphxfile.so`` that records all symbol calls."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}
        # When False, the async symbols raise AttributeError like a CDLL
        # without them (loading must then fail fast against a stale shim).
        self.has_async_api = True
        # Injectable return codes for each symbol.
        self.register_rc = 0
        self.deregister_rc = 0
        self.handle_register_rc = 0
        self.stream_register_rc = 0
        self.stream_deregister_rc = 0
        self.driver_close_rc = 0
        self.read_async_rc = 0
        self.write_async_rc = 0
        # Optional override for the value written into bytes_done by the
        # fake async call (None -> succeed with nbytes).
        self.read_async_bd: Optional[int] = None
        self.write_async_bd: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        if not self.has_async_api and name in (
            "phxFileReadAsync",
            "phxFileWriteAsync",
        ):
            raise AttributeError(name)

        def _record(*args: Any) -> Any:
            self.calls.setdefault(name, []).append(args)
            if name == "phxFileHandleRegister":
                # fh_p, fd_c -> box the fd into the out-param (identity).
                args[0]._obj.value = args[1].value
                return self.handle_register_rc
            if name == "phxFileBufRegister":
                return self.register_rc
            if name == "phxFileBufDeregister":
                return self.deregister_rc
            if name == "phxFileStreamRegister":
                return self.stream_register_rc
            if name == "phxFileStreamDeregister":
                return self.stream_deregister_rc
            if name == "phxFileDriverClose":
                return self.driver_close_rc
            if name == "phxFileReadAsync":
                # (fh, buf, nb_p, fo_p, bo_p, bd_p, stream)
                bd = self.read_async_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.read_async_rc
            if name == "phxFileWriteAsync":
                bd = self.write_async_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.write_async_rc
            return 0

        return _record


@pytest.fixture(autouse=True)
def _fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the lazy-loaded CDLL with the fake frozen-ABI library."""
    lib = _FakeLib()
    monkeypatch.setattr(pa, "_lib", lib)
    return lib


def _gpu_tensor(
    ptr: int = 0x100000, nbytes: int = 64 * 1024, cuda_index: int = 0
) -> Any:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
        device=SimpleNamespace(index=cuda_index),
    )


class TestApiSurface:
    def test_exports_required_names(self) -> None:
        required = (
            "AsyncHandle",
            "Submission",
            "close_driver",
            "register_handle",
            "deregister_handle",
            "register_buffer",
            "deregister_buffer",
            "register_stream",
            "deregister_stream",
        )
        for name in required:
            assert hasattr(pa, name), f"_phx_async missing {name!r}"


class TestLibLoading:
    def test_get_lib_raises_when_library_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(pa.ctypes.util, "find_library", lambda _: None)

        def _raise(path: str) -> Any:
            raise OSError(f"cannot open {path}")

        monkeypatch.setattr(pa.ctypes, "CDLL", _raise)
        with pytest.raises(OSError, match="libphxfile"):
            pa._get_lib()

    def test_get_lib_prefers_find_library_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(
            pa.ctypes.util, "find_library", lambda _: "/opt/lib/libphxfile.so.1"
        )

        class _Lib:
            def __getattr__(self, name: str) -> Any:
                return lambda *args: 0

        def _load(path: str) -> Any:
            loaded.append(path)
            return _Lib()

        monkeypatch.setattr(pa.ctypes, "CDLL", _load)
        pa._get_lib()
        assert loaded == ["/opt/lib/libphxfile.so.1"]

    def test_missing_async_symbols_fail_fast(
        self, _fake_lib: _FakeLib, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_lib.has_async_api = False
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(pa.ctypes.util, "find_library", lambda _: "libphxfile.so")

        def _load(path: str) -> Any:
            return _fake_lib

        monkeypatch.setattr(pa.ctypes, "CDLL", _load)
        with pytest.raises(RuntimeError, match="stream-ordered API"):
            pa._get_lib()


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self) -> None:
        with pytest.raises(ValueError, match="CUDA or ROCm"):
            pa.register_buffer(SimpleNamespace(is_cuda=False))  # type: ignore[arg-type]

    def test_rejects_empty_tensor(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pa.register_buffer(_gpu_tensor(nbytes=0))

    def test_passes_addr_and_raw_length(self, _fake_lib: _FakeLib) -> None:
        pa.register_buffer(_gpu_tensor(ptr=0x200000, nbytes=4096))
        addr, length = _fake_lib.calls["phxFileBufRegister"][0]
        assert addr.value == 0x200000
        # The raw length flows through unaligned: page-size rounding,
        # device probing and the registration table live inside the
        # shim, not in this wrapper.
        assert length.value == 4096

    def test_register_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.register_rc = -19  # ENODEV (no phxfs device present)
        with pytest.raises(RuntimeError, match="phxFileBufRegister"):
            pa.register_buffer(_gpu_tensor())


class TestBufferDeregistration:
    def test_passes_address_only(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_buffer(_gpu_tensor(ptr=0x200000))
        (addr,) = _fake_lib.calls["phxFileBufDeregister"][0]
        # Only the base address: the aligned length playback is the
        # shim's job.
        assert addr.value == 0x200000

    def test_deregister_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.deregister_rc = -22  # EINVAL
        with pytest.raises(RuntimeError, match="phxFileBufDeregister"):
            pa.deregister_buffer(_gpu_tensor(ptr=0x200000))


class TestStreamRegistration:
    """Frozen no-ops in the shim, but real symbols on the wrapper surface."""

    def test_register_stream_calls_symbol(self, _fake_lib: _FakeLib) -> None:
        pa.register_stream(0xABC)
        (stream,) = _fake_lib.calls["phxFileStreamRegister"][0]
        assert stream.value == 0xABC

    def test_deregister_stream_calls_symbol(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["phxFileStreamDeregister"][0]
        assert stream.value == 0xABC

    def test_register_stream_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.stream_register_rc = -5
        with pytest.raises(RuntimeError, match="phxFileStreamRegister"):
            pa.register_stream(0xABC)

    def test_deregister_stream_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.stream_deregister_rc = -5
        with pytest.raises(RuntimeError, match="phxFileStreamDeregister"):
            pa.deregister_stream(0xABC)


class TestHandleRegistration:
    def test_register_handle_boxes_fd(self, _fake_lib: _FakeLib) -> None:
        # Identity boxing today: the handle equals the fd.
        assert pa.register_handle(42) == 42

    def test_register_handle_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.handle_register_rc = -9  # EBADF
        with pytest.raises(RuntimeError, match="phxFileHandleRegister"):
            pa.register_handle(42)

    def test_deregister_handle_calls_symbol(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_handle(42)
        (fh,) = _fake_lib.calls["phxFileHandleDeregister"][0]
        assert fh.value == 42


class TestSubmission:
    def test_stores_arguments_and_result(self) -> None:
        submission = pa.Submission(size=4096, file_offset=8192, buf_offset=512)
        assert submission._size.value == 4096
        assert submission._file_offset.value == 8192
        assert submission._buf_offset.value == 512
        assert submission.bytes_done == 0
        submission._bytes_done.value = 4096
        assert submission.bytes_done == 4096


class TestAsyncHandleIO:
    """Stream-ordered submissions through the frozen phxFile* surface."""

    def _handle(self) -> pa.AsyncHandle:
        return pa.AsyncHandle.from_fd(
            fd=5,
            handle=5,  # phx handle == fd (identity boxing)
            path="/mnt/nvme/lmcache_gds_slab.bin",
            writable=True,
        )

    def test_read_async_submits_stream_ordered(self, _fake_lib: _FakeLib) -> None:
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=8192,
            buf_offset=512,
            raw_stream=0x9,
        )
        assert len(_fake_lib.calls["phxFileReadAsync"]) == 1
        fh, buf, nb_p, fo_p, bo_p, bd_p, stream = _fake_lib.calls["phxFileReadAsync"][0]
        assert fh.value == 5
        assert buf.value == 0x300000
        # The submission's ctypes storage is handed in by reference, in
        # hipFile parameter order: file_offset before buf_offset.
        assert nb_p._obj is submission._size
        assert fo_p._obj is submission._file_offset
        assert bo_p._obj is submission._buf_offset
        assert bd_p._obj is submission._bytes_done
        assert nb_p._obj.value == 4096
        assert fo_p._obj.value == 8192
        assert bo_p._obj.value == 512
        assert stream.value == 0x9
        # The fake completed the transfer synchronously.
        assert submission.bytes_done == 4096

    def test_read_async_submission_error_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.read_async_rc = -22  # submission-level failure
        with pytest.raises(RuntimeError, match="phxFileReadAsync"):
            self._handle().read_async(
                buf_base=0x300000,
                size=4096,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )

    def test_read_async_defers_dma_error_to_bytes_done(
        self, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.read_async_bd = -28  # ENOSPC during the transfer
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        # cuFile contract: the submission is accepted; the failure surfaces
        # in bytes_done after the stream sync -- no synchronous raise.
        assert submission.bytes_done == -28

    def test_io_needs_no_stream_registration(self, _fake_lib: _FakeLib) -> None:
        # No register_stream call anywhere: an unregistered stream is
        # submitted on first use (no-registration model, like cuFile).
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert "phxFileStreamRegister" not in _fake_lib.calls
        assert submission.bytes_done == 4096

    def test_write_async_submits_stream_ordered(self, _fake_lib: _FakeLib) -> None:
        submission = self._handle().write_async(
            buf_base=0x300000,
            size=2048,
            file_offset=512,
            buf_offset=128,
            raw_stream=0x9,
        )
        assert len(_fake_lib.calls["phxFileWriteAsync"]) == 1
        fh, buf, nb_p, fo_p, bo_p, bd_p, stream = _fake_lib.calls["phxFileWriteAsync"][
            0
        ]
        assert fh.value == 5
        assert buf.value == 0x300000
        assert nb_p._obj.value == 2048
        assert fo_p._obj.value == 512
        assert bo_p._obj.value == 128
        assert stream.value == 0x9
        assert submission.bytes_done == 2048

    def test_write_async_defers_dma_error_to_bytes_done(
        self, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.write_async_bd = -5  # EIO during the transfer
        submission = self._handle().write_async(
            buf_base=0x300000,
            size=2048,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert submission.bytes_done == -5


class TestAsyncHandleLifecycle:
    def test_constructor_opens_and_registers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[tuple[str, int]] = []
        registered: list[int] = []

        def open_device(path: str, flags: int) -> int:
            opened.append((path, flags))
            return 33

        def register_device(fd: int) -> int:
            registered.append(fd)
            return fd

        monkeypatch.setattr(pa.os, "open", open_device)
        monkeypatch.setattr(pa, "register_handle", register_device)
        handle = pa.AsyncHandle("/mnt/nvme/lmcache_gds_slab.bin")
        assert opened == [("/mnt/nvme/lmcache_gds_slab.bin", pa.os.O_RDWR)]
        assert registered == [33]
        assert handle.fd == 33
        assert handle.path == "/mnt/nvme/lmcache_gds_slab.bin"
        assert handle.writable is True

    def test_constructor_closes_fd_on_registration_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[int] = []
        monkeypatch.setattr(pa.os, "open", lambda path, flags: 33)
        monkeypatch.setattr(
            pa,
            "register_handle",
            lambda fd: (_ for _ in ()).throw(RuntimeError("register failed")),
        )
        monkeypatch.setattr(pa.os, "close", closed.append)
        with pytest.raises(RuntimeError, match="register failed"):
            pa.AsyncHandle("/mnt/nvme/lmcache_gds_slab.bin")
        assert closed == [33]

    def test_close_deregisters_handle_and_closes_fd_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deregistered: list[int] = []
        closed: list[int] = []
        monkeypatch.setattr(pa, "deregister_handle", deregistered.append)
        monkeypatch.setattr(pa.os, "close", closed.append)
        handle = pa.AsyncHandle.from_fd(
            fd=5,
            handle=5,
            path="/mnt/nvme/lmcache_gds_slab.bin",
            writable=True,
        )
        handle.close()
        handle.close()
        assert deregistered == [5]
        assert closed == [5]
        assert handle.fd == -1


class TestCloseDriver:
    def test_calls_driver_close(self, _fake_lib: _FakeLib) -> None:
        pa.close_driver()
        assert _fake_lib.calls["phxFileDriverClose"] == [()]

    def test_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.driver_close_rc = -5
        with pytest.raises(RuntimeError, match="phxFileDriverClose"):
            pa.close_driver()
