# SPDX-License-Identifier: Apache-2.0
"""Phoenix tests with a fake phxFile shim."""

# Standard
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.gds_backends import phx as pa


class _FakeLib:
    """Stand-in for ``libphxfile.so`` that records all symbol calls."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}
        self.has_async_api = True
        self.register_rc = 0
        self.deregister_rc = 0
        self.handle_register_rc = 0
        self.driver_close_rc = 0
        self.read_async_rc = 0
        self.write_async_rc = 0
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
                args[0]._obj.value = args[1].value
                return self.handle_register_rc
            if name == "phxFileBufRegister":
                return self.register_rc
            if name == "phxFileBufDeregister":
                return self.deregister_rc
            if name == "phxFileDriverClose":
                return self.driver_close_rc
            if name == "phxFileReadAsync":
                bd = self.read_async_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.read_async_rc
            if name == "phxFileWriteAsync":
                bd = self.write_async_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.write_async_rc
            return 0

        return _record


@pytest.fixture
def backend() -> pa.Backend:
    return pa.Backend()


@pytest.fixture(autouse=True)
def _fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the lazy-loaded CDLL with the fake frozen-ABI library."""
    lib = _FakeLib()
    monkeypatch.setattr(pa.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setattr(pa.ctypes.util, "find_library", lambda name: None)
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


class TestLibLoading:
    def test_missing_async_symbols_fail_fast(
        self,
        backend: pa.Backend,
        _fake_lib: _FakeLib,
    ) -> None:
        _fake_lib.has_async_api = False
        with pytest.raises(RuntimeError, match="stream-ordered API"):
            backend.library()


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self, backend: pa.Backend) -> None:
        with pytest.raises(ValueError, match="CUDA or ROCm"):
            backend.register_buffer(SimpleNamespace(is_cuda=False))  # type: ignore[arg-type]

    def test_rejects_empty_tensor(self, backend: pa.Backend) -> None:
        with pytest.raises(ValueError, match="empty"):
            backend.register_buffer(_gpu_tensor(nbytes=0))

    def test_passes_addr_and_raw_length(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_buffer(_gpu_tensor(ptr=0x200000, nbytes=4096))
        addr, length = _fake_lib.calls["phxFileBufRegister"][0]
        assert addr.value == 0x200000
        assert length.value == 4096
        backend.deregister_buffer(_gpu_tensor(ptr=0x200000))
        (addr,) = _fake_lib.calls["phxFileBufDeregister"][0]
        assert addr.value == 0x200000

    def test_register_failure_raises(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.register_rc = -19  # ENODEV (no phxfs device present)
        with pytest.raises(RuntimeError, match="phxFileBufRegister"):
            backend.register_buffer(_gpu_tensor())


class TestBufferDeregistration:
    def test_deregister_failure_raises(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.deregister_rc = -22  # EINVAL
        with pytest.raises(RuntimeError, match="phxFileBufDeregister"):
            backend.deregister_buffer(_gpu_tensor(ptr=0x200000))


class TestHandleRegistration:
    def test_register_handle_boxes_fd(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        assert backend.register_handle(42) == 42
        backend.deregister_handle(42)
        (handle,) = _fake_lib.calls["phxFileHandleDeregister"][0]
        assert handle.value == 42

    def test_register_handle_failure_raises(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.handle_register_rc = -9  # EBADF
        with pytest.raises(RuntimeError, match="phxFileHandleRegister"):
            backend.register_handle(42)


class TestAsyncHandleIO:
    """Stream-ordered submissions through the frozen phxFile* surface."""

    def _handle(self, backend: pa.Backend) -> pa.AsyncHandle:
        return pa.AsyncHandle(
            backend=backend,
            fd=5,
            handle=5,  # phx handle == fd (identity boxing)
            path="/mnt/nvme/lmcache_gds_slab.bin",
        )

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_marshals_arguments(
        self, backend: pa.Backend, _fake_lib: _FakeLib, operation: str
    ) -> None:
        handle = self._handle(backend)
        submit = handle.read_async if operation == "read" else handle.write_async
        submission = submit(
            buf_base=0x300000,
            size=4096,
            file_offset=8192,
            buf_offset=512,
            raw_stream=0x9,
        )
        calls = _fake_lib.calls[f"phxFile{operation.title()}Async"]
        assert len(calls) == 1
        fh, buf, nb_p, fo_p, bo_p, bd_p, stream = calls[0]
        assert fh.value == 5
        assert buf.value == 0x300000
        assert nb_p._obj is submission.size
        assert fo_p._obj is submission.file_offset
        assert bo_p._obj is submission.buf_offset
        assert bd_p._obj is submission.result
        assert nb_p._obj.value == 4096
        assert fo_p._obj.value == 8192
        assert bo_p._obj.value == 512
        assert stream.value == 0x9
        assert submission.bytes_done == 4096
        assert "phxFileStreamRegister" not in _fake_lib.calls

    def test_read_async_submission_error_raises(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.read_async_rc = -22  # submission-level failure
        with pytest.raises(RuntimeError, match="phxFileReadAsync"):
            self._handle(backend).read_async(
                buf_base=0x300000,
                size=4096,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_dma_error_is_reported_in_result(
        self, backend: pa.Backend, _fake_lib: _FakeLib, operation: str
    ) -> None:
        setattr(_fake_lib, f"{operation}_async_bd", -28)  # ENOSPC
        handle = self._handle(backend)
        submit = handle.read_async if operation == "read" else handle.write_async
        submission = submit(
            buf_base=0x300000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert submission.bytes_done == -28


class TestCloseDriver:
    def test_registration_does_not_explicitly_open_driver(
        self, backend: pa.Backend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_handle(7)
        backend.register_buffer(_gpu_tensor())
        backend.register_stream(13)
        assert "phxFileDriverOpen" not in _fake_lib.calls
        backend.close_driver()
        backend.close_driver()
        assert _fake_lib.calls["phxFileDriverClose"] == [(), ()]

    def test_failure_raises(self, backend: pa.Backend, _fake_lib: _FakeLib) -> None:
        _fake_lib.driver_close_rc = -5
        backend.library()
        with pytest.raises(RuntimeError, match="phxFileDriverClose"):
            backend.close_driver()
