# SPDX-License-Identifier: Apache-2.0
"""Exercise the cuFile backend against fake native bindings, without CUDA."""

# Standard
from unittest.mock import Mock
import ctypes
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.gds_backends.cufile import Backend as CuFileBackend


class _Error(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


@pytest.fixture
def bindings(monkeypatch: pytest.MonkeyPatch) -> Mock:
    bindings = Mock()
    bindings.CUfileError = _Error
    bindings.cuFileHandleRegister.return_value = 17
    for operation in ("cuFileReadAsync", "cuFileWriteAsync", "cuFileStreamRegister"):
        function = getattr(bindings.libcufile, operation)
        function.argtypes = None
        function.return_value = _Error()
    monkeypatch.setitem(sys.modules, "cufile", Mock())
    monkeypatch.setitem(sys.modules, "cufile.bindings", bindings)
    return bindings


@pytest.fixture
def backend() -> CuFileBackend:
    return CuFileBackend()


def test_driver_opens_once_and_can_reopen_after_close(
    bindings: Mock, backend: CuFileBackend
) -> None:
    backend.register_stream(7)
    backend.register_stream(7)
    bindings.cuFileDriverOpen.assert_called_once()
    backend.close_driver()
    backend.close_driver()
    bindings.cuFileDriverClose.assert_called_once()
    backend.register_stream(7)
    assert bindings.cuFileDriverOpen.call_count == 2
    backend.close_driver()


def test_failed_driver_open_can_retry(bindings: Mock, backend: CuFileBackend) -> None:
    bindings.cuFileDriverOpen.side_effect = RuntimeError("open failed")
    with pytest.raises(RuntimeError, match="open failed"):
        backend.register_stream(7)
    backend.close_driver()
    bindings.cuFileDriverClose.assert_not_called()
    bindings.cuFileDriverOpen.side_effect = None
    backend.register_stream(7)
    assert bindings.cuFileDriverOpen.call_count == 2
    backend.close_driver()
    bindings.cuFileDriverClose.assert_called_once()


def test_failed_driver_close_resets_state(
    bindings: Mock, backend: CuFileBackend
) -> None:
    backend.register_stream(7)
    bindings.cuFileDriverClose.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        backend.close_driver()
    backend.close_driver()
    bindings.cuFileDriverClose.assert_called_once()
    bindings.cuFileDriverClose.side_effect = None
    backend.register_stream(7)
    assert bindings.cuFileDriverOpen.call_count == 2
    backend.close_driver()


def test_driver_state_is_per_instance(bindings: Mock, backend: CuFileBackend) -> None:
    other = CuFileBackend()
    backend.register_stream(7)
    other.register_stream(9)
    assert bindings.cuFileDriverOpen.call_count == 2
    backend.close_driver()
    bindings.cuFileDriverClose.assert_called_once()
    other.close_driver()
    assert bindings.cuFileDriverClose.call_count == 2


@pytest.mark.parametrize("operation", ["read", "write"])
def test_io_retains_native_argument_storage(
    bindings: Mock, backend: CuFileBackend, operation: str
) -> None:
    handle = backend.open_handle(7, "/slab")
    submit = handle.read_async if operation == "read" else handle.write_async
    sub = submit(0x1000, 4096, 8192, 512, 13)
    native = (
        bindings.libcufile.cuFileReadAsync
        if operation == "read"
        else bindings.libcufile.cuFileWriteAsync
    )
    native_handle, buf, size, offset, buf_offset, result, stream = native.call_args.args
    assert native_handle == 17
    assert buf.value == 0x1000
    assert size._obj is sub.size
    assert offset._obj is sub.file_offset
    assert buf_offset._obj is sub.buf_offset
    assert result._obj is sub.result
    assert stream.value == 13
    result._obj.value = 4096
    assert sub.bytes_done == 4096


def test_io_submission_error_propagates(bindings: Mock, backend: CuFileBackend) -> None:
    handle = backend.open_handle(7, "/slab")
    bindings.libcufile.cuFileReadAsync.return_value = _Error(err=5, cu_err=700)
    with pytest.raises(RuntimeError, match="cuFileReadAsync"):
        handle.read_async(0x1000, 4096, 0, 0, 7)
