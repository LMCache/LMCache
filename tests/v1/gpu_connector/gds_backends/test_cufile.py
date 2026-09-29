# SPDX-License-Identifier: Apache-2.0
"""cuFile tests with fake native bindings."""

# Standard
from collections.abc import Iterator
from unittest.mock import Mock
import ctypes
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector.gds_backends.cufile import AsyncHandle
from lmcache.v1.gpu_connector.gds_backends.cufile import Backend as CuFileBackend
from lmcache.v1.gpu_connector.gds_context import GDSContext


class _Error(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


@pytest.fixture
def bindings(monkeypatch: pytest.MonkeyPatch) -> Mock:
    bindings = Mock()
    bindings.CUfileError = _Error
    bindings.cuFileHandleRegister.return_value = 17
    for operation in (
        "cuFileReadAsync",
        "cuFileWriteAsync",
        "cuFileStreamRegister",
        "cuFileStreamDeregister",
        "cuFileBufRegister",
        "cuFileBufDeregister",
    ):
        function = getattr(bindings.libcufile, operation)
        function.argtypes = None
        function.return_value = _Error()
    monkeypatch.setitem(sys.modules, "cufile", Mock())
    monkeypatch.setitem(sys.modules, "cufile.bindings", bindings)
    return bindings


@pytest.fixture
def backend(bindings: Mock) -> Iterator[CuFileBackend]:
    backend = CuFileBackend()
    yield backend
    backend.close_driver()


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


@pytest.mark.parametrize("first_resource", ["handle", "buffer", "stream"])
def test_closing_one_backend_keeps_other_resources_alive(
    bindings: Mock, backend: CuFileBackend, first_resource: str
) -> None:
    # Unlike a bare Mock, closing this native session invalidates subsequent IO.
    active = False

    def open_driver() -> None:
        nonlocal active
        assert not active
        active = True

    def close_driver() -> None:
        nonlocal active
        active = False

    def require_active(*args: object) -> _Error:
        assert active, "native driver was closed while still in use"
        return _Error()

    bindings.cuFileDriverOpen.side_effect = open_driver
    bindings.cuFileDriverClose.side_effect = close_driver
    bindings.libcufile.cuFileBufRegister.side_effect = require_active
    bindings.libcufile.cuFileStreamRegister.side_effect = require_active
    bindings.libcufile.cuFileReadAsync.side_effect = require_active
    buffer = Mock(is_cuda=True)
    buffer.data_ptr.return_value = 0x1000
    buffer.numel.return_value = 4096
    buffer.element_size.return_value = 1
    other = CuFileBackend()
    try:
        backend.register_stream(7)
        if first_resource == "handle":
            other.register_handle(8)
        elif first_resource == "buffer":
            other.register_buffer(buffer)
        else:
            other.register_stream(9)
        bindings.cuFileDriverOpen.assert_called_once()
        backend.deregister_stream(7)
        backend.close_driver()
        bindings.cuFileDriverClose.assert_not_called()
        if first_resource != "buffer":
            other.register_buffer(buffer)
        other.register_stream(11)
        native_handle = 17 if first_resource == "handle" else other.register_handle(8)
        handle = AsyncHandle(other, -1, native_handle, "/slab")
        handle.read_async(0x1000, 4096, 0, 0, 11)
        bindings.libcufile.cuFileReadAsync.assert_called_once()
        other.deregister_handle(native_handle)
        other.deregister_stream(11)
        if first_resource == "stream":
            other.deregister_stream(9)
        other.deregister_buffer(buffer)
    finally:
        backend.close_driver()
        other.close_driver()
    bindings.cuFileDriverClose.assert_called_once()


def test_failed_context_setup_keeps_peer_driver_alive(
    bindings: Mock, backend: CuFileBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = CuFileBackend()
    monkeypatch.setattr(other, "validate_environment", lambda: None)

    def fail_slab(location: str, size: int, direct_io: bool) -> AsyncHandle:
        other.register_handle(8)
        other.deregister_handle(17)
        raise RuntimeError("slab setup failed")

    monkeypatch.setattr(other, "open_slab", fail_slab)
    backend.register_stream(7)
    try:
        context = GDSContext(other)
        with pytest.raises(RuntimeError, match="slab setup failed"):
            context.initialize(GdsL1Config(file_location="/slab", size_in_bytes=4096))
        bindings.cuFileDriverClose.assert_not_called()
        backend.register_stream(9)
        bindings.cuFileDriverOpen.assert_called_once()
        backend.deregister_stream(7)
        backend.deregister_stream(9)
    finally:
        other.close_driver()
        backend.close_driver()
    bindings.cuFileDriverClose.assert_called_once()


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
