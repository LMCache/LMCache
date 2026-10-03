# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SmartIO muFile GDS backend."""

# Standard
from types import SimpleNamespace
from typing import Any
import ctypes

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.gds_backends import mufile


def _ok() -> mufile._MUfileError:
    return mufile._MUfileError(err=mufile._MUFILE_SUCCESS)


def _fake_tensor(ptr: int = 0x3000, nbytes: int = 4096):
    return SimpleNamespace(
        is_musa=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class _FakeLib:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[Any, ...]] = {}

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> Any:
            self.calls[name] = args
            if name == "muFileHandleRegister":
                args[0]._obj.value = 0xFEED
            if name in ("muFileReadAsync", "muFileWriteAsync"):
                return 0
            return _ok()

        return call


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch):
    value = mufile.Backend()
    library = _FakeLib()
    monkeypatch.setattr(value, "library", lambda: library)
    yield value, library
    value.close_driver()


def test_mufile_requires_musa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mufile, "torch_device_type", "cuda")
    with pytest.raises(ValueError, match="MUSA"):
        mufile.Backend().validate_environment()


def test_mufile_is_default_on_musa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mufile, "torch_device_type", "musa")
    assert mufile.Backend.is_default()


def test_mufile_requires_direct_io(backend) -> None:
    value, _ = backend
    with pytest.raises(ValueError, match="direct-io"):
        value.open_slab("/unused", 4096, direct_io=False)


def test_register_handle_retains_descriptor_until_deregister(backend) -> None:
    value, library = backend
    handle = value.register_handle(42)
    assert handle == 0xFEED
    assert handle in value._handle_descr_registry

    observed: dict[str, bool] = {}

    def deregister(native_handle):
        observed["retained"] = handle in value._handle_descr_registry
        return _ok()

    library.muFileHandleDeregister = deregister
    value.deregister_handle(handle)
    assert observed["retained"]
    assert handle not in value._handle_descr_registry


def test_deregister_failure_keeps_descriptor(backend) -> None:
    value, library = backend
    handle = value.register_handle(42)
    library.muFileHandleDeregister = lambda *_args: mufile._MUfileError(err=7)
    with pytest.raises(RuntimeError, match="muFileHandleDeregister"):
        value.deregister_handle(handle)
    assert handle in value._handle_descr_registry


def test_register_buffer_and_stream(backend) -> None:
    value, library = backend
    value.register_buffer(_fake_tensor(ptr=0x4000, nbytes=8192))
    base, length, flags = library.calls["muFileBufRegister"]
    assert base.value == 0x4000
    assert length.value == 8192
    assert flags.value == 0

    value.register_stream(0xABC)
    stream, flags = library.calls["muFileStreamRegister"]
    assert stream.value == 0xABC
    assert flags == mufile._STREAM_REGISTER_FLAGS


def test_rejects_non_musa_buffer(backend) -> None:
    value, _ = backend
    with pytest.raises(ValueError, match="MUSA"):
        value.register_buffer(SimpleNamespace(is_musa=False))


def test_async_io_uses_shared_submission(backend) -> None:
    value, library = backend

    def read(*args):
        args[5].contents.value = 4096
        library.calls["muFileReadAsync"] = args
        return 0

    library.muFileReadAsync = read
    handle = mufile.AsyncHandle(value, 5, 0xFEED, "/slab")
    sub = handle.read_async(
        buf_base=0x3000,
        size=4096,
        file_offset=0,
        buf_offset=0,
        raw_stream=0x9,
    )
    assert sub.bytes_done == 4096
    native = library.calls["muFileReadAsync"]
    assert native[0].value == 0xFEED
    assert native[1].value == 0x3000
    assert native[6].value == 0x9


def test_async_io_rejects_unaligned_operands(backend) -> None:
    value, library = backend
    handle = mufile.AsyncHandle(value, 5, 0xFEED, "/slab")
    with pytest.raises(ValueError, match="4096"):
        handle.write_async(0x3001, 4096, 0, 0, 0x9)
    assert "muFileWriteAsync" not in library.calls


def test_async_submit_error(backend) -> None:
    value, library = backend
    library.muFileWriteAsync = lambda *_args: -5
    handle = mufile.AsyncHandle(value, 5, 0xFEED, "/slab")
    with pytest.raises(RuntimeError, match="muFileWriteAsync"):
        handle.write_async(0x3000, 4096, 0, 0, 0x9)


def test_mufile_struct_layout() -> None:
    assert ctypes.sizeof(mufile._MUfileError) == 4
    assert ctypes.sizeof(mufile._MUFileDescr) == 16
    assert mufile._MUFileDescr.type.offset == 0
    assert mufile._MUFileDescr.handle.offset == 8
