# SPDX-License-Identifier: Apache-2.0
"""Process-wide driver ownership, exercised without native storage libraries."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import ctypes
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector._gds_async import GDSBackend, GDSHandle
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends.cufile import Backend as CuFileBackend
from lmcache.v1.gpu_connector.gds_backends.hipfile import Backend as HipFileBackend
from lmcache.v1.gpu_connector.gds_backends.hipfile import _HipFileError
from lmcache.v1.gpu_connector.gds_backends.phx import Backend as PhxBackend
from lmcache.v1.gpu_connector.gds_backends.ugds import Backend as UgdsBackend
from lmcache.v1.gpu_connector.gds_backends.ugds import _uGDSError_t
from lmcache.v1.gpu_connector.gds_context import GDSContext

Drivers = tuple[GDSBackend, GDSBackend, Mock, Mock]


class _CuFileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


@pytest.fixture(
    params=[CuFileBackend, HipFileBackend, UgdsBackend, PhxBackend],
    ids=lambda backend: backend.name,
)
def drivers(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Drivers]:
    backend_type = request.param
    prefix, result = {
        CuFileBackend: ("cuFile", _CuFileError()),
        HipFileBackend: ("hipFile", _HipFileError()),
        UgdsBackend: ("uGDS", _uGDSError_t()),
        PhxBackend: ("phxFile", 0),
    }[backend_type]
    lib = Mock()
    for operation in ("DriverOpen", "DriverClose", "StreamRegister", "ReadAsync"):
        symbol = getattr(lib, prefix + operation)
        symbol.return_value = result
        symbol.argtypes = None
    bindings = Mock(libcufile=lib, CUfileError=_CuFileError)
    bindings.cuFileDriverOpen = lib.cuFileDriverOpen
    bindings.cuFileDriverClose = lib.cuFileDriverClose
    monkeypatch.setitem(sys.modules, "cufile", Mock())
    monkeypatch.setitem(sys.modules, "cufile.bindings", bindings)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args: lib)
    monkeypatch.setattr(torch.version, "cuda", "test")
    monkeypatch.setattr(torch.version, "hip", "test")
    a, b = create_backend(backend_type.name), backend_type()
    try:
        yield (
            a,
            b,
            getattr(lib, prefix + "DriverOpen"),
            getattr(lib, prefix + "DriverClose"),
        )
    finally:
        a.close_driver()
        b.close_driver()


def test_closing_one_backend_keeps_other_backend_driver_alive(drivers: Drivers) -> None:
    a, b, opened, closed = drivers
    a.register_stream(11)
    b.register_stream(22)
    b.close_driver()
    b.close_driver()
    closed.assert_not_called()
    a.register_stream(33)
    if not isinstance(a, PhxBackend):
        opened.assert_called_once()
    a.close_driver()
    closed.assert_called_once()


def test_unused_backend_cannot_close_another_backend_driver(drivers: Drivers) -> None:
    a, b, _, closed = drivers
    a.register_stream(11)
    b.close_driver()
    closed.assert_not_called()
    a.close_driver()
    closed.assert_called_once()


def test_can_reopen_after_last_owner_closes(drivers: Drivers) -> None:
    a, b, opened, closed = drivers
    a.register_stream(11)
    b.register_stream(22)
    a.close_driver()
    b.close_driver()
    closed.assert_called_once()
    b.register_stream(33)
    if not isinstance(b, PhxBackend):
        assert opened.call_count == 2
    b.close_driver()
    assert closed.call_count == 2


def test_failed_context_initialization_preserves_other_owner(
    drivers: Drivers, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, closed = drivers
    a.register_stream(11)

    def fail_slab(location: str, size: int, direct_io: bool) -> GDSHandle:
        b.register_stream(22)
        raise RuntimeError("slab registration failed")

    monkeypatch.setattr(b, "open_slab", fail_slab)
    context = GDSContext(backend=b)
    with pytest.raises(RuntimeError, match="slab registration failed"):
        context.initialize(GdsL1Config(file_location="/slab", size_in_bytes=4096))
    assert not context.initialized
    closed.assert_not_called()
    a.register_stream(33)
    a.close_driver()
    closed.assert_called_once()


def test_failed_open_does_not_acquire_ownership() -> None:
    driver = SharedDriver()
    a, b = object(), object()
    opened, closed = Mock(side_effect=RuntimeError("open failed")), Mock()
    with pytest.raises(RuntimeError, match="open failed"):
        driver.acquire(a, opened)
    driver.release(a, closed)
    closed.assert_not_called()
    opened.side_effect = None
    driver.acquire(b, opened)
    assert opened.call_count == 2
    driver.release(b, closed)
    closed.assert_called_once()


def test_driver_ownership_is_thread_safe() -> None:
    driver = SharedDriver()
    owners = [object() for _ in range(16)]
    opened, closed = Mock(), Mock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda owner: driver.acquire(owner, opened), owners))
        opened.assert_called_once()
        list(pool.map(lambda owner: driver.release(owner, closed), owners))
        closed.assert_called_once()
