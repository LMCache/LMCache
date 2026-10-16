# SPDX-License-Identifier: Apache-2.0
"""Descriptor ownership and file slab setup."""

# Standard
from pathlib import Path
from unittest.mock import Mock
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSBackend
from lmcache.v1.gpu_connector.gds_backends.cufile import Backend


@pytest.mark.parametrize("fails", [False, True])
def test_handle_closes_fd_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fails: bool,
) -> None:
    backend = Backend()
    fd = os.open(tmp_path / "slab", os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(backend, "register_handle", Mock(return_value=17))
    handle = backend.open_handle(fd, "slab")
    deregister = Mock(side_effect=RuntimeError("deregister") if fails else None)
    monkeypatch.setattr(backend, "deregister_handle", deregister)
    if fails:
        with pytest.raises(RuntimeError, match="deregister"):
            handle.close()
    else:
        handle.close()
    handle.close()
    deregister.assert_called_once_with(17)
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("name", ["cufile", "hipfile", "ugds", "phx"])
def test_registration_failure_closes_owned_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", "6.3")
    backend = create_backend(name)
    assert isinstance(backend, FDGDSBackend)
    fd = os.open(tmp_path / "slab", os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(
        backend, "register_handle", Mock(side_effect=RuntimeError("register"))
    )
    with pytest.raises(RuntimeError, match="register"):
        backend.open_handle(fd, "slab")
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("direct_io", [False, True])
def test_file_slab_creation_order(
    monkeypatch: pytest.MonkeyPatch,
    direct_io: bool,
) -> None:
    backend = Backend()
    calls: list[object] = []

    def open_file(path: str, flags: int, *args: object) -> int:
        calls.append(("open", flags))
        return 7

    def register(fd: int) -> int:
        calls.append(("register", fd))
        return 17

    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: calls.append("mkdir"))
    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(
        os, "posix_fallocate", lambda *a: calls.append(("allocate", a)), raising=False
    )
    monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
    monkeypatch.setattr(backend, "register_handle", register)
    monkeypatch.setattr(backend, "deregister_handle", lambda handle: None)
    handle = backend.open_slab("/slabs", 4096, direct_io)
    assert handle.path == "/slabs/lmcache_gds_slab.bin"
    assert calls == [
        "mkdir",
        ("open", os.O_CREAT | os.O_RDWR | os.O_TRUNC),
        ("allocate", (7, 0, 4096)),
        ("close", 7),
        ("open", os.O_RDWR | (os.O_DIRECT if direct_io else 0)),
        ("register", 7),
    ]
    handle.close()
