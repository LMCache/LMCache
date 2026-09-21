# SPDX-License-Identifier: Apache-2.0
"""Backend contracts and context integration."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock
import os
import sys
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import gds_backends, gds_context
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends._file import FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_context import GDSContext, SlabDirection
from lmcache.v1.memory_management import GDSMemoryObject


class OtherBackend(GDSBackend):
    """A backend with no CUDA/ROCm requirement or filesystem dependency."""

    name = "other"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[object] = []

    def open_slab(self, location: str, size: int, direct_io: bool) -> GDSHandle:
        self.calls.append(("open", location, size, direct_io))
        return OtherHandle(self, -1, 0, location)

    def open_handle(self, fd: int, path: str) -> GDSHandle:
        return OtherHandle(self, fd, self.register_handle(fd), path)

    def register_handle(self, fd: int) -> int:
        return fd

    def deregister_handle(self, handle: int) -> None:
        self.calls.append(("deregister", handle))

    def register_buffer(self, buf: torch.Tensor) -> None:
        self.calls.append(("buffer", buf.numel()))

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        self.calls.append("deregister_buffer")

    def register_stream(self, raw_stream: int) -> None:
        self.calls.append(("stream", raw_stream))

    def deregister_stream(self, raw_stream: int) -> None:
        self.calls.append("deregister_stream")

    def close_driver(self) -> None:
        self.calls.append("close_driver")


class OtherHandle(GDSHandle):
    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        return Submission(size, file_offset, buf_offset)

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        return Submission(size, file_offset, buf_offset)


@pytest.fixture
def gpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", "6.3")


@pytest.fixture
def extra_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Add a real module to the scanned directory without changing the factory."""
    (tmp_path / "other.py").write_text(
        f"from {__name__} import OtherBackend\n"
        "class Backend(OtherBackend):\n"
        "    @classmethod\n"
        "    def is_default(cls):\n"
        "        return True\n"
    )
    monkeypatch.setattr(
        gds_backends, "__path__", [*gds_backends.__path__, str(tmp_path)]
    )
    yield
    sys.modules.pop(f"{gds_backends.__name__}.other", None)


@pytest.mark.parametrize(
    ("cuda_version", "hip_version", "name", "error"),
    [
        (None, "6.3", "cufile", "CUDA"),
        ("12.9", None, "hipfile", "ROCm"),
        (None, None, "ugds", "ROCm or CUDA"),
        (None, None, "phx", "ROCm or CUDA"),
        (None, None, "auto", "no default GDS backend"),
    ],
)
def test_existing_backend_restrictions_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    hip_version: str | None,
    name: str,
    error: str,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    with pytest.raises(ValueError, match=error):
        create_backend(name)


@pytest.mark.parametrize("name", ["ugds", "phx"])
@pytest.mark.parametrize(("cuda", "hip"), [("12.9", None), (None, "6.3")])
def test_explicit_backend_accepts_both_existing_builds(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cuda: str | None,
    hip: str | None,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", hip)
    assert create_backend(name).name == name


@pytest.mark.usefixtures("extra_backend")
@pytest.mark.parametrize("selection", ["other", "auto"])
def test_new_backend_needs_no_public_platform_or_dispatch_change(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    backend = create_backend(selection)
    assert isinstance(backend, OtherBackend)
    ctx = GDSContext(backend)
    ctx.initialize(GdsL1Config(file_location="device://slab", size_in_bytes=1))
    assert backend.calls == [("open", "device://slab", 4096, True)]
    assert ctx.backend is backend
    ctx.close()
    assert backend.calls[-1] == "close_driver"
    assert not ctx.initialized


def test_context_rejects_reinitializing_a_live_slab() -> None:
    backend = OtherBackend()
    ctx = GDSContext(backend)
    config = GdsL1Config(file_location="device://slab", size_in_bytes=4096)
    ctx.initialize(config)
    with pytest.raises(RuntimeError, match="already initialized"):
        ctx.initialize(config)
    assert len(backend.calls) == 1
    ctx.close()


def test_context_releases_driver_after_open_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OtherBackend()
    monkeypatch.setattr(backend, "open_slab", Mock(side_effect=RuntimeError("open")))
    ctx = GDSContext(backend)
    with pytest.raises(RuntimeError, match="open"):
        ctx.initialize(GdsL1Config(file_location="device://slab", size_in_bytes=1))
    assert backend.calls == ["close_driver"]
    assert not ctx.initialized


def test_context_keeps_submission_until_completion_and_closes_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OtherBackend()
    ctx = GDSContext(backend)
    ctx.initialize(GdsL1Config(file_location="device://slab", size_in_bytes=4096))
    stream = Mock(cuda_stream=7)
    event = Mock()
    event.query.return_value = False
    monkeypatch.setattr(gds_context.torch_dev, "current_stream", lambda: stream)
    monkeypatch.setattr(gds_context.torch_dev, "Event", lambda: event)
    monkeypatch.setattr(
        gds_context.torch_dev,
        "synchronize",
        lambda **kw: backend.calls.append("synchronize"),
    )
    monkeypatch.setattr(gds_context, "_SUBMISSION_CHECKPOINT_EVERY", 1)
    submissions: list[weakref.ReferenceType[Submission]] = []

    def write(
        self: OtherHandle,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        sub = Submission(size, file_offset, buf_offset)
        submissions.append(weakref.ref(sub))
        return sub

    monkeypatch.setattr(OtherHandle, "write_async", write)
    monkeypatch.setattr(
        OtherHandle, "close", lambda self: backend.calls.append("close_handle")
    )
    buf = torch.empty(4096, dtype=torch.uint8)
    ctx.register_gpu_buffer(buf)
    memory = Mock(spec=GDSMemoryObject, slab_offset=0)
    memory.get_size.return_value = 4096
    ctx.transfer_async(memory, buf, SlabDirection.WRITE)
    assert submissions[0]() is not None
    event.record.assert_called_once_with(stream)
    event.query.return_value = True
    ctx.transfer_async(memory, buf, SlabDirection.WRITE)
    assert submissions[0]() is None
    ctx.close()
    assert backend.calls[-5:] == [
        "synchronize",
        "deregister_buffer",
        "deregister_stream",
        "close_handle",
        "close_driver",
    ]


@pytest.mark.parametrize("fails", [False, True])
def test_handle_closes_fd_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fails: bool,
) -> None:
    backend = OtherBackend()
    fd = os.open(tmp_path / "slab", os.O_CREAT | os.O_RDWR, 0o600)
    handle = backend.open_handle(fd, "slab")
    deregister = Mock(side_effect=RuntimeError("deregister") if fails else None)
    monkeypatch.setattr(backend, "deregister_handle", deregister)
    if fails:
        with pytest.raises(RuntimeError, match="deregister"):
            handle.close()
    else:
        handle.close()
    handle.close()
    deregister.assert_called_once()
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("name", ["cufile", "hipfile", "ugds", "phx"])
@pytest.mark.usefixtures("gpu_build")
def test_registration_failure_closes_owned_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    backend = create_backend(name)
    fd = os.open(tmp_path / "slab", os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(
        backend, "register_handle", Mock(side_effect=RuntimeError("register"))
    )
    with pytest.raises(RuntimeError, match="register"):
        backend.open_handle(fd, "slab")
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("direct_io", [False, True])
@pytest.mark.usefixtures("gpu_build")
def test_file_slab_creation_order(
    monkeypatch: pytest.MonkeyPatch,
    direct_io: bool,
) -> None:
    backend = create_backend("cufile")
    assert isinstance(backend, FileGDSBackend)
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
