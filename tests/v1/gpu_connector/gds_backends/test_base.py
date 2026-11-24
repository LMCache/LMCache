# SPDX-License-Identifier: Apache-2.0
"""Backend contracts and context integration."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock
import sys
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import gds_backends, gds_context
from lmcache.v1.gpu_connector._gds_backends import create_backend
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
        return OtherHandle(self, location)

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
    def __init__(self, backend: OtherBackend, path: str) -> None:
        super().__init__(path)
        self._backend = backend
        self._closed = False

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

    def close(self) -> None:
        if not self._closed:
            self._backend.calls.append("close_handle")
            self._closed = True


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
    assert backend.calls[-2:] == ["close_handle", "close_driver"]
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


@pytest.mark.parametrize("direction", [SlabDirection.READ, SlabDirection.WRITE])
def test_context_keeps_submission_until_completion_and_closes_in_order(
    monkeypatch: pytest.MonkeyPatch,
    direction: SlabDirection,
) -> None:
    backend = OtherBackend()
    ctx = GDSContext(backend)
    ctx.initialize(GdsL1Config(file_location="device://slab", size_in_bytes=4096))
    stream = Mock(cuda_stream=7)
    event = Mock()
    event.is_complete.return_value = False
    record_completion = Mock(return_value=event)
    monkeypatch.setattr(
        gds_context.platform_stream, "current_stream", lambda device: stream
    )
    monkeypatch.setattr(
        gds_context.platform_stream,
        "stream_handle",
        lambda device, stream: stream.cuda_stream,
    )
    monkeypatch.setattr(
        gds_context.platform_stream, "record_completion_event", record_completion
    )
    monkeypatch.setattr(
        gds_context.platform_stream,
        "synchronize_device",
        lambda device: backend.calls.append("synchronize"),
    )
    monkeypatch.setattr(gds_context, "_SUBMISSION_CHECKPOINT_EVERY", 1)
    submissions: list[weakref.ReferenceType[Submission]] = []

    def submit(
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

    monkeypatch.setattr(
        OtherHandle,
        "read_async" if direction is SlabDirection.READ else "write_async",
        submit,
    )
    buf = torch.empty(4096, dtype=torch.uint8)
    ctx.register_gpu_buffer(buf)
    memory = Mock(spec=GDSMemoryObject, slab_offset=0)
    memory.get_size.return_value = 4096
    ctx.transfer_async(memory, buf, direction)
    assert submissions[0]() is not None
    record_completion.assert_called_once_with(buf.device, stream)
    event.is_complete.return_value = True
    ctx.transfer_async(memory, buf, direction)
    assert submissions[0]() is None
    ctx.close()
    assert backend.calls[-5:] == [
        "synchronize",
        "deregister_buffer",
        "deregister_stream",
        "close_handle",
        "close_driver",
    ]


class _ScriptedDriver:
    """Test harness: a GDSContext with a fake driver that completes each
    transfer with a set byte count.

    The real driver writes Submission.result when the stream runs the copy. The
    fake writes it at submit time, which the context cannot tell apart once
    synchronize_and_check has synchronized the stream.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.stream = Mock(cuda_stream=7)
        self.event = Mock()
        self.event.is_complete.return_value = False
        platform = gds_context.platform_stream
        monkeypatch.setattr(platform, "current_stream", lambda device: self.stream)
        monkeypatch.setattr(
            platform, "stream_handle", lambda device, stream: stream.cuda_stream
        )
        monkeypatch.setattr(
            platform, "record_completion_event", lambda device, stream: self.event
        )
        monkeypatch.setattr(platform, "synchronize_stream", lambda device, stream: None)
        monkeypatch.setattr(platform, "synchronize_device", lambda device: None)
        monkeypatch.setattr(OtherHandle, "read_async", self._submit)
        monkeypatch.setattr(OtherHandle, "write_async", self._submit)
        self._bytes_done = 0
        self.ctx = GDSContext(OtherBackend())
        self.ctx.initialize(
            GdsL1Config(file_location="device://slab", size_in_bytes=4096)
        )
        self.buf = torch.empty(4096, dtype=torch.uint8)
        self.ctx.register_gpu_buffer(self.buf)
        self.memory = Mock(spec=GDSMemoryObject, slab_offset=0)
        self.memory.get_size.return_value = 4096

    def transfer(self, direction: SlabDirection, bytes_done: int) -> None:
        """Run one 4096-byte transfer that the driver completes with bytes_done."""
        self._bytes_done = bytes_done
        self.ctx.transfer_async(self.memory, self.buf, direction)

    def check(self) -> None:
        """Call synchronize_and_check on the stream the transfers used."""
        self.ctx.synchronize_and_check(self.buf.device, self.stream)

    def _submit(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        sub = Submission(size, file_offset, buf_offset)
        sub.result.value = self._bytes_done
        return sub


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ScriptedDriver]:
    driver = _ScriptedDriver(monkeypatch)
    yield driver
    driver.ctx.close()


@pytest.mark.parametrize("direction", [SlabDirection.READ, SlabDirection.WRITE])
@pytest.mark.parametrize("bytes_done", [2048, -5])
def test_synchronize_and_check_raises_on_short_or_failed_transfer(
    driver: _ScriptedDriver, direction: SlabDirection, bytes_done: int
) -> None:
    driver.transfer(direction, bytes_done)
    with pytest.raises(
        RuntimeError, match=f"expected 4096 bytes, completed {bytes_done}"
    ):
        driver.check()


@pytest.mark.parametrize("direction", [SlabDirection.READ, SlabDirection.WRITE])
def test_synchronize_and_check_accepts_complete_transfer(
    driver: _ScriptedDriver, direction: SlabDirection
) -> None:
    driver.transfer(direction, 4096)
    driver.check()


def test_synchronize_and_check_reports_failure_from_released_batch(
    driver: _ScriptedDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A checkpoint after every transfer with a completed event releases the
    # failed batch from inflight before synchronize_and_check runs; the later
    # successful transfer must not hide it.
    monkeypatch.setattr(gds_context, "_SUBMISSION_CHECKPOINT_EVERY", 1)
    driver.event.is_complete.return_value = True
    driver.transfer(SlabDirection.READ, -5)
    driver.transfer(SlabDirection.READ, 4096)
    with pytest.raises(RuntimeError, match="completed -5"):
        driver.check()
