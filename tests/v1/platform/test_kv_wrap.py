# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for per-layer KV-cache wrapping on the worker side."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import os
import select
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import kv_wrap
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.fd_channel import FdChannelServer

pytestmark = pytest.mark.no_shared_allocator


class _RecordingFactory:
    def __init__(self) -> None:
        self.wrapped: list[Any] = []

    def __call__(self, value: Any) -> Any:
        self.wrapped.append(value)
        return f"wrapper-{len(self.wrapped)}"


def test_wrap_one_kv_cache_dispatches_on_value_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingFactory()
    monkeypatch.setattr(kv_wrap, "resolve_kv_wrapper_factory", lambda _: factory)

    tensor = torch.zeros(2)
    assert kv_wrap.wrap_one_kv_cache(tensor) == "wrapper-1"
    assert factory.wrapped == [tensor]


def test_wrap_one_kv_cache_dispatches_sequence_on_first_plane_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-layer plane sequence dispatches on its first tensor's device."""
    factory = _RecordingFactory()
    seen_types: list[str] = []
    orig = factory

    def recording(value: Any) -> Any:
        return orig(value)

    def fake_resolve(device_type: str) -> Any:
        seen_types.append(device_type)
        return recording

    monkeypatch.setattr(kv_wrap, "resolve_kv_wrapper_factory", fake_resolve)

    k, v = torch.zeros(2), torch.zeros(3)
    planes = (k, v)
    assert kv_wrap.wrap_one_kv_cache(planes) == "wrapper-1"

    assert seen_types == ["cpu"]
    # The whole sequence is handed to the device factory as one value.
    assert factory.wrapped == [(k, v)]


def test_wrap_kv_caches_wraps_one_wrapper_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingFactory()
    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", factory)
    k0, v0 = torch.zeros(2), torch.zeros(3)
    single = torch.zeros(4)

    wrappers = kv_wrap.wrap_kv_caches({"layer.0": (k0, v0), "layer.1": single})

    # One wrapper per registered layer; the tuple is not flattened.
    assert factory.wrapped == [(k0, v0), single]
    assert wrappers == ["wrapper-1", "wrapper-2"]


def test_wrap_kv_caches_releases_partial_wrappers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unlinked: list[str] = []

    def _release(ws: list[Any]) -> None:
        for w in ws:
            name = getattr(w, "shm_name", None)
            if isinstance(name, str):
                unlinked.append(name)

    class _ShmWrapper:
        shm_name = "seg-1"

    class _FailingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, value: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return _ShmWrapper()
            raise RuntimeError("boom")

    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", _FailingFactory())
    monkeypatch.setattr(kv_wrap, "_release_partial_kv_wrappers", _release)

    with pytest.raises(RuntimeError, match="boom"):
        kv_wrap.wrap_kv_caches({"a": torch.zeros(1), "b": torch.zeros(1)})

    assert unlinked == ["seg-1"]


class _FdWrapper(DeviceIPCWrapper):
    """Exporter-side wrapper whose handle is a process-local fd."""

    def __init__(self, fd: int) -> None:
        self.export_id = uuid.uuid4().bytes
        self.fd: int | None = fd

    def fd_payload(self) -> tuple[bytes, int] | None:
        return None if self.fd is None else (self.export_id, self.fd)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@pytest.fixture
def fd_channel(tmp_path: Path) -> Iterator[FdChannelServer]:
    server = FdChannelServer(str(tmp_path / "fd.sock"))
    yield server
    server.close()


def test_fds_stay_claimable_for_the_registration(
    fd_channel: FdChannelServer,
) -> None:
    """The server can claim every wrapper fd inside the context, the worker's
    copies are already closed, and unclaimed fds go once it exits."""
    claimed_read, claimed_write = os.pipe()
    unclaimed_read, unclaimed_write = os.pipe()
    claimed, unclaimed = _FdWrapper(claimed_write), _FdWrapper(unclaimed_write)
    try:
        with kv_wrap.kv_wrapper_fds_delivered(
            [claimed, unclaimed], fd_channel.path, timeout=10
        ):
            assert claimed.fd_payload() is None
            assert unclaimed.fd_payload() is None
            os.close(fd_channel.take(claimed.export_id))
        ready, _, _ = select.select([unclaimed_read], [], [], 10)
        assert ready and os.read(unclaimed_read, 1) == b""
        assert os.read(claimed_read, 1) == b""
    finally:
        os.close(claimed_read)
        os.close(unclaimed_read)


@pytest.mark.parametrize("path", ["", "unserved.sock"])
def test_wrappers_without_fds_need_no_channel(tmp_path: Path, path: str) -> None:
    """Without fds nothing connects, so no path or serving channel is needed."""
    channel_path = str(tmp_path / path) if path else ""
    with kv_wrap.kv_wrapper_fds_delivered(
        [DeviceIPCWrapper()], channel_path, timeout=1
    ):
        pass


def test_fd_wrappers_without_a_channel_path_fail_and_close_their_fds() -> None:
    read_fd, write_fd = os.pipe()
    wrapper = _FdWrapper(write_fd)
    try:
        with pytest.raises(RuntimeError, match="no channel path"):
            with kv_wrap.kv_wrapper_fds_delivered([wrapper], "", timeout=1):
                pytest.fail("entered without delivering")
        assert wrapper.fd_payload() is None
        assert os.read(read_fd, 1) == b""
    finally:
        os.close(read_fd)
