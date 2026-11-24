# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm event IPC backend (GIL-free import of polled events)."""

# Standard
from enum import IntEnum
from typing import Any
import gc
import time

# Third Party
import pytest
import torch
import torch.multiprocessing as mp

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend
from lmcache.v1.platform.devices.rocm import RocmDeviceSpec
from lmcache.v1.platform.devices.rocm.event_ipc import (
    HipIpcEvent,
    RocmEventIPCBackend,
)
from lmcache.v1.platform.ipc_policy import is_isolated_ipc, set_isolated_ipc


class _HipError(IntEnum):
    hipSuccess = 0
    hipErrorInvalidValue = 1
    hipErrorNotReady = 600


class _FakeHip:
    """``rocm.bindings.hip`` stand-in: calls return tuples led by the error."""

    hipError_t = _HipError

    def __init__(self) -> None:
        self.open_error = _HipError.hipSuccess
        self.ready: set[int] = set()
        self.opened: list[bytes] = []
        self.queried: list[int] = []
        self.destroyed: list[int] = []
        self.devices: list[int] = []
        self.cleared = 0

    def hipSetDevice(self, index: int) -> tuple:
        self.devices.append(index)
        return (_HipError.hipSuccess,)

    def hipIpcOpenEventHandle(self, handle: bytes) -> tuple:
        if self.open_error != _HipError.hipSuccess:
            return (self.open_error, None)
        self.opened.append(bytes(handle))
        return (_HipError.hipSuccess, len(self.opened))

    def hipEventQuery(self, event: int) -> tuple:
        self.queried.append(event)
        ready = event in self.ready
        return (_HipError.hipSuccess if ready else _HipError.hipErrorNotReady,)

    def hipEventDestroy(self, event: int) -> tuple:
        self.destroyed.append(event)
        return (_HipError.hipSuccess,)

    def hipGetLastError(self) -> tuple:
        self.cleared += 1
        return (_HipError.hipSuccess,)


class _TorchEvent:
    def __init__(self, handle: bytes) -> None:
        self.handle = handle
        self.waited: list[object] = []

    def wait(self, stream: object) -> None:
        self.waited.append(stream)

    def query(self) -> bool:
        return True


class _TorchEventModule:
    """``torch.cuda`` stand-in that counts ``Event.from_ipc_handle`` calls."""

    def __init__(self) -> None:
        module = self
        self.imports: list[tuple[object, bytes]] = []

        class Event:
            @classmethod
            def from_ipc_handle(cls, device: object, handle: bytes) -> _TorchEvent:
                module.imports.append((device, handle))
                return _TorchEvent(handle)

        self.Event = Event


def _handle(tag: bytes) -> bytes:
    return tag.ljust(64, b"\0")


def _export_pending_event(handle_q: Any, release: Any) -> None:
    """Export a real ROCm IPC event and keep its producer alive."""
    try:
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        event = torch.cuda.Event(interprocess=True)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            event.record(stream)
        handle_q.put(("ok", bytes(event.ipc_handle())))
        stream.synchronize()
        release.wait(180)
    except BaseException as exc:
        handle_q.put(("error", repr(exc)))
        raise


@pytest.fixture
def hip() -> _FakeHip:
    return _FakeHip()


@pytest.fixture
def torch_events() -> _TorchEventModule:
    return _TorchEventModule()


@pytest.fixture
def backend(hip: _FakeHip, torch_events: _TorchEventModule) -> RocmEventIPCBackend:
    return RocmEventIPCBackend(
        event_module=torch_events, device_type="cuda", hip_module=hip
    )


def test_polled_import_opens_through_hip_only(
    backend: RocmEventIPCBackend, hip: _FakeHip, torch_events: _TorchEventModule
) -> None:
    event = backend.import_event_for_query(_handle(b"s"), 3)
    assert isinstance(event, HipIpcEvent)
    assert hip.opened == [_handle(b"s")]
    assert hip.devices == [3]
    assert torch_events.imports == []


def test_polled_import_accepts_a_torch_like_device(
    backend: RocmEventIPCBackend, hip: _FakeHip
) -> None:
    class _Device:
        index = 5

    event = backend.import_event_for_query(_handle(b"s"), _Device())
    assert isinstance(event, HipIpcEvent)
    assert hip.devices == [5]


def test_query_of_a_hip_event_clears_the_not_ready_error(
    backend: RocmEventIPCBackend, hip: _FakeHip
) -> None:
    event = backend.import_event_for_query(_handle(b"s"), 0)
    assert backend.query_event(event) is False
    assert hip.cleared == 1  # hipErrorNotReady must not stay on the thread
    hip.ready.add(1)
    assert backend.query_event(event) is True
    assert hip.cleared == 1
    assert hip.queried == [1, 1]


def test_open_failure_clears_the_error_and_falls_back_to_torch(
    backend: RocmEventIPCBackend, hip: _FakeHip, torch_events: _TorchEventModule
) -> None:
    hip.open_error = _HipError.hipErrorInvalidValue
    event = backend.import_event_for_query(_handle(b"s"), 0)
    assert isinstance(event, _TorchEvent)
    assert hip.cleared == 1
    assert torch_events.imports == [(0, _handle(b"s"))]
    assert backend.query_event(event) is True


@pytest.mark.parametrize("handle", [b"short", _handle(b"s") + b"x"])
def test_handle_of_another_size_never_reaches_hip(
    backend: RocmEventIPCBackend,
    hip: _FakeHip,
    torch_events: _TorchEventModule,
    handle: bytes,
) -> None:
    assert isinstance(backend.import_event_for_query(handle, 0), _TorchEvent)
    assert hip.opened == [] and hip.devices == []
    assert torch_events.imports == [(0, handle)]


def test_missing_bindings_fall_back_to_torch(torch_events: _TorchEventModule) -> None:
    backend = RocmEventIPCBackend(
        event_module=torch_events, device_type="cuda", hip_module=None
    )
    assert isinstance(backend.import_event_for_query(_handle(b"s"), 0), _TorchEvent)
    assert torch_events.imports == [(0, _handle(b"s"))]


def test_stream_wait_import_opens_the_handle_once_through_torch(
    backend: RocmEventIPCBackend, hip: _FakeHip, torch_events: _TorchEventModule
) -> None:
    """Blend / qstore path: ``import_event`` then ``wait_event``."""
    event: Any = backend.import_event(_handle(b"r"), 2)
    backend.wait_event(event, "stream")
    assert torch_events.imports == [(2, _handle(b"r"))]
    assert event.waited == ["stream"]
    assert hip.opened == [] and hip.devices == []


def test_hip_event_is_destroyed_when_released(
    backend: RocmEventIPCBackend, hip: _FakeHip
) -> None:
    event = backend.import_event_for_query(_handle(b"s"), 0)
    del event
    gc.collect()
    assert hip.destroyed == [1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires a ROCm device",
)
def test_real_rocm_query_import_opens_a_torch_ipc_handle() -> None:
    """Exercise the generated binding with PyTorch's real 64-byte handle."""
    ctx = mp.get_context("spawn")
    handle_q = ctx.Queue()
    release = ctx.Event()
    producer = ctx.Process(target=_export_pending_event, args=(handle_q, release))
    producer.start()

    imported = None
    try:
        status, payload = handle_q.get(timeout=180)
        assert status == "ok", payload

        backend = RocmEventIPCBackend(
            event_module=torch.cuda,
            device_type="cuda",
        )
        imported = backend.import_event_for_query(payload, torch.device("cuda:0"))
        assert isinstance(imported, HipIpcEvent)

        deadline = time.monotonic() + 180
        while not backend.query_event(imported):
            assert time.monotonic() < deadline, "timed out waiting for the IPC event"
            time.sleep(0.001)

        del imported
        imported = None
        gc.collect()
    finally:
        release.set()
        producer.join(timeout=60)
        if producer.is_alive():
            producer.terminate()
            producer.join(timeout=10)
        del imported

    assert producer.exitcode == 0


@pytest.fixture
def restore_isolated_ipc():
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


def test_rocm_device_spec_exposes_the_rocm_event_backend(restore_isolated_ipc) -> None:
    set_isolated_ipc(False)
    spec = RocmDeviceSpec()
    backend = spec.event_ipc_backend
    assert isinstance(backend, RocmEventIPCBackend)
    assert spec.event_ipc_backend is backend


def test_rocm_device_spec_keeps_the_cuda_backend_when_isolated(
    restore_isolated_ipc,
) -> None:
    set_isolated_ipc(True)
    backend = RocmDeviceSpec().event_ipc_backend
    assert not isinstance(backend, DefaultEventIPCBackend)
