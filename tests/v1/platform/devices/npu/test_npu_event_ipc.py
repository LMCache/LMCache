# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the NPU event IPC backend (no Ascend hardware)."""

# Standard
from contextlib import contextmanager
from typing import Any, Iterator
import gc
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.devices.npu import NpuDeviceSpec, event_ipc
from lmcache.v1.platform.devices.npu.event_ipc import NpuEventIPCBackend

pytestmark = pytest.mark.no_shared_allocator


class _FakeEvent:
    """torch.npu-style Event with the CUDA interprocess ABI."""

    def __init__(self, interprocess: bool = False) -> None:
        if not interprocess:
            raise ValueError("fake events must be interprocess")
        self.calls: list[tuple[Any, ...]] = []

    @classmethod
    def from_ipc_handle(cls, device: Any, handle: bytes) -> "_FakeEvent":
        event = cls(interprocess=True)
        event.calls.append(("from_ipc_handle", device, handle))
        return event

    def ipc_handle(self) -> bytes:
        self.calls.append(("ipc_handle",))
        return b"npu-handle"

    def record(self, stream: Any = None) -> None:
        self.calls.append(("record", stream))

    def wait(self, stream: Any = None) -> None:
        self.calls.append(("wait", stream))

    def query(self) -> bool:
        self.calls.append(("query",))
        return True

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))


class _FakeNpuModule:
    Event = _FakeEvent
    # Mimics torch.npu's thread-local current device; "ambient" stands in for
    # whatever device the calling thread happened to be on.
    current_device: Any = "ambient"

    @contextmanager
    def device(self, device: Any) -> Iterator[None]:
        """Mimic ``torch.npu.device``: pin ``current_device`` for the block."""
        previous = _FakeNpuModule.current_device
        _FakeNpuModule.current_device = device
        try:
            yield
        finally:
            _FakeNpuModule.current_device = previous


class _DevicePinningEvent(_FakeEvent):
    """Records the module's current device at ``ipc_handle()`` time."""

    seen_device: Any = "never-exported"

    def ipc_handle(self) -> bytes:
        _DevicePinningEvent.seen_device = _FakeNpuModule.current_device
        return super().ipc_handle()


class _AbiLessModule:
    class Event:  # noqa: N801 - mirrors torch module surface
        pass


def _device() -> Any:
    class _Device:
        type = "npu"

    return _Device()


def test_backend_satisfies_protocol() -> None:
    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    assert isinstance(backend, EventIPCBackend)
    assert backend.device_type == "npu"


def test_check_event_support_fails_closed_without_abi() -> None:
    backend = NpuEventIPCBackend(event_module=_AbiLessModule())
    with pytest.raises(RuntimeError, match="interprocess"):
        backend.check_event_support(_device())


def test_check_event_support_passes_with_abi() -> None:
    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    backend.check_event_support(_device())


def test_check_event_support_rejects_runtime_without_ipc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Python IPC signature does not imply native runtime support."""
    attempted: list[tuple[bool, object]] = []
    native_error = RuntimeError(
        "Parameter interprocess is not supported, please upgrade the "
        "HDK(driver) or CANN package."
    )

    class _UnsupportedEvent(_FakeEvent):
        def __init__(self, interprocess: bool = False) -> None:
            attempted.append((interprocess, _FakeNpuModule.current_device))
            raise native_error

    module = _FakeNpuModule()
    monkeypatch.setattr(module, "Event", _UnsupportedEvent)
    backend = NpuEventIPCBackend(event_module=module)

    with pytest.raises(RuntimeError, match="npu:3") as raised:
        backend.check_event_support("npu:3")

    assert raised.value.__cause__ is native_error
    assert str(native_error) in str(raised.value)
    assert attempted == [(True, "npu:3")]
    assert _FakeNpuModule.current_device == "ambient"


@pytest.mark.parametrize("device", [3, "npu:3", _device()])
def test_check_event_support_probes_device_and_releases_event(
    monkeypatch: pytest.MonkeyPatch, device: object
) -> None:
    """The temporary event is released without recording or exporting it."""
    seen_devices: list[object] = []
    events: list[weakref.ReferenceType[_FakeEvent]] = []
    event_calls: list[list[tuple[object, ...]]] = []

    class _TrackedEvent(_FakeEvent):
        def __init__(self, interprocess: bool = False) -> None:
            super().__init__(interprocess=interprocess)
            seen_devices.append(_FakeNpuModule.current_device)
            events.append(weakref.ref(self))
            event_calls.append(self.calls)

    module = _FakeNpuModule()
    monkeypatch.setattr(module, "Event", _TrackedEvent)
    backend = NpuEventIPCBackend(event_module=module)

    backend.check_event_support(device)
    gc.collect()

    assert seen_devices == [device]
    assert event_calls == [[]]
    assert events[0]() is None
    assert _FakeNpuModule.current_device == "ambient"


def test_check_event_support_retries_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed attempt must not suppress validation on a later call."""
    attempts: list[object] = []

    class _RetryEvent(_FakeEvent):
        def __init__(self, interprocess: bool = False) -> None:
            attempts.append(_FakeNpuModule.current_device)
            if len(attempts) == 1:
                raise RuntimeError("runtime not ready")
            super().__init__(interprocess=interprocess)

    module = _FakeNpuModule()
    monkeypatch.setattr(module, "Event", _RetryEvent)
    backend = NpuEventIPCBackend(event_module=module)

    with pytest.raises(RuntimeError, match="runtime not ready"):
        backend.check_event_support("npu:2")
    backend.check_event_support("npu:2")

    assert attempts == ["npu:2", "npu:2"]
    assert _FakeNpuModule.current_device == "ambient"


def test_check_event_support_validates_each_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success on one device must not hide another device's runtime error."""
    attempts: list[object] = []

    class _DeviceDependentEvent(_FakeEvent):
        def __init__(self, interprocess: bool = False) -> None:
            device = _FakeNpuModule.current_device
            attempts.append(device)
            if device == "npu:3":
                raise RuntimeError("IPC unavailable on this device")
            super().__init__(interprocess=interprocess)

    module = _FakeNpuModule()
    monkeypatch.setattr(module, "Event", _DeviceDependentEvent)
    backend = NpuEventIPCBackend(event_module=module)

    backend.check_event_support("npu:2")
    with pytest.raises(RuntimeError, match="IPC unavailable on this device"):
        backend.check_event_support("npu:3")

    assert attempts == ["npu:2", "npu:3"]
    assert _FakeNpuModule.current_device == "ambient"


def test_backend_creates_exports_and_imports() -> None:
    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    event = backend.create_event(_device())
    handle = backend.export_event(event, _device())
    assert handle == b"npu-handle"
    imported = backend.import_event(handle, _device())
    assert isinstance(imported, _FakeEvent)


def test_event_operations_delegate_to_torch_npu() -> None:
    """NPU operations use the torch_npu interprocess Event API."""
    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    device = _device()

    backend.check_event_support(device)
    event = backend.create_event(device)
    assert isinstance(event, _FakeEvent)
    assert backend.export_event(event, device) == b"npu-handle"
    remote = backend.import_event(b"npu-handle", device)
    assert isinstance(remote, _FakeEvent)
    backend.record_event(event, "STREAM")
    backend.wait_event(remote, "STREAM")
    assert backend.query_event(remote) is True
    backend.synchronize_event(remote, device)

    assert ("record", "STREAM") in event.calls
    assert ("wait", "STREAM") in remote.calls
    assert ("query",) in remote.calls
    assert ("synchronize",) in remote.calls
    assert ("from_ipc_handle", device, b"npu-handle") in remote.calls


def test_torch_npu_module_raises_without_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(torch, "npu", raising=False)
    with pytest.raises(RuntimeError, match="torch_npu"):
        event_ipc._torch_npu_module()


def test_device_spec_exposes_cached_event_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event_ipc, "_torch_npu_module", lambda: _FakeNpuModule())
    spec = NpuDeviceSpec()
    first = spec.event_ipc_backend
    assert first.device_type == "npu"
    assert spec.event_ipc_backend is first


def test_export_serializes_under_the_requested_device() -> None:
    """CANN derives an interprocess handle from the thread's current device,
    so export_event must pin ``device`` current for the ipc_handle() call
    regardless of the caller's ambient device."""
    # First Party
    from lmcache.v1.platform.devices.npu.event_ipc import NpuEventIPCBackend

    class _PinningModule(_FakeNpuModule):
        Event = _DevicePinningEvent

    backend = NpuEventIPCBackend(event_module=_PinningModule())
    event = backend.create_event(_device())
    handle = backend.export_event(event, "npu:3")

    assert handle == b"npu-handle"
    assert _DevicePinningEvent.seen_device == "npu:3"
    # The caller's ambient device is restored after the export.
    assert _FakeNpuModule.current_device == "ambient"


def test_export_pins_source_events() -> None:
    """Source-event liveness pinning is inherited from the default backend."""
    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    events = [backend.create_event(_device()) for _ in range(3)]
    for event in events:
        backend.export_event(event, _device())

    assert list(backend._exported_events) == events
