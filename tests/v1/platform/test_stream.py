# SPDX-License-Identifier: Apache-2.0
"""Tests for platform-neutral stream execution helpers."""

# Standard
from types import SimpleNamespace

# First Party
from lmcache.v1.platform import stream
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec
from lmcache.v1.platform.devices.musa import MusaDeviceSpec


class _FakeDeviceSpec:
    """Records calls made through the stream abstraction."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.event_complete = False

    def current_stream(self, device: object) -> object:
        self.calls.append(("current_stream", device))
        return "stream"

    def get_stream_handle(self, value: object) -> int:
        self.calls.append(("get_stream_handle", value))
        return 17

    def synchronize_stream(self, value: object) -> None:
        self.calls.append(("synchronize_stream", value))

    def synchronize_device(self, device: object) -> None:
        self.calls.append(("synchronize_device", device))

    def create_stream_event(self, device: object) -> object:
        self.calls.append(("create_stream_event", device))
        return "event"

    def record_stream_event(self, event: object, value: object) -> None:
        self.calls.append(("record_stream_event", event, value))

    def is_stream_event_complete(self, event: object) -> bool:
        self.calls.append(("is_stream_event_complete", event))
        return self.event_complete


def test_stream_operations_dispatch_through_device_spec(monkeypatch) -> None:
    """Generic callers only use DeviceSpec stream capabilities."""
    spec = _FakeDeviceSpec()
    device = SimpleNamespace(type="example")
    monkeypatch.setattr(stream, "get_device_spec", lambda device_type: spec)

    current = stream.current_stream(device)
    assert current == "stream"
    assert stream.stream_handle(device, current) == 17
    stream.synchronize_stream(device, current)
    stream.synchronize_device(device)

    completion = stream.record_completion_event(device, current)
    assert completion.is_complete() is False
    spec.event_complete = True
    assert completion.is_complete() is True

    assert spec.calls == [
        ("current_stream", device),
        ("get_stream_handle", "stream"),
        ("synchronize_stream", "stream"),
        ("synchronize_device", device),
        ("create_stream_event", device),
        ("record_stream_event", "event", "stream"),
        ("is_stream_event_complete", "event"),
        ("is_stream_event_complete", "event"),
    ]


def test_unknown_device_type_is_rejected() -> None:
    """A stream operation cannot silently use the wrong platform backend."""
    device = SimpleNamespace(type="unknown")
    try:
        stream.current_stream(device)
    except RuntimeError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown device type must not resolve a stream adapter")


def test_accelerator_specs_own_native_stream_handle_layouts() -> None:
    """The generic stream facade does not need accelerator-specific fields."""
    assert CudaDeviceSpec().get_stream_handle(SimpleNamespace(cuda_stream=17)) == 17
    assert MusaDeviceSpec().get_stream_handle(SimpleNamespace(musa_stream=23)) == 23
