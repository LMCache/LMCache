# SPDX-License-Identifier: Apache-2.0
"""Tests for platform-neutral stream execution helpers."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, call

# Third Party
import pytest

# First Party
from lmcache.v1.platform import stream
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec
from lmcache.v1.platform.devices.musa import MusaDeviceSpec


@pytest.fixture(autouse=True)
def reset_stream_spec_cache() -> Iterator[None]:
    """Keep mocked device specifications isolated between tests."""
    stream._get_spec_for_type.cache_clear()
    yield
    stream._get_spec_for_type.cache_clear()


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


def test_stream_operations_dispatch_through_device_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic callers only use DeviceSpec stream capabilities."""
    spec = _FakeDeviceSpec()
    device = SimpleNamespace(type="example")
    lookup = Mock(return_value=spec)
    monkeypatch.setattr(stream, "get_device_spec", lookup)

    current = stream.current_stream(device)
    assert current == "stream"
    assert stream.stream_handle(device, current) == 17
    stream.synchronize_stream(device, current)
    stream.synchronize_device(device)

    completion = stream.record_completion_event(device, current)
    assert completion.is_complete() is False
    spec.event_complete = True
    assert completion.is_complete() is True

    lookup.assert_called_once_with("example")
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


def test_stream_specs_are_cached_by_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device indices share a spec, but different device types stay separate."""
    specs = {"first": _FakeDeviceSpec(), "second": _FakeDeviceSpec()}
    lookup = Mock(side_effect=specs.__getitem__)
    monkeypatch.setattr(stream, "get_device_spec", lookup)
    first_device = SimpleNamespace(type="first", index=0)
    next_device = SimpleNamespace(type="first", index=1)
    other_device = SimpleNamespace(type="second", index=0)

    stream.current_stream(first_device)
    stream.current_stream(next_device)
    stream.current_stream(other_device)
    stream.current_stream(first_device)

    assert lookup.call_args_list == [call("first"), call("second")]
    assert specs["first"].calls == [
        ("current_stream", first_device),
        ("current_stream", next_device),
        ("current_stream", first_device),
    ]
    assert specs["second"].calls == [("current_stream", other_device)]


def test_failed_spec_lookup_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsuccessful resolution must not poison subsequent stream calls."""
    spec = _FakeDeviceSpec()
    lookup = Mock(side_effect=[None, spec])
    monkeypatch.setattr(stream, "get_device_spec", lookup)
    device = SimpleNamespace(type="example")

    with pytest.raises(RuntimeError, match="No platform DeviceSpec"):
        stream.current_stream(device)
    assert stream.current_stream(device) == "stream"
    assert lookup.call_args_list == [call("example"), call("example")]


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
