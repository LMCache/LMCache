# SPDX-License-Identifier: Apache-2.0
"""Platform-neutral stream and completion-event helpers backed by DeviceSpec."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from functools import lru_cache

# First Party
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec


def _get_spec(device: object) -> DeviceSpec:
    """Resolve a torch device's specification, raising on unknown device types."""
    device_type = getattr(device, "type", None)
    if not isinstance(device_type, str):
        raise RuntimeError(f"Cannot resolve a platform DeviceSpec for {device!r}.")
    return _get_spec_for_type(device_type)


@lru_cache(maxsize=None)
def _get_spec_for_type(device_type: str) -> DeviceSpec:
    """Resolve each device type once, with backend configuration set at startup."""
    spec = get_device_spec(device_type)
    if spec is None:
        raise RuntimeError(
            f"No platform DeviceSpec is registered for device type {device_type!r}."
        )
    return spec


@dataclass(frozen=True)
class CompletionEvent:
    """A pollable completion event wrapping a platform-specific event."""

    _spec: DeviceSpec
    _event: object

    def is_complete(self) -> bool:
        """Return whether the event's preceding stream work has completed."""
        return self._spec.is_stream_event_complete(self._event)


def current_stream(device: object) -> object:
    """Return the current stream for ``device`` through its DeviceSpec."""
    return _get_spec(device).current_stream(device)


def stream_handle(device: object, stream: object) -> int:
    """Return the native handle for ``stream`` on ``device``."""
    return _get_spec(device).get_stream_handle(stream)


def synchronize_stream(device: object, stream: object) -> None:
    """Wait for queued work on ``stream`` owned by ``device`` to complete."""
    _get_spec(device).synchronize_stream(stream)


def synchronize_device(device: object) -> None:
    """Wait for work already queued on ``device`` to complete."""
    _get_spec(device).synchronize_device(device)


def record_completion_event(device: object, stream: object) -> CompletionEvent:
    """Record a pollable completion event after work queued on ``stream``."""
    spec = _get_spec(device)
    event = spec.create_stream_event(device)
    spec.record_stream_event(event, stream)
    return CompletionEvent(spec, event)
