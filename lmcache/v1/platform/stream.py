# SPDX-License-Identifier: Apache-2.0
"""Platform-neutral stream execution helpers.

Native GPU libraries commonly need a raw stream handle, while their Python
callers need to synchronize streams and retain work until a completion event
fires. This module keeps those operations behind :class:`DeviceSpec`, so
generic call sites do not depend on CUDA, ROCm, or another runtime's stream
and event object layouts.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec


def _get_spec(device: object) -> DeviceSpec:
    """Return the registered device specification for ``device``.

    Args:
        device: A torch device object.

    Returns:
        The device's registered platform specification.

    Raises:
        RuntimeError: If the device does not expose a registered device type.
    """
    device_type = getattr(device, "type", None)
    if not isinstance(device_type, str):
        raise RuntimeError(f"Cannot resolve a platform DeviceSpec for {device!r}.")
    spec = get_device_spec(device_type)
    if spec is None:
        raise RuntimeError(
            f"No platform DeviceSpec is registered for device type {device_type!r}."
        )
    return spec


@dataclass(frozen=True)
class CompletionEvent:
    """A platform-owned stream completion event.

    The raw event is intentionally private: callers only need to ask whether
    its stream-ordered work has completed.
    """

    _spec: DeviceSpec
    _event: object

    def is_complete(self) -> bool:
        """Return whether the event's preceding stream work has completed."""
        return self._spec.is_stream_event_complete(self._event)


def current_stream(device: object) -> object:
    """Return the current stream for ``device`` through its DeviceSpec.

    Args:
        device: A torch device object.

    Returns:
        The platform's current stream object.
    """
    return _get_spec(device).current_stream(device)


def stream_handle(device: object, stream: object) -> int:
    """Return ``stream``'s native handle through the platform adapter.

    Args:
        device: A torch device object that owns ``stream``.
        stream: A stream returned by :func:`current_stream`.

    Returns:
        The native stream handle required by a stream-aware library.
    """
    return _get_spec(device).get_stream_handle(stream)


def synchronize_stream(device: object, stream: object) -> None:
    """Wait for work already queued on ``stream`` to complete.

    Args:
        device: A torch device object that owns ``stream``.
        stream: A stream returned by :func:`current_stream`.
    """
    _get_spec(device).synchronize_stream(stream)


def synchronize_device(device: object) -> None:
    """Wait for work already queued on ``device`` to complete.

    Args:
        device: A torch device object.
    """
    _get_spec(device).synchronize_device(device)


def record_completion_event(device: object, stream: object) -> CompletionEvent:
    """Record a completion event after work already queued on ``stream``.

    Args:
        device: A torch device object that owns ``stream``.
        stream: A stream returned by :func:`current_stream`.

    Returns:
        A completion event that can be polled with
        :meth:`CompletionEvent.is_complete`.
    """
    spec = _get_spec(device)
    event = spec.create_stream_event(device)
    spec.record_stream_event(event, stream)
    return CompletionEvent(spec, event)
