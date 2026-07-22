# SPDX-License-Identifier: Apache-2.0

"""Unit tests for UUID->ordinal resolution in
``lmcache.v1.platform.base.ipc_wrapper.DeviceIPCWrapper``.

The focus is ``_get_device_index_from_uuid``'s error behavior, in
particular the device-incapable-host path (issue #4186): a CPU-only
``lmcache server`` that receives a CUDA worker's IPC handle must fail
with actionable guidance toward ``engine_driven`` transfer mode instead
of the generic "device not found" message. These tests stay
platform-agnostic -- they patch the ``torch_dev`` abstraction and the
device-discovery hook rather than touching real accelerators.
"""

# Standard
from typing import Iterator

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base import ipc_wrapper
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


@pytest.fixture
def restore_device_mapping() -> Iterator[None]:
    """Snapshot and restore the shared class-level discovery cache.

    ``_discovered_device_mapping`` is a class attribute shared across all
    wrappers, so a test that seeds it must not leak into other tests.
    """
    saved = dict(DeviceIPCWrapper._discovered_device_mapping)
    try:
        yield
    finally:
        DeviceIPCWrapper._discovered_device_mapping = saved


def test_cpu_only_host_raises_actionable_engine_driven_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device-incapable process points the operator at engine_driven mode."""
    monkeypatch.setattr(ipc_wrapper.torch_dev, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as excinfo:
        DeviceIPCWrapper._get_device_index_from_uuid("some-cuda-uuid")

    message = str(excinfo.value)
    # The actionable guidance, not the misleading generic message.
    assert "engine_driven" in message
    assert "some-cuda-uuid" in message
    assert "no accelerator devices" in message
    assert "not found in the discovered" not in message


def test_unknown_uuid_on_capable_host_raises_generic_error(
    monkeypatch: pytest.MonkeyPatch,
    restore_device_mapping: None,
) -> None:
    """On a device-capable host, an unknown UUID keeps the original error."""
    monkeypatch.setattr(ipc_wrapper.torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(
        DeviceIPCWrapper, "_discover_devices", classmethod(lambda cls: None)
    )
    DeviceIPCWrapper._discovered_device_mapping = {}

    with pytest.raises(RuntimeError) as excinfo:
        DeviceIPCWrapper._get_device_index_from_uuid("missing-uuid")

    message = str(excinfo.value)
    assert "not found in the discovered" in message
    assert "engine_driven" not in message


def test_known_uuid_returns_ordinal(
    monkeypatch: pytest.MonkeyPatch,
    restore_device_mapping: None,
) -> None:
    """A discovered UUID resolves to its physical ordinal."""
    monkeypatch.setattr(ipc_wrapper.torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(
        DeviceIPCWrapper, "_discover_devices", classmethod(lambda cls: None)
    )
    DeviceIPCWrapper._discovered_device_mapping = {"uuid-a": 0, "uuid-b": 3}

    assert DeviceIPCWrapper._get_device_index_from_uuid("uuid-b") == 3
