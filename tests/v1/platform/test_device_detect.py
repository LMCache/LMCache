# SPDX-License-Identifier: Apache-2.0
"""Tests for platform device discovery and shared registry wiring."""

# Standard
from collections.abc import Iterator
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.cpu import CpuDeviceSpec
import lmcache.v1.platform as platform_pkg
import lmcache.v1.platform._device_detect as device_detect


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(
        self,
        *,
        name: str,
        value: str,
        loaded: Any | None = None,
        group: str = "lmcache.v1.device_specs",
        exc: Exception | None = None,
    ) -> None:
        self.group = group
        self.name = name
        self.value = value
        self._loaded = loaded
        self._exc = exc

    def load(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._loaded


class _FakeEntryPoints(list[_FakeEntryPoint]):
    """Collection exposing the ``select(group=...)`` API."""

    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        return [entry_point for entry_point in self if entry_point.group == group]


@pytest.fixture
def clear_device_registry_cache() -> Iterator[None]:
    """Keep the discovery cache isolated across tests."""
    device_detect._build_device_registry.cache_clear()
    try:
        yield
    finally:
        device_detect._build_device_registry.cache_clear()


def test_build_device_registry_loads_device_spec_from_entry_point(
    clear_device_registry_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed wheels can register a device backend through entry points."""

    class PluginDeviceOps(DeviceOps):
        device_type = "entrypoint-device"

    class PluginDeviceSpec(DeviceSpec):
        @property
        def device_type(self) -> str:
            return "entrypoint-device"

        @property
        def torch_module_name(self) -> str:
            return "entrypoint_device"

        @property
        def ops_cls(self) -> type[DeviceOps]:
            return PluginDeviceOps

    monkeypatch.setattr(
        device_detect.importlib_metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [
                _FakeEntryPoint(
                    name="entrypoint-device",
                    value="fake.plugin:PluginDeviceSpec",
                    loaded=PluginDeviceSpec,
                )
            ]
        ),
    )

    registry = device_detect._build_device_registry()

    spec = registry["entrypoint-device"]
    assert isinstance(spec, PluginDeviceSpec)
    assert type(spec.get_ops()) is PluginDeviceOps


def test_build_device_registry_skips_invalid_entry_point_target(
    clear_device_registry_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery warns and skips entry points that do not resolve to DeviceSpec."""
    warnings: list[str] = []
    monkeypatch.setattr(
        device_detect.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        device_detect.importlib_metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [
                _FakeEntryPoint(
                    name="broken-device",
                    value="fake.plugin:not_a_spec",
                    loaded=object(),
                )
            ]
        ),
    )

    registry = device_detect._build_device_registry()

    assert "broken-device" not in registry
    assert any(
        "must resolve to a DeviceSpec subclass or instance" in warning
        for warning in warnings
    )


def test_build_device_registry_preserves_existing_device_type_on_collision(
    clear_device_registry_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External wheels must not override an existing built-in device type."""

    class DuplicateCpuDeviceSpec(DeviceSpec):
        @property
        def device_type(self) -> str:
            return "cpu"

        @property
        def torch_module_name(self) -> str:
            return "cpu"

    monkeypatch.setattr(
        device_detect.importlib_metadata,
        "entry_points",
        lambda: _FakeEntryPoints(
            [
                _FakeEntryPoint(
                    name="cpu",
                    value="fake.plugin:DuplicateCpuDeviceSpec",
                    loaded=DuplicateCpuDeviceSpec,
                )
            ]
        ),
    )

    registry = device_detect._build_device_registry()

    assert isinstance(registry["cpu"], CpuDeviceSpec)


def test_get_device_spec_reads_shared_platform_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_device_detect.get_device_spec`` resolves through the shared registry."""

    class SharedPluginDeviceSpec(DeviceSpec):
        @property
        def device_type(self) -> str:
            return "shared-plugin"

    spec = SharedPluginDeviceSpec()
    monkeypatch.setattr(
        platform_pkg,
        "_DEVICE_REGISTRY",
        {**platform_pkg._DEVICE_REGISTRY, spec.device_type: spec},
    )

    assert device_detect.get_device_spec(spec.device_type) is spec
