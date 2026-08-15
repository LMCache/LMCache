# SPDX-License-Identifier: Apache-2.0
"""Tests for wheel-installed device plugins discovered by entry point."""

# Standard
from collections.abc import Callable
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _device_detect
from lmcache.v1.platform.base.device_spec import DeviceSpec


class _ExternalDeviceSpec(DeviceSpec):
    """Valid external device specification used by discovery tests."""

    @property
    def device_type(self) -> str:
        """Return the entry-point name used by the test plugin."""
        return "external"

    @property
    def torch_module_name(self) -> str:
        """Return the test torch module name."""
        return "external"


class _CpuOverrideDeviceSpec(DeviceSpec):
    """External specification that attempts to replace the built-in CPU."""

    @property
    def device_type(self) -> str:
        """Return the colliding built-in device type."""
        return "cpu"


class _ExternalDeviceSpecOverride(DeviceSpec):
    """Second external specification for deterministic duplicate handling."""

    @property
    def device_type(self) -> str:
        """Return the colliding external device type."""
        return "external"


class _FakeEntryPoint:
    """Small importlib.metadata.EntryPoint stand-in."""

    def __init__(self, name: str, value: str, loader: Callable[[], Any]) -> None:
        self.name = name
        self.value = value
        self._loader = loader

    def load(self) -> Any:
        """Return or raise from the configured plugin loader."""
        return self._loader()


@pytest.fixture(autouse=True)
def reset_device_registry_cache() -> Any:
    """Keep the process-wide registry cache isolated between tests."""
    registry_builder = _device_detect._build_device_registry
    registry_builder.cache_clear()
    yield
    registry_builder.cache_clear()


def _set_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    entry_points: list[_FakeEntryPoint],
) -> None:
    """Replace installed entry-point discovery with a deterministic list."""

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        assert group == _device_detect.DEVICE_PLUGIN_ENTRY_POINT_GROUP
        return entry_points

    monkeypatch.setattr(
        _device_detect.importlib.metadata,
        "entry_points",
        fake_entry_points,
    )


def test_external_device_spec_is_added_to_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid wheel entry point contributes its DeviceSpec at runtime."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "external",
                "lmcache_external:ExternalDeviceSpec",
                lambda: _ExternalDeviceSpec,
            )
        ],
    )

    registry = _device_detect._build_device_registry()

    assert isinstance(registry["external"], _ExternalDeviceSpec)
    assert _device_detect.get_device_spec("external") is registry["external"]


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("external", lambda: object),
        ("wrong-name", lambda: _ExternalDeviceSpec),
    ],
)
def test_invalid_device_plugin_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    loader: Callable[[], Any],
) -> None:
    """Malformed entry points cannot prevent built-in registry creation."""
    _set_entry_points(
        monkeypatch,
        [_FakeEntryPoint(name, "broken_plugin:device", loader)],
    )

    registry = _device_detect._build_device_registry()

    assert "external" not in registry
    assert "wrong-name" not in registry
    assert "cpu" in registry


def test_plugin_import_failure_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception raised while importing one plugin is contained."""

    def fail_to_load() -> Any:
        raise ImportError("missing vendor runtime")

    _set_entry_points(
        monkeypatch,
        [_FakeEntryPoint("external", "broken_plugin:device", fail_to_load)],
    )

    registry = _device_detect._build_device_registry()

    assert "external" not in registry
    assert "cpu" in registry


def test_external_device_overrides_builtin_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed package can override an LMCache built-in backend."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "cpu",
                "lmcache_external:CpuOverrideDeviceSpec",
                lambda: _CpuOverrideDeviceSpec,
            )
        ],
    )

    registry = _device_detect._build_device_registry()

    assert isinstance(registry["cpu"], _CpuOverrideDeviceSpec)


def test_first_external_device_wins_duplicate_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate external wheels resolve deterministically to the first match."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "external",
                "plugin_a:ExternalDeviceSpec",
                lambda: _ExternalDeviceSpec,
            ),
            _FakeEntryPoint(
                "external",
                "plugin_b:ExternalDeviceSpecOverride",
                lambda: _ExternalDeviceSpecOverride,
            ),
        ],
    )

    registry = _device_detect._build_device_registry()

    assert isinstance(registry["external"], _ExternalDeviceSpec)


def test_external_device_participates_in_runtime_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered external spec is selected by the normal detector."""
    spec = _ExternalDeviceSpec()
    monkeypatch.setattr(spec, "is_available", lambda: True)
    monkeypatch.setattr(
        _device_detect,
        "_build_device_registry",
        lambda: {"external": spec},
    )

    # Third Party
    import torch

    external_torch_module = object()
    monkeypatch.setattr(torch, "external", external_torch_module, raising=False)
    monkeypatch.setenv("DEVICE_TYPE", "external")

    torch_module, device_type = _device_detect._detect_device()

    assert torch_module is external_torch_module
    assert device_type == "external"
