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
from lmcache.v1.platform.cuda import CudaDeviceSpec
from lmcache.v1.platform.rocm import RocmDeviceSpec


class _ExternalDeviceSpec(DeviceSpec):
    """Valid external device specification used by discovery tests."""

    @property
    def device_type(self) -> str:
        """Return the torch-facing device type used by the test plugin."""
        return "external"

    @property
    def backend_name(self) -> str:
        """Return the explicit backend selector for the test plugin."""
        return "external"

    @property
    def torch_module_name(self) -> str:
        """Return the test torch module name."""
        return "external"


class _ExternalDeviceSpecOverride(DeviceSpec):
    """Second external specification that reuses the same backend name."""

    @property
    def device_type(self) -> str:
        """Return a different device type to prove backend-name uniqueness."""
        return "other"

    @property
    def backend_name(self) -> str:
        """Return the colliding backend name."""
        return "external"


class _SharedVendorADeviceSpec(DeviceSpec):
    """External spec that shares a device type with another backend."""

    @property
    def device_type(self) -> str:
        """Return the shared device type."""
        return "shared"

    @property
    def backend_name(self) -> str:
        """Return the first backend selector."""
        return "vendor-a"

    @property
    def torch_module_name(self) -> str:
        """Return the shared torch module name."""
        return "shared"


class _SharedVendorBDeviceSpec(DeviceSpec):
    """Second external spec for the same device type."""

    @property
    def device_type(self) -> str:
        """Return the shared device type."""
        return "shared"

    @property
    def backend_name(self) -> str:
        """Return the second backend selector."""
        return "vendor-b"

    @property
    def torch_module_name(self) -> str:
        """Return the shared torch module name."""
        return "shared"


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
    backend_registry_builder = _device_detect._build_backend_registry
    device_registry_builder = _device_detect._build_device_registry
    backend_registry_builder.cache_clear()
    device_registry_builder.cache_clear()
    yield
    backend_registry_builder.cache_clear()
    device_registry_builder.cache_clear()


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


def test_external_device_spec_is_added_to_registries(
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

    backend_registry = _device_detect._build_backend_registry()
    device_registry = _device_detect._build_device_registry()

    assert isinstance(backend_registry["external"], _ExternalDeviceSpec)
    assert device_registry["external"] == (backend_registry["external"],)
    assert _device_detect.get_device_spec("external") is backend_registry["external"]


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

    backend_registry = _device_detect._build_backend_registry()

    assert "external" not in backend_registry
    assert "wrong-name" not in backend_registry
    assert "cpu" in backend_registry


def test_plugin_import_failure_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception raised while importing one plugin is contained."""

    def fail_to_load() -> Any:
        raise ImportError("missing vendor runtime")

    _set_entry_points(
        monkeypatch,
        [_FakeEntryPoint("external", "broken_plugin:device", fail_to_load)],
    )

    backend_registry = _device_detect._build_backend_registry()

    assert "external" not in backend_registry
    assert "cpu" in backend_registry


def test_duplicate_backend_name_is_ignored_after_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends must be unique even when device types differ."""
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

    backend_registry = _device_detect._build_backend_registry()

    assert isinstance(backend_registry["external"], _ExternalDeviceSpec)


def test_same_device_type_can_register_multiple_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different backends can intentionally share one torch device type."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "vendor-a",
                "plugin_a:SharedVendorADeviceSpec",
                lambda: _SharedVendorADeviceSpec,
            ),
            _FakeEntryPoint(
                "vendor-b",
                "plugin_b:SharedVendorBDeviceSpec",
                lambda: _SharedVendorBDeviceSpec,
            ),
        ],
    )

    backend_registry = _device_detect._build_backend_registry()
    device_registry = _device_detect._build_device_registry()

    assert isinstance(backend_registry["vendor-a"], _SharedVendorADeviceSpec)
    assert isinstance(backend_registry["vendor-b"], _SharedVendorBDeviceSpec)
    assert device_registry["shared"] == (
        backend_registry["vendor-a"],
        backend_registry["vendor-b"],
    )


def test_explicit_backend_selects_requested_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend env var disambiguates shared device types explicitly."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "vendor-a",
                "plugin_a:SharedVendorADeviceSpec",
                lambda: _SharedVendorADeviceSpec,
            ),
            _FakeEntryPoint(
                "vendor-b",
                "plugin_b:SharedVendorBDeviceSpec",
                lambda: _SharedVendorBDeviceSpec,
            ),
        ],
    )
    monkeypatch.setattr(_SharedVendorADeviceSpec, "is_available", lambda self: True)
    monkeypatch.setattr(_SharedVendorBDeviceSpec, "is_available", lambda self: True)

    # Third Party
    import torch

    shared_torch_module = object()
    monkeypatch.setattr(torch, "shared", shared_torch_module, raising=False)
    monkeypatch.setenv(_device_detect.DEVICE_BACKEND_ENV_VAR, "vendor-b")

    torch_module, device_type, backend_name = _device_detect._detect_device()

    assert torch_module is shared_torch_module
    assert device_type == "shared"
    assert backend_name == "vendor-b"


def test_explicit_device_type_with_multiple_backends_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVICE_TYPE alone is insufficient when multiple backends are available."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "vendor-a",
                "plugin_a:SharedVendorADeviceSpec",
                lambda: _SharedVendorADeviceSpec,
            ),
            _FakeEntryPoint(
                "vendor-b",
                "plugin_b:SharedVendorBDeviceSpec",
                lambda: _SharedVendorBDeviceSpec,
            ),
        ],
    )
    monkeypatch.setattr(_SharedVendorADeviceSpec, "is_available", lambda self: True)
    monkeypatch.setattr(_SharedVendorBDeviceSpec, "is_available", lambda self: True)

    # Third Party
    import torch

    monkeypatch.setattr(torch, "shared", object(), raising=False)
    monkeypatch.setenv("DEVICE_TYPE", "shared")

    with pytest.raises(RuntimeError, match="LMCACHE_DEVICE_BACKEND"):
        _device_detect._detect_device()


def test_get_device_spec_uses_explicit_backend_with_shared_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared device types resolve through the explicit backend selector."""
    _set_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint(
                "vendor-a",
                "plugin_a:SharedVendorADeviceSpec",
                lambda: _SharedVendorADeviceSpec,
            ),
            _FakeEntryPoint(
                "vendor-b",
                "plugin_b:SharedVendorBDeviceSpec",
                lambda: _SharedVendorBDeviceSpec,
            ),
        ],
    )
    monkeypatch.setenv(_device_detect.DEVICE_BACKEND_ENV_VAR, "vendor-b")

    spec = _device_detect.get_device_spec("shared")

    assert isinstance(spec, _SharedVendorBDeviceSpec)


def test_external_device_participates_in_runtime_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit external device type resolves through normal detection."""
    monkeypatch.setenv("DEVICE_TYPE", "external")
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
    monkeypatch.setattr(_ExternalDeviceSpec, "is_available", lambda self: True)

    # Third Party
    import torch

    external_torch_module = object()
    monkeypatch.setattr(torch, "external", external_torch_module, raising=False)

    torch_module, device_type, backend_name = _device_detect._detect_device()

    assert torch_module is external_torch_module
    assert device_type == "external"
    assert backend_name == "external"


@pytest.mark.parametrize(
    (
        "cuda_version",
        "hip_version",
        "expected_spec_type",
        "expected_backend_name",
    ),
    [
        ("13.0", None, CudaDeviceSpec, "cuda"),
        (None, "7.2", RocmDeviceSpec, "rocm"),
    ],
)
def test_cuda_device_type_auto_selects_runtime_backend(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    hip_version: str | None,
    expected_spec_type: type[DeviceSpec],
    expected_backend_name: str,
) -> None:
    """CUDA and ROCm builds select one backend for torch's cuda device type."""
    _set_entry_points(monkeypatch, [])

    # Third Party
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)

    torch_module, device_type, backend_name = _device_detect._detect_device()

    assert torch_module is torch.cuda
    assert device_type == "cuda"
    assert backend_name == expected_backend_name
    assert isinstance(
        _device_detect._build_backend_registry()[backend_name], expected_spec_type
    )


@pytest.mark.parametrize(
    ("cuda_version", "hip_version"),
    [
        ("13.0", None),
        (None, "7.2"),
        (None, None),
    ],
)
def test_cuda_device_type_prefers_default_backend_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    hip_version: str | None,
) -> None:
    """Explicit cuda lookups stay deterministic even without a GPU."""
    _set_entry_points(monkeypatch, [])

    # Third Party
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)

    assert isinstance(_device_detect.get_device_spec("cuda"), CudaDeviceSpec)
