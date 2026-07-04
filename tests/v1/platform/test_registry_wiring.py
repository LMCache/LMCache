# SPDX-License-Identifier: Apache-2.0
"""Production wiring tests for registry-based pin-memory resolution."""

# Standard

# Third Party
import pytest

# First Party
from lmcache.v1.platform._registry import (
    _discover_base_classes,
    get_impl,
    resolve_impl,
)
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.device_ext import DeviceExt


def test_pin_memory_backend_is_discovered_from_base_package() -> None:
    """PinMemoryBackend is discoverable from ``lmcache.v1.platform.base``."""
    assert PinMemoryBackend in _discover_base_classes()


def test_get_impl_returns_cuda_pin_memory_backend() -> None:
    """Strict lookup resolves the CUDA concrete implementation."""
    assert get_impl(PinMemoryBackend, "cuda", "default") is CudaPinMemoryBackend


def test_get_impl_cpu_is_strict_and_raises() -> None:
    """Strict lookup raises when no CPU pin-memory backend exists."""
    with pytest.raises(ValueError):
        get_impl(PinMemoryBackend, "cpu", "default")


def test_resolve_impl_cpu_returns_noop_base_fallback() -> None:
    """resolve_impl falls back to PinMemoryBackend for CPU."""
    assert resolve_impl(PinMemoryBackend, "cpu", "default") is PinMemoryBackend


def test_device_ext_uses_override_backend_when_provided() -> None:
    """DeviceExt resolves CUDA pin memory through the registry."""

    class _FakeCudaDeviceInfo:
        @property
        def device_type(self) -> str:
            return "cuda"

    ext = DeviceExt(_FakeCudaDeviceInfo())  # type: ignore[arg-type]
    assert isinstance(ext._pin, CudaPinMemoryBackend)


def test_device_ext_uses_noop_backend_for_cpu_without_override() -> None:
    """DeviceExt resolves to no-op PinMemoryBackend for CPU/no override."""

    class _FakeCpuDeviceInfo:
        @property
        def device_type(self) -> str:
            return "cpu"

        @property
        def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
            return None

    ext = DeviceExt(_FakeCpuDeviceInfo())  # type: ignore[arg-type]
    assert type(ext._pin) is PinMemoryBackend
    assert ext.is_pin_supported is False
