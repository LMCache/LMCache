# SPDX-License-Identifier: Apache-2.0
"""Tests for ROCm host-memory pinning selection."""

# First Party
from lmcache.v1.platform.devices.rocm import RocmDeviceSpec
from lmcache.v1.platform.devices.rocm.pin_memory import RocmPinMemoryBackend


def test_rocm_device_spec_uses_rocm_pin_memory_backend() -> None:
    """ROCm selects its HIP-aware pin-memory backend."""
    assert RocmDeviceSpec().pin_memory_backend is RocmPinMemoryBackend
