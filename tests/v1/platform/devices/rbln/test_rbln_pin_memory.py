# SPDX-License-Identifier: Apache-2.0
"""RBLN host registration ownership and server pool policy."""

# Standard
from collections.abc import Iterator
from uuid import uuid4
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import config
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.devices.rbln.pin_memory import RblnPinMemoryBackend
from lmcache.v1.platform.posix_shm import shm_open_pool_as_mmap
from lmcache.v1.platform.torch_ops import mem_alloc


class FakeRbln:
    """Model live driver registrations on each logical device."""

    def __init__(self) -> None:
        self.device = 0
        self.registered: dict[tuple[int, int], int] = {}
        self.admission = True
        self.register_error = False
        self.unpin_error = False

    def current_device(self) -> int:
        """Return the caller's selected device."""
        return self.device

    def host_register(self, ptr: int, nbytes: int, device: int) -> bool:
        """Admit a live range unless the driver refuses it."""
        if self.register_error:
            raise RuntimeError("injected registration failure")
        if not self.admission:
            return False
        key = (ptr, device)
        if key in self.registered:
            raise RuntimeError("overlapping registration")
        self.registered[key] = nbytes
        return True

    def host_unregister(self, ptr: int, device: int) -> None:
        """Release a range, preserving it on an injected driver failure."""
        if self.unpin_error:
            raise RuntimeError("injected unregister failure")
        del self.registered[ptr, device]


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRbln]:
    """Provide a stand-in that retains only the driver's live ranges."""
    rbln = FakeRbln()
    monkeypatch.setattr(torch, "rbln", rbln, raising=False)
    yield rbln


def test_registration_is_owned_by_original_devices(runtime: FakeRbln) -> None:
    backend = RblnPinMemoryBackend()
    assert backend.is_pin_supported
    assert backend.pin_memory(4096, 4096, flags=2)
    assert not backend.pin_memory(4096, 4096)
    runtime.device = 1
    assert backend.pin_memory(4096, 4096)
    assert runtime.registered == {(4096, 0): 4096, (4096, 1): 4096}
    runtime.device = 2
    assert backend.unpin_memory(4096)
    assert not runtime.registered


@pytest.mark.parametrize("raises", [False, True], ids=["refused", "runtime-error"])
def test_failed_registration_stays_unpinned(runtime: FakeRbln, raises: bool) -> None:
    backend = RblnPinMemoryBackend()
    runtime.admission = False
    runtime.register_error = raises
    assert not backend.pin_memory(4096, 4096)
    assert not runtime.registered
    assert not backend.unpin_memory(4096)
    runtime.admission = True
    runtime.register_error = False
    assert backend.pin_memory(4096, 4096)


def test_failed_unpin_keeps_ownership_for_retry(runtime: FakeRbln) -> None:
    backend = RblnPinMemoryBackend()
    assert backend.pin_memory(4096, 4096)
    runtime.unpin_error = True
    assert not backend.unpin_memory(4096)
    assert runtime.registered == {(4096, 0): 4096}
    assert not backend.pin_memory(4096, 4096)
    runtime.unpin_error = False
    assert backend.unpin_memory(4096)
    assert not runtime.registered


@pytest.mark.parametrize("missing", ["host_register", "host_unregister"])
def test_both_registration_apis_are_required(
    monkeypatch: pytest.MonkeyPatch, runtime: FakeRbln, missing: str
) -> None:
    monkeypatch.setattr(runtime, missing, None)
    spec = RblnDeviceSpec()
    assert not RblnPinMemoryBackend().is_pin_supported
    assert not spec.pin_memory(4096, 4096)
    assert not runtime.registered


def test_worker_registration_with_allocator_pinning_disabled(runtime: FakeRbln) -> None:
    spec = RblnDeviceSpec()
    assert not spec.is_pin_supported
    assert spec.pin_memory_backend is RblnPinMemoryBackend
    assert spec.pin_memory(4096, 4096)
    assert runtime.registered == {(4096, 0): 4096}
    assert spec.unpin_memory(4096)
    assert not runtime.registered


def test_server_shm_pool_is_usable_without_registration(
    monkeypatch: pytest.MonkeyPatch, runtime: FakeRbln
) -> None:
    spec = RblnDeviceSpec()
    monkeypatch.setattr(mem_alloc, "current_device_spec", lambda: spec)
    ops = RblnDeviceOps()
    name = f"lmcache_test_rbln_pin_{uuid4().hex}"
    size = 4096
    ptr = ops.alloc_shm_pinned_ptr(size, name)
    try:
        with shm_open_pool_as_mmap(name, size) as worker:
            ctypes.memmove(ptr, b"cache", 5)
            assert worker[:5] == b"cache"
            assert not runtime.registered
    finally:
        ops.free_shm_pinned_ptr(ptr, size, name)
    with pytest.raises(FileNotFoundError), shm_open_pool_as_mmap(name, size):
        pytest.fail("The freed SHM pool was still accessible")
    assert spec.pin_memory(4096, 4096)
    assert spec.unpin_memory(4096)


def test_l1_config_preserves_shm_and_disables_lazy(
    monkeypatch: pytest.MonkeyPatch, runtime: FakeRbln
) -> None:
    monkeypatch.setattr(config, "current_device_spec", RblnDeviceSpec())
    cfg = config.L1MemoryManagerConfig(
        size_in_bytes=1 << 20, use_lazy=True, shm_name="rbln-test-l1"
    )
    assert not cfg.use_lazy
    assert cfg.shm_name == "rbln-test-l1"
    assert not runtime.registered
