# SPDX-License-Identifier: Apache-2.0
"""RBLN server SHM pool policy: the server never registers host memory.

The worker's registration round trip runs on an NPU in
``test_rbln_host_registration.py``. These tests remove ``torch.rbln``
entirely, so any registration attempt on the server path fails them.
"""

# Standard
from uuid import uuid4
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import config
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.posix_shm import shm_open_pool_as_mmap
from lmcache.v1.platform.torch_ops import mem_alloc


@pytest.fixture(autouse=True)
def no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ``torch.rbln``: the server path must not touch a device."""
    monkeypatch.setattr(torch, "rbln", None, raising=False)


def test_server_shm_pool_is_usable_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server creates, shares and frees its SHM pool without an NPU."""
    monkeypatch.setattr(mem_alloc, "current_device_spec", lambda: RblnDeviceSpec())
    ops = RblnDeviceOps()
    name = f"lmcache_test_rbln_pin_{uuid4().hex}"
    size = 4096
    ptr = ops.alloc_shm_pinned_ptr(size, name)
    try:
        with shm_open_pool_as_mmap(name, size) as worker:
            ctypes.memmove(ptr, b"cache", 5)
            assert worker[:5] == b"cache"
    finally:
        ops.free_shm_pinned_ptr(ptr, size, name)
    with pytest.raises(FileNotFoundError), shm_open_pool_as_mmap(name, size):
        pytest.fail("The freed SHM pool was still accessible")


def test_l1_config_preserves_shm_and_disables_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RBLN declines allocator-side pinning, so L1 stays eager on named SHM."""
    monkeypatch.setattr(config, "current_device_spec", RblnDeviceSpec())
    cfg = config.L1MemoryManagerConfig(
        size_in_bytes=1 << 20, use_lazy=True, shm_name="rbln-test-l1"
    )
    assert not cfg.use_lazy
    assert cfg.shm_name == "rbln-test-l1"
