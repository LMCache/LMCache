# SPDX-License-Identifier: Apache-2.0
"""RBLN SHM registration policy without an NPU.

The server never registers host memory, and a worker on a torch-rbln release
without host registration fails with an upgrade message. The worker's
registration round trip runs on an NPU in ``test_rbln_host_registration.py``.
These tests remove ``torch.rbln``, so any registration attempt fails them.
"""

# Standard
from types import SimpleNamespace
from uuid import uuid4
import ctypes
import importlib.metadata

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


def test_worker_registration_requires_torch_rbln_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older torch-rbln is refused by version before ``torch.rbln`` is used."""
    # torch-rbln 0.12.0 has no host_register; an empty module makes any
    # attribute lookup fail, so reaching torch.rbln would raise AttributeError.
    monkeypatch.setattr(torch, "rbln", SimpleNamespace(), raising=False)
    versions = {"torch-rbln": "0.12.0"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match=r"torch-rbln 0\.12\.1a0.*0\.12\.0"):
        RblnPinMemoryBackend().pin_memory(4096, 4096)
