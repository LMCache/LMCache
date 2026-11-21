# SPDX-License-Identifier: Apache-2.0
"""Byte-exact transfers between an NPU and a registered SHM mapping."""

# Standard
from uuid import uuid4

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_open_pool_as_mmap,
    shm_unlink,
)

pytestmark = [
    pytest.mark.rbln,
    pytest.mark.skipif(
        not RblnDeviceSpec().is_available(), reason="RBLN NPU is required"
    ),
]


def test_registered_shm_round_trip() -> None:
    spec = RblnDeviceSpec()
    nbytes = 2 << 20
    name = f"lmcache_test_registered_{uuid4().hex}"
    server_addr = shm_create_readwrite(name, 2 * nbytes)
    try:
        with shm_open_pool_as_mmap(name, 2 * nbytes) as mapping:
            raw = torch.frombuffer(mapping, dtype=torch.uint8)
            offset = (-raw.data_ptr()) % nbytes
            host = raw[offset : offset + nbytes].view(torch.bfloat16)
            del raw
            ptr = host.data_ptr()
            device = torch.rbln.current_device()
            assert spec.pin_memory(ptr, nbytes)
            try:
                bits = torch.arange(host.numel(), dtype=torch.int32).to(torch.int16)
                host.view(torch.int16).copy_(bits)
                target = torch.empty_like(host, device=f"rbln:{device}")
                torch.rbln.bind_device_memory(target)
                target.copy_(host)
                torch.rbln.synchronize()
                host.zero_()
                host.copy_(target)
                torch.rbln.synchronize()
                assert torch.equal(host.view(torch.int16), bits)
            finally:
                assert spec.unpin_memory(ptr)
                del host
    finally:
        shm_munmap(server_addr, 2 * nbytes)
        shm_unlink(name)
