# SPDX-License-Identifier: Apache-2.0
"""Regression tests for gpu_ops against LazyMemoryAllocator.

On AMD/ROCm, the previous C++ implementation of ``lmcache_memcpy_async``
used an unqualified ``min(size_t, size_t)`` which HIP could resolve to an
``int`` overload, silently truncating the arguments and producing an
impossible-size ``hipMemcpyAsync`` once the LazyMemoryAllocator handed
out addresses at or past the 2 GB virtual-offset boundary. That caused
the second long request in an MP-mode run to fail with
``HIP error: invalid argument``.

These tests exercise the high-level Python entry points with allocator
addresses that straddle the 2 GB mark and the pin-chunk boundary between
the initial pin and the lazy expansion region.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator

GIB = 1 << 30
MIB = 1 << 20


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)


def _wait_for_expansion(alloc: LazyMemoryAllocator) -> None:
    # Give the background expansion thread time to pin and commit everything.
    for _ in range(50):
        if alloc._curr_size >= alloc._final_size:
            return
        time.sleep(0.1)


@pytest.fixture
def alloc_2gb_plus():
    """Allocator with 2 GB initial pin and 6 GB final — spans the 2 GB
    boundary that tripped the previous kernel implementation."""
    alloc = LazyMemoryAllocator(init_size=2 * GIB, final_size=6 * GIB)
    _wait_for_expansion(alloc)
    try:
        yield alloc
    finally:
        alloc.close()


@pytest.mark.parametrize("obj_size_mb", [64, 128])
def test_d2h_then_h2d_across_2gb_boundary(alloc_2gb_plus, obj_size_mb):
    """Sequentially allocate and copy 128 MB objects until we're well past
    2 GB. Verifies that both directions succeed and the payload round-trips
    through the lazy pinned buffer.
    """
    torch.cuda.set_device(0)

    obj_bytes = obj_size_mb * MIB
    gpu_src = torch.empty(obj_bytes, dtype=torch.uint8, device="cuda:0").fill_(0x5A)
    gpu_dst = torch.empty(obj_bytes, dtype=torch.uint8, device="cuda:0")

    # 30 × 128 MB = 3.75 GB → last object starts past 2 GB.
    n_iters = (4 * GIB) // obj_bytes
    objs = []
    for _ in range(n_iters):
        obj = alloc_2gb_plus.allocate(torch.Size([obj_bytes]), torch.uint8)
        if obj is None:
            pytest.fail("allocator returned None before reaching 4 GB")
        lmcache_memcpy_async_d2h(gpu_src, obj)
        torch.cuda.synchronize()
        objs.append(obj)

    assert any(o.meta.address >= 2 * GIB for o in objs), (
        "test did not actually reach past the 2 GB boundary"
    )

    # Round-trip: H2D back and verify payload
    for obj in objs:
        gpu_dst.zero_()
        lmcache_memcpy_async_h2d(obj, gpu_dst)
        torch.cuda.synchronize()
        assert int(gpu_dst[0].item()) == 0x5A
        assert int(gpu_dst[-1].item()) == 0x5A


def test_d2h_exact_pin_chunk_boundary(alloc_2gb_plus):
    """Exercise transfers whose destination straddles the 2 GB initial-pin
    boundary and lands strictly inside the lazy expansion region. These are
    the addresses that triggered the original failure.
    """
    torch.cuda.set_device(0)

    obj_bytes = 128 * MIB
    gpu = torch.empty(obj_bytes, dtype=torch.uint8, device="cuda:0").fill_(0xA5)

    saw_boundary = False
    saw_expansion = False
    held = []
    # Walk through enough memory to reach the expansion region.
    for _ in range((3 * GIB) // obj_bytes):
        obj = alloc_2gb_plus.allocate(torch.Size([obj_bytes]), torch.uint8)
        if obj is None:
            break
        held.append(obj)
        addr = obj.meta.address
        if addr + obj_bytes == 2 * GIB:
            saw_boundary = True
        if addr >= 2 * GIB:
            saw_expansion = True
        # Every D2H here used to trip the kernel for addr >= 2 GB - 128 MB.
        lmcache_memcpy_async_d2h(gpu, obj)
        torch.cuda.synchronize()

    assert saw_boundary, "test never covered the exact 2 GB boundary"
    assert saw_expansion, "test never reached the expansion region"
