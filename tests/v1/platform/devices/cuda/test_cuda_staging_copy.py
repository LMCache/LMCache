# SPDX-License-Identifier: Apache-2.0
"""Regression tests for lazy CUDA staging without the native extension."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.gpu_connector import gpu_ops
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.devices.cuda import device_ops
from lmcache.v1.platform.torch_ops import mem_kernels


@pytest.mark.parametrize(
    ("direction", "kind"), [(TransferDirection.H2D, 1), (TransferDirection.D2H, 2)]
)
def test_cuda_pointer_copy_uses_current_stream_and_splits_registrations(
    monkeypatch: pytest.MonkeyPatch, direction: TransferDirection, kind: int
) -> None:
    """Each piece of a cross-registration copy is queued on the current stream."""
    monkeypatch.setattr(torch.version, "hip", None)
    runtime = SimpleNamespace(
        cudaMemcpy=Mock(return_value=0), cudaMemcpyAsync=Mock(return_value=0)
    )
    monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)
    monkeypatch.setattr(device_ops, "_get_copy_lib", lambda: runtime)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0x3000)
    )

    device_ops.CudaDeviceOps().lmcache_memcpy_async(
        0x1000, 0x2000, 16, direction, 14, 16
    )

    calls = [
        tuple(arg.value for arg in call.args)
        for call in runtime.cudaMemcpyAsync.call_args_list
    ]
    assert calls == [
        (0x1000, 0x2000, 2, kind, 0x3000),
        (0x1002, 0x2002, 14, kind, 0x3000),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires NVIDIA CUDA for the non-default stream regression",
)
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_lazy_cuda_staging_preserves_stream_order(
    monkeypatch: pytest.MonkeyPatch, direction: TransferDirection
) -> None:
    """Lazy staging preserves stream order across a registration boundary."""
    # A fresh ops instance keeps the Python fallback active in native builds.
    monkeypatch.setattr(gpu_ops, "device_ops", device_ops.CudaDeviceOps())
    registration_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
    allocator = LazyMemoryAllocator(2 * registration_size, 2 * registration_size)
    chunk_bytes = 28 << 20
    objects: list[MemoryObj] = []
    stream = torch.cuda.Stream()
    try:
        for _ in range(3):
            allocated = allocator.allocate(
                torch.Size((chunk_bytes // 2,)), torch.bfloat16
            )
            assert allocated is not None
            objects.append(allocated)
        memory_obj = objects[-1]
        host = memory_obj.raw_tensor
        assert host is not None
        assert memory_obj.meta.address < registration_size
        assert memory_obj.meta.address + chunk_bytes > registration_size
        staging = torch.full((chunk_bytes,), 17, dtype=torch.uint8, device="cuda")
        previous_batch = torch.empty_like(staging)
        previous_batch.copy_(staging)
        host.fill_(51)
        torch.cuda._sleep(1)
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            # Leave a wide window for a copy on the wrong stream to overtake
            # the pending write (STORE) or the previous reader (RETRIEVE).
            torch.cuda._sleep(250_000_000)
            if direction == TransferDirection.D2H:
                staging.fill_(51)
                gpu_ops.lmcache_memcpy_async_d2h(staging, memory_obj)
            else:
                previous_batch.copy_(staging)
                gpu_ops.lmcache_memcpy_async_h2d(memory_obj, staging)
        stream.synchronize()

        if direction == TransferDirection.D2H:
            assert torch.all(host == 51)
        else:
            assert torch.all(previous_batch == 17)
            assert torch.all(staging == 51)
    finally:
        stream.synchronize()
        for memory_obj in objects:
            memory_obj.ref_count_down()
        allocator.close()
