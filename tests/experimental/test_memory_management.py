import torch
import pytest

from lmcache.experimental.memory_management import (TensorMemoryAllocator,
                                                    HostMemoryAllocator,
                                                    PinMemoryAllocator,
                                                    GPUMemoryAllocator)


def check_allocator(allocator, max_size):
    data1 = allocator.allocate([512, 512], torch.float)
    assert data1 is not None
    assert data1.tensor().dtype == torch.float
    assert data1.tensor().shape == (512, 512)

    data2 = allocator.allocate([1024, 1024], dtype=torch.bfloat16)
    assert data2 is not None
    assert data2.tensor().dtype == torch.bfloat16
    assert data2.tensor().shape == (1024, 1024)

    data3 = allocator.allocate([2048, 2048], dtype=torch.int8)
    assert data3 is not None
    assert data3.tensor().dtype == torch.int8
    assert data3.tensor().shape == (2048, 2048)

    allocator.free(data2)
    assert data2.tensor() is None

    allocator.free(data2)  # This should not crash

    data4 = allocator.allocate([3, 5, 7], dtype=torch.half)
    assert data4 is not None
    assert data4.tensor().dtype == torch.half
    assert data4.tensor().shape == (3, 5, 7)

    data_fail = allocator.allocate([max_size],
                                   dtype=torch.float)  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()


def test_tensor_allocator():
    total_size = 1 << 25  # 32MB
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")

    allocator = TensorMemoryAllocator(tensor_buffer)

    check_allocator(allocator, total_size)


@pytest.mark.parametrize("alloc_cls", [
    HostMemoryAllocator,
    PinMemoryAllocator,
    GPUMemoryAllocator,
])
def test_device_allocators(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    check_allocator(allocator, total_size)
