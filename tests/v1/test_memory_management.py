# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)


def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    data1 = allocator.allocate([512, 512], torch.float)
    assert data1 is not None
    assert data1.tensor.dtype == torch.float
    assert data1.tensor.shape == (512, 512)

    # 1024 * 1024 * 2 = 2MB
    data2 = allocator.allocate([1024, 1024], dtype=torch.bfloat16)
    assert data2 is not None
    assert data2.tensor.dtype == torch.bfloat16
    assert data2.tensor.shape == (1024, 1024)

    # 2048 * 2048 * 1 = 4MB
    data3 = allocator.allocate([2048, 2048], dtype=torch.int8)
    assert data3 is not None
    assert data3.tensor.dtype == torch.int8
    assert data3.tensor.shape == (2048, 2048)

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    data4 = allocator.allocate([3, 5, 7], dtype=torch.half)
    assert data4 is not None
    assert data4.tensor.dtype == torch.half
    assert data4.tensor.shape == (3, 5, 7)

    data_fail = allocator.allocate([max_size], dtype=torch.float)  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()

    allocator.close()


def check_paged_allocator(allocator, shape, dtype, fmt, max_num_pages):
    # Allocate one page
    data1 = allocator.allocate(shape, dtype, fmt)
    assert data1 is not None
    assert data1.tensor.dtype == dtype
    assert data1.tensor.shape == shape

    # Allocate another 2 pages
    data2 = allocator.batched_allocate(shape, dtype, 2, fmt)

    for data in data2:
        assert data is not None
        assert data.tensor.dtype == dtype
        assert data.tensor.shape == shape

    # Allocate a smaller page
    smaller_shape = torch.Size([2, 32, 8, 1024])
    data3 = allocator.allocate(smaller_shape, dtype, fmt)
    assert data3 is not None
    assert data3.tensor.dtype == dtype
    assert data3.tensor.shape == smaller_shape

    allocator.free(data3)
    assert allocator.memcheck()

    allocator.batched_free(data2)
    assert allocator.memcheck()

    allocator.free(data1)
    assert allocator.memcheck()

    data_fail = allocator.batched_allocate(
        shape, dtype, max_num_pages + 1, fmt
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize(
    "use_paging",
    [True, False],
)
def test_tensor_allocator(use_paging):
    total_size = 1024 * 1024 * 128  # 128MB
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")
    if use_paging:
        shape = torch.Size([2, 32, 16, 1024])  # 64 pages
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD
        num_pages = 64
        allocator = PagedTensorMemoryAllocator(tensor_buffer, shape, dtype, fmt)
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        allocator = TensorMemoryAllocator(tensor_buffer)
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
@pytest.mark.parametrize(
    "use_paging",
    [
        False,
        True,
    ],
)
def test_device_allocators(alloc_cls, use_paging):
    total_size = 1024 * 1024 * 128  # 128MB

    shape = torch.Size([2, 32, 16, 1024])  # 64 pages
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_2LTD

    allocator = alloc_cls(
        total_size, use_paging=use_paging, shape=shape, dtype=dtype, fmt=fmt
    )

    if use_paging:
        num_pages = 64
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_inplace_modification(alloc_cls):
    total_size = 1024 * 1024
    allocator = alloc_cls(total_size)

    data = allocator.allocate([4096], torch.float)
    assert data is not None
    assert data.tensor.dtype == torch.float
    assert data.tensor.shape == (4096,)

    data.tensor.fill_(1.0)
    assert torch.all(data.tensor == 1.0)

    data.tensor[1] = 2.0
    assert data.tensor[1] == 2.0

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_boundary_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 10], torch.float)
    allocator.allocate([512, 10], torch.float)
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate([512, 10], torch.float)

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_batched_alloc(alloc_cls):
    total_size = 32 * 100 * 2 * 1024 * 2
    batch_size = 32
    allocator = alloc_cls(total_size)
    objs = allocator.batched_allocate(
        [100, 2, 1024], torch.bfloat16, batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == torch.bfloat16
        assert obj.tensor.shape == (100, 2, 1024)
    allocator.batched_free(objs)

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        MixedMemoryAllocator,
    ],
)
def test_mixed_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 0], None, MemoryFormat.BINARY_BUFFER)
    allocator.allocate([512, 10], torch.float)
    allocator.free(data1)

    assert len(allocator.pin_allocator.explicit_list) == 1

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    allocator.close()
