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
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)


def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    shape1 = torch.Size([512, 512])
    data1 = allocator.allocate(shape1, torch.float)
    assert data1 is not None
    assert data1.tensor.dtype == torch.float
    assert data1.tensor.shape == shape1

    # 1024 * 1024 * 2 = 2MB
    shape2 = torch.Size([1024, 1024])
    data2 = allocator.allocate(shape2, torch.bfloat16)
    assert data2 is not None
    assert data2.tensor.dtype == torch.bfloat16
    assert data2.tensor.shape == shape2

    # 2048 * 2048 * 1 = 4MB
    shape3 = torch.Size([2048, 2048])
    data3 = allocator.allocate(shape3, torch.int8)
    assert data3 is not None
    assert data3.tensor.dtype == torch.int8
    assert data3.tensor.shape == shape3

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    shape4 = torch.Size([3, 5, 7])
    data4 = allocator.allocate(shape4, torch.half)
    assert data4 is not None
    assert data4.tensor.dtype == torch.half
    assert data4.tensor.shape == shape4

    data_fail = allocator.allocate(
        torch.Size([max_size]), torch.float
    )  # This should fail
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
        allocator = PagedTensorMemoryAllocator(tensor_buffer, [shape], [dtype], fmt)
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
        total_size, use_paging=use_paging, shapes=[shape], dtypes=[dtype], fmt=fmt
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

    shape = torch.Size([4096])
    data = allocator.allocate(shape, torch.float)
    assert data is not None
    assert data.tensor.dtype == torch.float
    assert data.tensor.shape == shape

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

    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, torch.float)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate(shape, torch.float)

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
    shape = torch.Size([100, 2, 1024])
    objs = allocator.batched_allocate(
        shape, torch.bfloat16, batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == torch.bfloat16
        assert obj.tensor.shape == shape
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
    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, [], MemoryFormat.BINARY_BUFFER)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    assert len(allocator.pin_allocator.explicit_list) == 1

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    allocator.close()


def test_memory_obj_metadata_to_and_from_dict():
    shape1 = torch.Size([128, 10])
    dtype1 = torch.float
    shape2 = torch.Size([256, 10])
    dtype2 = torch.uint8
    shapes = [shape1, shape2]
    dtypes = [dtype1, dtype2]
    metadata1 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
    )
    dict1 = metadata1.to_dict()
    metadata_from_dict_1 = MemoryObjMetadata.from_dict(dict1)
    assert metadata_from_dict_1.shape == shape1
    assert metadata_from_dict_1.dtype == dtype1
    assert metadata_from_dict_1.shapes is None
    assert metadata_from_dict_1.dtypes is None

    metadata2 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
        shapes=shapes,
        dtypes=dtypes,
    )
    dict2 = metadata2.to_dict()
    metadata_from_dict_2 = MemoryObjMetadata.from_dict(dict2)
    assert metadata_from_dict_2.shape == shape1
    assert metadata_from_dict_2.dtype == dtype1
    assert metadata_from_dict_2.shapes == shapes
    assert metadata_from_dict_2.dtypes == dtypes
