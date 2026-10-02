# SPDX-License-Identifier: Apache-2.0
"""Check logical layouts when the paged allocator reuses physical pages."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryObj


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
@pytest.mark.parametrize("grouped", [False, True], ids=["partial", "groups"])
def test_page_allocation_uses_requested_layout(batched: bool, grouped: bool) -> None:
    """A reused page exposes requested bytes and group views, then its full size."""
    page_shape = torch.Size([64])
    count = 2 if batched else 1
    allocator = PagedTensorMemoryAllocator(
        tensor=torch.zeros(count * page_shape.numel(), dtype=torch.uint8),
        shapes=[page_shape],
        dtypes=[torch.uint8],
        fmt=MemoryFormat.BINARY_BUFFER,
    )
    shapes = [torch.Size([4]), torch.Size([8])] if grouped else [torch.Size([16])]
    dtypes = [torch.float16, torch.uint8] if grouped else [torch.uint8]

    def allocate(
        requested_shapes: list[torch.Size], requested_dtypes: list[torch.dtype]
    ) -> list[TensorMemoryObj]:
        if batched:
            objects = allocator.batched_allocate(
                requested_shapes,
                requested_dtypes,
                batch_size=count,
                fmt=MemoryFormat.BINARY_BUFFER,
            )
            assert objects is not None
            return objects
        obj = allocator.allocate(
            requested_shapes, requested_dtypes, fmt=MemoryFormat.BINARY_BUFFER
        )
        assert obj is not None
        return [obj]

    objects = allocate(shapes, dtypes)
    try:
        for obj in objects:
            expected_size = sum(
                shape.numel() * dtype.itemsize
                for shape, dtype in zip(shapes, dtypes, strict=True)
            )
            assert obj.get_size() == expected_size
            assert len(obj.byte_array) == expected_size
            assert obj.get_shapes() == shapes
            assert obj.get_dtypes() == dtypes
            for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
                view = obj.get_tensor(index)
                assert view is not None
                assert view.shape == shape
                assert view.dtype == dtype
                view.fill_(index + 1)
            for index in range(len(shapes)):
                view = obj.get_tensor(index)
                assert view is not None
                assert torch.all(view == index + 1)
    finally:
        for obj in objects:
            obj.ref_count_down()

    recycled = allocate([page_shape], [torch.uint8])
    try:
        for obj in recycled:
            assert obj.get_size() == page_shape.numel()
            assert len(obj.byte_array) == page_shape.numel()
            view = obj.get_tensor(0)
            assert view is not None
            assert view.shape == page_shape
    finally:
        for obj in recycled:
            obj.ref_count_down()
