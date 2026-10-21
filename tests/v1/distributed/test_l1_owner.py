# SPDX-License-Identifier: Apache-2.0
"""L1 identity follows an allocation, not serialized cache metadata."""

# Standard
from collections.abc import Iterator
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    GDSMemoryObject,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


@pytest.fixture
def managers() -> Iterator[tuple[L1Manager, L1Manager]]:
    """Keep real L1 lifecycle/allocation, replacing only CPU pinning calls."""
    with (
        patch(
            "lmcache.v1.memory_management._allocate_cpu_memory",
            side_effect=lambda size, *args, **kwargs: torch.empty(
                size, dtype=torch.uint8
            ),
        ),
        patch("lmcache.v1.memory_management._free_cpu_memory"),
    ):
        config = L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4096, use_lazy=False, align_bytes=64, shm_name=""
            )
        )
        first, second = L1Manager(config), L1Manager(config)
        try:
            yield first, second
        finally:
            first.close()
            second.close()


def test_l1_reservation_owner_survives_prefetch_completion(
    managers: tuple[L1Manager, L1Manager],
) -> None:
    """Direct and prefetch reservations stamp the actual L1 before exposure."""
    first, second = managers
    assert first.l1_manager_id != second.l1_manager_id
    key = ObjectKey(ObjectKey.IntHash2Bytes(1), "owner-test", 0)
    layout = MemoryLayoutDesc([torch.Size([64])], [torch.uint8])
    error, obj = first.reserve_write([key], [False], layout, tag="prefetch")[key]
    assert error == L1Error.SUCCESS and obj is not None
    assert obj.get_l1_manager() == first.l1_manager_id
    obj.set_l1_manager(first.l1_manager_id)
    with pytest.raises(ValueError, match="another L1 manager"):
        obj.set_l1_manager(second.l1_manager_id)

    assert first.finish_write_and_reserve_read([key], tag="prefetch")[key] == (
        L1Error.SUCCESS,
        obj,
    )
    assert obj.get_l1_manager() == first.l1_manager_id
    assert second.reserve_read([key])[key] == (L1Error.KEY_NOT_EXIST, None)
    assert first.finish_read([key])[key] == L1Error.SUCCESS


@pytest.mark.parametrize(
    "obj_type", [TensorMemoryObj, BytesBufferMemoryObj, GDSMemoryObject]
)
def test_owner_is_not_serialized_metadata(obj_type: type[MemoryObj]) -> None:
    """All object kinds start unowned, including metadata-based reconstruction."""
    metadata = MemoryObjMetadata(torch.Size([64]), torch.uint8, 0, 64, 1)

    def create(meta: MemoryObjMetadata) -> MemoryObj:
        if obj_type is TensorMemoryObj:
            return TensorMemoryObj(torch.empty(64, dtype=torch.uint8), meta, None)
        if obj_type is BytesBufferMemoryObj:
            return BytesBufferMemoryObj(b"x" * 64, meta)
        return GDSMemoryObject(meta)

    obj = create(metadata)
    before = obj.metadata.to_dict()
    assert obj.get_l1_manager() is None
    obj.set_l1_manager(123)
    assert obj.metadata.to_dict() == before
    reconstructed = create(MemoryObjMetadata.from_dict(obj.metadata.to_dict()))
    assert reconstructed.get_l1_manager() is None


@pytest.mark.parametrize("batched", [False, True])
def test_recycled_page_starts_without_previous_owner(batched: bool) -> None:
    """The same Python object may back a new allocation in a paged allocator."""
    shapes, dtypes = [torch.Size([64])], [torch.uint8]
    allocator = PagedTensorMemoryAllocator(
        torch.empty(64, dtype=torch.uint8), shapes, dtypes
    )
    obj = allocator.allocate(shapes, dtypes)
    assert obj is not None
    obj.set_l1_manager(123)
    allocator.free(obj)

    if batched:
        objects = allocator.batched_allocate(shapes, dtypes, 1)
        assert objects is not None
        recycled = objects[0]
    else:
        recycled = allocator.allocate(shapes, dtypes)
    assert recycled is obj
    assert recycled.get_l1_manager() is None
    recycled.set_l1_manager(456)
    allocator.free(recycled)
    assert allocator.memcheck()
    recycled.invalidate()
