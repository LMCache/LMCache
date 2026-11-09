# SPDX-License-Identifier: Apache-2.0
"""The memory manager of a shadow L1.

A shadow L1 stages a :class:`~lmcache.v1.memory_management.BorrowedMemoryObj`
placeholder per key as an owned L1 stages a write buffer, but the placeholder
holds no bytes: the borrowing L2 adapter's load binds it to the lease it holds
at the owner and to a read of the bytes into a device region at retrieve time.
"""

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryObj,
    MemoryObjMetadata,
)


class ShadowL1MemoryManager:
    """Memory manager behind :class:`ShadowL1Manager`: hands out
    :class:`BorrowedMemoryObj` placeholders with no capacity check, since the
    bytes are someone else's, and frees an object by ending its lease.
    Satisfies ``L1MemoryManagerProtocol``."""

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Return ``(SUCCESS, placeholders)``, ``count`` unbound objects with
        ``layout_desc``; never out of memory."""
        size = sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )
        objects: list[MemoryObj] = [
            BorrowedMemoryObj(
                MemoryObjMetadata(
                    shape=layout_desc.shapes[0],
                    dtype=layout_desc.dtypes[0],
                    address=-1,
                    phy_size=size,
                    ref_count=0,
                    shapes=list(layout_desc.shapes),
                    dtypes=list(layout_desc.dtypes),
                )
            )
            for _ in range(count)
        ]
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """End each object's lease; an unbound placeholder just goes away.

        Raises:
            TypeError: If an object is not a :class:`BorrowedMemoryObj`,
                which would mean a wiring bug placed local memory here.
        """
        for obj in mem_objs:
            if not isinstance(obj, BorrowedMemoryObj):
                raise TypeError(f"Shadow L1 holds only borrowed objects, got {obj!r}")
            obj.release()
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(0, 0)``: borrowed bytes are not local memory."""
        return 0, 0

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None; there is no local buffer to register."""
        return None

    def close(self) -> None:
        """Do nothing; there is nothing to unmap."""

    def memcheck(self) -> bool:
        """Return True; there is no allocator bookkeeping to check."""
        return True
