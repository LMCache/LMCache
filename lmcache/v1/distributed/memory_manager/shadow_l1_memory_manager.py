# SPDX-License-Identifier: Apache-2.0
"""Memory manager for an L1 that indexes borrowed views and stores no payload."""

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import BorrowedMemoryObj, MemoryObj


class ShadowL1MemoryManager:
    """Memory manager behind :class:`ShadowL1Manager`.

    Its objects are :class:`BorrowedMemoryObj` views whose bytes someone
    else owns, so it never allocates and has no capacity; freeing an object
    ends its lease. Satisfies ``L1MemoryManagerProtocol``.
    """

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Refuse every allocation; shadows are admitted, never written.

        Args:
            layout_desc: Ignored.
            count: Ignored.

        Returns:
            ``(OUT_OF_MEMORY, [])``.
        """
        return L1Error.OUT_OF_MEMORY, []

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """End the lease of each borrowed view.

        Args:
            mem_objs: Views admitted into the shadow L1.

        Returns:
            ``SUCCESS``.

        Raises:
            TypeError: If an object is not a borrowed view.
        """
        for obj in mem_objs:
            if not isinstance(obj, BorrowedMemoryObj):
                raise TypeError(f"Shadow L1 holds only borrowed views, got {obj!r}")
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
