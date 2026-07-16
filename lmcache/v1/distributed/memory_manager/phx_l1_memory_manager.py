# SPDX-License-Identifier: Apache-2.0
"""Phoenix-aware L1 memory manager.

A subclass of :class:`L1MemoryManager` (CPU pinned-DRAM tier) that
overrides :meth:`free` to transparently dispatch device-resident
MemoryObjs back to their own parent allocator.

When a :class:`PhxL2Adapter` loads data from NVMe via Phoenix DMA, it
produces device-resident MemoryObjs (allocated by
:class:`PhxDeviceMemoryAllocator`) and injects them into L1 via
:meth:`L1Manager.replace_memory_obj`.  These device objs are *not* owned
by the L1 CPU allocator, so freeing them through the plain
``L1MemoryManager.free`` (which calls ``batched_free`` on the CPU slab)
would corrupt bookkeeping.

This manager recognises device-resident objs by inspecting
``raw_tensor.device.type`` and routes them to ``obj.parent().free(obj)``
(the allocator that originally allocated them), while CPU objs continue
through the normal CPU slab ``batched_free`` path.
"""

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class PhxL1MemoryManager(L1MemoryManager):
    """CPU pinned-DRAM L1 manager with device-obj dispatch in ``free()``.

    Identical to :class:`L1MemoryManager` in every respect except
    :meth:`free`: device-resident objs (``raw_tensor.device.type !=
    "cpu"``) are returned to their parent allocator instead of being
    passed to the CPU slab ``batched_free``.  This lets L1 hold a mix of
    CPU-pinned and device-resident MemoryObjs without the upper
    :class:`L1Manager` layer needing to distinguish them.
    """

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free a mix of CPU and device memory objs.

        CPU objs are batched-freed via the CPU slab allocator.
        Device-resident objs are freed via their own ``parent()``
        allocator (e.g. :class:`PhxDeviceMemoryAllocator`), since the L1
        CPU allocator does not own GPU memory.

        Args:
            mem_objs: Objects to free (may be CPU-pinned or
                device-resident).

        Returns:
            ``L1Error.SUCCESS``.
        """
        cpu_objs: list[MemoryObj] = []
        for o in mem_objs:
            if o is None:
                continue
            try:
                rt = o.raw_tensor
            except Exception:
                rt = None
            if rt is not None and rt.device.type != "cpu":
                parent = o.parent()
                if parent is not None:
                    parent.free(o)
                else:
                    logger.warning(
                        "PhxL1MemoryManager: device MemoryObj has no "
                        "parent allocator; cannot free, leaking device "
                        "memory"
                    )
            else:
                cpu_objs.append(o)
        if cpu_objs:
            self._allocator.batched_free(cpu_objs)
        return L1Error.SUCCESS
