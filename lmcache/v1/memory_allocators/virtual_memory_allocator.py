# SPDX-License-Identifier: Apache-2.0

# Standard
import abc
import threading
from typing import TYPE_CHECKING, Callable, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

if TYPE_CHECKING:
    from lmcache.v1.resource_manifest import ResourceDescriptor

logger = init_logger(__name__)


class MemoryRegion:
    """A named, typed slice of physical memory managed by one leaf allocator.

    Attributes:
        name: Human-readable label used in hit-rate stats ("dram", "bar_0", ...).
        allocator: The leaf allocator that owns this physical buffer.
        size_bytes: Logical capacity in bytes.
    """

    def __init__(
        self,
        name: str,
        allocator: MemoryAllocatorInterface,
        size_bytes: int,
    ):
        self.name = name
        self.allocator = allocator
        self.size_bytes = size_bytes

    def owns(self, memory_obj: MemoryObj) -> bool:
        """Check whether ``memory_obj`` was allocated from this region.

        Args:
            memory_obj: The memory object to test.

        Returns:
            True if the object's data pointer falls within this region's buffer.
        """
        ptr = memory_obj.data_ptr
        buf = getattr(self.allocator, "buffer", None)
        if buf is None:
            inner = getattr(self.allocator, "pin_allocator", None)
            buf = getattr(inner, "buffer", None) if inner else None
        if buf is None:
            return False
        start = buf.data_ptr()
        return start <= ptr < start + self.size_bytes

    def __str__(self):
        return f"MemoryRegion({self.name}, {self.size_bytes // 1024**3} GB)"


class AllocationPolicy(abc.ABC):
    """Chooses the preferred order in which to try regions for each allocation.

    ``rank(regions)`` returns a list of region indices ordered from most to
    least preferred. ``VirtualMemoryAllocator`` tries them left-to-right and
    stops at the first successful allocation.

    ``on_allocated(region_index)`` is called once when an allocation succeeds
    so that stateful policies can update counters.
    """

    @abc.abstractmethod
    def rank(self, regions: list[MemoryRegion]) -> list[int]:
        """Return region indices ordered from most to least preferred."""
        raise NotImplementedError

    @abc.abstractmethod
    def on_allocated(self, region_index: int):
        """Called when an allocation succeeds in region at region_index."""
        raise NotImplementedError


class LinearPolicy(AllocationPolicy):
    """Always try regions in fixed order (index 0 first).

    Region 0 fills completely before region 1 is touched, which is the right
    behaviour for a DRAM->PCIe BAR spill hierarchy.
    """

    def rank(self, regions: list[MemoryRegion]) -> list[int]:
        return list(range(len(regions)))

    def on_allocated(self, region_index: int):
        pass


class InterleavingPolicy(AllocationPolicy):
    """Distribute allocations across regions proportional to capacity.

    Tracks how many allocations each region has received and ranks regions by
    their deficit (how far below the target fraction each one is), effectively
    interleaving allocations across regions.
    """

    def __init__(self, sizes: list[int]):
        total = sum(sizes)
        self._targets = [s / total for s in sizes]
        self._counts = [0] * len(sizes)
        self._total = 0
        self._lock = threading.Lock()

    def rank(self, regions: list[MemoryRegion]) -> list[int]:
        with self._lock:
            if self._total == 0:
                return sorted(
                    range(len(regions)),
                    key=lambda i: regions[i].size_bytes,
                    reverse=True,
                )
            deficit = [
                self._targets[i] - self._counts[i] / self._total
                for i in range(len(regions))
            ]
            return sorted(range(len(regions)), key=lambda i: deficit[i], reverse=True)

    def on_allocated(self, region_index: int):
        with self._lock:
            self._counts[region_index] += 1
            self._total += 1


def choose_allocation_policy(
    regions: list["MemoryRegion"],
    policy_name: str,
) -> "AllocationPolicy":
    """Pick an AllocationPolicy by name.

    Args:
        regions: The regions the policy will govern.
        policy_name: ``"linear"``, ``"interleaving"``, or ``"auto"``
            (interleaving when more than one region, linear otherwise).

    Returns:
        An AllocationPolicy instance.
    """
    if policy_name == "interleaving" or (
        policy_name == "auto" and len(regions) > 1
    ):
        return InterleavingPolicy([r.size_bytes for r in regions])
    if policy_name in ("linear", "auto"):
        return LinearPolicy()
    raise ValueError(f"Unknown allocation policy: {policy_name!r}")


def build_regions_from_descriptors(
    descriptors: list["ResourceDescriptor"],
    make_dram_allocator: Callable[[int], MemoryAllocatorInterface],
) -> list["MemoryRegion"]:
    """Build MemoryRegions from resource descriptors.

    Args:
        descriptors: Resource descriptors (from a manifest or env vars).
        make_dram_allocator: Factory ``(size_bytes) -> allocator`` for DRAM
            regions. Callers inject NUMA or config-specific behaviour here.

    Returns:
        One MemoryRegion per descriptor, in input order.
    """
    from lmcache.v1.memory_allocators.pcie_bar_memory_allocator import (
        PcieBarMemoryAllocator,
    )

    regions: list[MemoryRegion] = []
    for desc in descriptors:
        size_bytes = int(desc.capacity_gb * 1024**3)
        if desc.type == "pcie_bar":
            if not desc.path:
                raise ValueError(
                    f"Resource {desc.name!r} has type='pcie_bar' but no path"
                )
            allocator = PcieBarMemoryAllocator(
                size=size_bytes,
                bar_path=desc.path,
                bar_offset=desc.offset,
            )
        elif desc.type == "dram":
            allocator = make_dram_allocator(size_bytes)
        else:
            raise ValueError(
                f"Unknown resource type: {desc.type!r} in {desc.name!r}"
            )
        regions.append(MemoryRegion(desc.name, allocator, size_bytes))
    return regions


class VirtualMemoryAllocator(MemoryAllocatorInterface):
    """Unified virtual memory layer over an ordered list of MemoryRegions.

    Presents a single allocator interface to LMCache while routing each
    allocation to one of N underlying regions according to a pluggable policy.

    Supported policies:
        LinearPolicy: fill region 0 first, then region 1, ...
        InterleavingPolicy: distribute proportional to region sizes.
    """

    def __init__(
        self,
        regions: list[MemoryRegion],
        policy: AllocationPolicy,
        gpu_to_regions: dict | None = None,
    ):
        if not regions:
            raise ValueError("At least one region is required")
        self._regions = regions
        self._policy = policy
        self._gpu_to_regions = gpu_to_regions

        self._inner_to_region: dict = {}
        for region in regions:
            inner = self._leaf_inner(region.allocator)
            if inner is not None:
                self._inner_to_region[id(inner)] = region

        total_gb = sum(r.size_bytes for r in regions) / 1024**3
        region_lines = "\n".join(
            f"  [{i}] {r.name:<12}  {r.size_bytes / 1024**3:>6.1f} GB  "
            f"({type(r.allocator).__name__})"
            for i, r in enumerate(regions)
        )
        gpu_note = (
            f"  gpu_to_regions: {gpu_to_regions}" if gpu_to_regions else ""
        )
        logger.info(
            "VirtualMemoryAllocator initialized — policy=%s  total=%.1f GB\n%s%s",
            type(policy).__name__,
            total_gb,
            region_lines,
            gpu_note,
        )

    @staticmethod
    def _leaf_inner(allocator: MemoryAllocatorInterface) -> MemoryAllocatorInterface | None:
        """Return the innermost TensorMemoryAllocator that a leaf allocator wraps."""
        inner = getattr(allocator, "_inner", None)
        if inner is not None:
            return inner
        pin = getattr(allocator, "pin_allocator", None)
        return pin

    def _active_regions(self) -> list[MemoryRegion]:
        if self._gpu_to_regions is None:
            return self._regions
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                gpu_id = _torch.cuda.current_device()
            else:
                gpu_id = 0
        except Exception:
            gpu_id = 0
        indices = self._gpu_to_regions.get(gpu_id, list(range(len(self._regions))))
        return [self._regions[i] for i in indices]

    @property
    def regions(self) -> list[MemoryRegion]:
        """The ordered list of memory regions managed by this allocator."""
        return self._regions

    @property
    def align_bytes(self) -> int:
        """Alignment granularity of the first region's allocator."""
        return self._regions[0].allocator.align_bytes

    @property
    def total_size_bytes(self) -> int:
        """Total capacity in bytes across all regions."""
        return sum(r.size_bytes for r in self._regions)

    def region_of(self, memory_obj: MemoryObj) -> str:
        """Return the name of the region that owns ``memory_obj``.

        Args:
            memory_obj: A previously allocated memory object.

        Returns:
            The region name, or ``"unknown"`` if no region claims it.
        """
        for region in self._regions:
            if region.owns(memory_obj):
                return region.name
        return "unknown"

    def is_bar_backed(self, memory_obj: MemoryObj) -> bool:
        """Return True if ``memory_obj`` was allocated from a PCIe BAR region.

        Args:
            memory_obj: A previously allocated memory object.

        Returns:
            True when the owning region's allocator has ``is_pcie_bar_memory``.
        """
        for region in self._regions:
            if region.owns(memory_obj):
                return getattr(region.allocator, "is_pcie_bar_memory", False)
        return False

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
    ) -> MemoryObj | None:
        """Allocate a memory object from the highest-priority region with space.

        Args:
            shapes: Tensor shape(s) for the allocation.
            dtypes: Tensor dtype(s).
            fmt: Memory format.
            allocator_type: Passed through to the leaf allocator.

        Returns:
            A MemoryObj on success, or None if all regions are full.
        """
        active = self._active_regions()
        for idx in self._policy.rank(active):
            region = active[idx]
            result = region.allocator.allocate(shapes, dtypes, fmt, allocator_type)
            if result is not None:
                self._policy.on_allocated(idx)
                logger.debug(
                    "alloc -> %s  ptr=0x%x  shape=%s",
                    region.name,
                    result.data_ptr,
                    shapes,
                )
                return result
            logger.info("alloc: %s full, trying next region", region.name)
        logger.info("alloc: all regions full — shape=%s", shapes)
        return None

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
        distribute: bool = False,
    ) -> list[MemoryObj] | None:
        """Allocate batch_size equal-sized slots.

        When distribute=False (default), the entire batch is placed on the
        single region that the policy ranks first.

        When distribute=True, each slot is allocated independently via
        allocate(), so slots from one batch can land on different regions.
        """
        active = self._active_regions()
        if distribute:
            results: list[MemoryObj] = []
            for _ in range(batch_size):
                obj = self.allocate(shapes, dtypes, fmt, allocator_type)
                if obj is None:
                    logger.debug(
                        "batched_alloc(distribute): all regions full after %d/%d slots",
                        len(results),
                        batch_size,
                    )
                    for r in results:
                        self.free(r)
                    return None
                results.append(obj)
            return results

        for idx in self._policy.rank(active):
            region = active[idx]
            result = region.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, allocator_type
            )
            if result is not None:
                self._policy.on_allocated(idx)
                logger.debug(
                    "batched_alloc(%d slots) -> %s  ptr=0x%x",
                    batch_size,
                    region.name,
                    result[0].data_ptr,
                )
                return result
            logger.debug(
                "batched_alloc(%d slots): %s full, trying next region",
                batch_size,
                region.name,
            )
        logger.debug("batched_alloc(%d slots): all regions full", batch_size)
        return None

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: str | None = None) -> None:
        """Free a single memory object, returning it to its owning region.

        Args:
            memory_obj: The object to free.
            allocator_type: Ignored; present for interface compatibility.
        """
        region_name = self.region_of(memory_obj)
        logger.debug("free <- %s  ptr=0x%x", region_name, memory_obj.data_ptr)
        parent = memory_obj.parent()
        if parent is not None:
            parent.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch of memory objects, grouped by owning region.

        Args:
            memory_objs: Objects to free.
            allocator_type: Ignored; present for interface compatibility.
            update_stats: Forwarded to the leaf allocator.
        """
        if not memory_objs:
            return
        groups: dict = {}
        for obj in memory_objs:
            parent = obj.parent()
            pid = id(parent)
            if pid not in groups:
                groups[pid] = (parent, [])
            groups[pid][1].append(obj)

        for pid, (parent_alloc, objs) in groups.items():
            region = self._inner_to_region.get(pid)
            if region is not None:
                logger.debug("batched_free %d slots <- %s", len(objs), region.name)
                region.allocator.batched_free(objs, update_stats=update_stats)
            else:
                logger.debug("batched_free %d slots <- (unknown region)", len(objs))
                parent_alloc.batched_free(objs, update_stats=update_stats)

    def get_memory_usage(self) -> tuple[int, int]:
        """Aggregate memory usage across all regions.

        Returns:
            (used_bytes, total_bytes) summed over every region.
        """
        used = 0
        total = 0
        for region in self._regions:
            alloc = region.allocator
            if hasattr(alloc, "get_memory_usage"):
                u, t = alloc.get_memory_usage()
                used += u
                total += t
            else:
                inner = self._leaf_inner(alloc)
                if inner is not None and hasattr(inner, "address_manager"):
                    am = inner.address_manager
                    free_size = am.get_free_size()
                    heap_size = am.get_heap_size()
                    used += heap_size - free_size
                    total += heap_size
                else:
                    total += region.size_bytes
        return used, total

    def memcheck(self) -> bool:
        """Return True if all regions pass their internal consistency check."""
        return all(r.allocator.memcheck() for r in self._regions)

    def close(self) -> None:
        """Close all underlying region allocators and release resources."""
        for region in self._regions:
            region.allocator.close()

    def __str__(self):
        parts = ", ".join(str(r) for r in self._regions)
        policy_name = type(self._policy).__name__
        return f"VirtualMemoryAllocator([{parts}], policy={policy_name})"
