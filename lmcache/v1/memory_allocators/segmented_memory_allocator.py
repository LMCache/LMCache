# SPDX-License-Identifier: Apache-2.0
"""L1 allocator over several independently allocated host memory segments.

Used when one allocation cannot hold the whole L1 (a ROCm GTT ``hipHostMalloc``
must stay below 512 GiB). ``MemoryObjMetadata.address`` is a logical address
over all segments; every object lies inside exactly one segment.
"""

# Standard
from typing import Callable
import bisect
import ctypes
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import AddressManager, FreeBlock

logger = init_logger(__name__)


def _split(size: int, segment_size: int) -> list[int]:
    """Split ``size`` bytes into ``segment_size`` pieces plus the remainder."""
    sizes = [segment_size] * (size // segment_size)
    return sizes + [size % segment_size] if size % segment_size else sizes


class _SegmentedAddressManager(AddressManager):
    """An :class:`AddressManager` whose free blocks never cross a segment.

    Segment ``k`` starts at the sum of the sizes of the earlier segments.
    Bases are only appended, beyond every existing block, so concurrent
    ``free`` calls always see the right boundaries.
    """

    def __init__(self, segment_sizes: list[int], align_bytes: int) -> None:
        """
        Args:
            segment_sizes: Initial segment sizes, multiples of ``align_bytes``.
            align_bytes: The alignment requirement for allocations.
        """
        super().__init__(segment_sizes[0], align_bytes)
        self._bases = [0]
        for size in segment_sizes[1:]:
            self.add_segment(size)

    def add_segment(self, size: int) -> None:
        """Append a segment of ``size`` bytes to the address space."""
        self._bases.append(self.get_heap_size())
        self.sbrk(size)

    def locate(self, address: int) -> tuple[int, int]:
        """Return ``(segment index, offset in that segment)`` of an address."""
        index = bisect.bisect_right(self._bases, address) - 1
        return index, address - self._bases[index]

    def free(self, address: int, size: int) -> None:
        """Free a range, split at segment boundaries.

        ``TensorMemoryAllocator.batched_free`` coalesces logically adjacent
        objects, which may lie in two segments.

        Args:
            address: Start of the range to free.
            size: Size of the range in bytes.
        """
        end = address + size
        while address < end:
            piece_end = min(end, self._segment_end(address))
            super().free(address, piece_end - address)
            address = piece_end

    def check_consistency(self) -> bool:
        """Check coalescing within segments and the size total.

        Returns:
            True if consistent, False otherwise.
        """
        blocks = list(self._explicit_list)
        if any(b.start + b.size > self._segment_end(b.start) for b in blocks):
            return False
        if any(
            self._can_merge_with_succ(a, b)
            for a, b in zip(blocks, blocks[1:], strict=False)
        ):
            return False
        free = sum(b.size for b in blocks)
        return free + self.total_allocated_size == self.get_heap_size()

    def _segment_end(self, address: int) -> int:
        index = bisect.bisect_right(self._bases, address)
        return self._bases[index] if index < len(self._bases) else self.get_heap_size()

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        return super()._can_merge_with_prev(
            curr_block, prev_block
        ) and curr_block.start < self._segment_end(prev_block.start)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        return super()._can_merge_with_succ(
            curr_block, succ_block
        ) and succ_block.start < self._segment_end(curr_block.start)


class SegmentedMemoryAllocator(TensorMemoryAllocator):
    """Lazily expanded allocator over non-contiguous host memory segments.

    The constructor allocates the initial capacity; a daemon thread allocates
    the rest one segment at a time and publishes each segment once it is
    usable. Allocation, batching and freeing are those of
    :class:`TensorMemoryAllocator` over a :class:`_SegmentedAddressManager`.
    """

    def __init__(
        self,
        init_size: int,
        final_size: int,
        max_segment_size: int,
        alloc_segment: Callable[[int], int],
        free_segment: Callable[[int], None],
        align_bytes: int = AddressManager.ALIGN_BYTES,
    ) -> None:
        """
        Args:
            init_size: Bytes allocated before the constructor returns.
            final_size: Total bytes, reached by background expansion.
            max_segment_size: Largest segment in bytes, rounded down to a
                multiple of ``max(align_bytes, page size)``.
            alloc_segment: Returns a host pointer to the given number of
                bytes, aligned to ``align_bytes``, or raises ``RuntimeError``.
            free_segment: Frees a pointer returned by ``alloc_segment``.
            align_bytes: Allocation alignment, a power of two.

        Raises:
            ValueError: If ``align_bytes`` is not a power of two or
                ``init_size`` is not positive.
            RuntimeError: If an initial segment cannot be allocated or is
                misaligned; the segments allocated so far are freed first.
        """
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")
        if init_size <= 0:
            raise ValueError("init_size must be positive")
        granule = max(align_bytes, os.sysconf("SC_PAGE_SIZE"))
        segment_size = max_segment_size // granule * granule
        final = -(-final_size // granule) * granule
        init = min(-(-init_size // granule) * granule, final)
        initial_sizes = _split(init, segment_size)
        expansion_sizes = _split(final - init, segment_size)

        self._align = align_bytes
        self._alloc_segment = alloc_segment
        self._free_segment = free_segment
        self._final_size = final
        self._segments: list[torch.Tensor] = []
        """Flat uint8 views of the published segments, in logical order."""
        self._lock = threading.Lock()
        self._closed = False

        start = time.monotonic()
        try:
            for size in initial_sizes:
                self._segments.append(self._allocate_segment(size))
        except BaseException:
            for segment in self._segments:
                free_segment(segment.data_ptr())
            raise
        logger.info(
            "Segmented L1: %d bytes in %d segment(s) allocated in %.1f s, "
            "%d bytes in %d segment(s) to allocate in the background",
            init,
            len(initial_sizes),
            time.monotonic() - start,
            final - init,
            len(expansion_sizes),
        )

        # There is no contiguous buffer; slices come from the segment views.
        super().__init__(torch.empty(0, dtype=torch.uint8), align_bytes=align_bytes)
        self._segmented = _SegmentedAddressManager(initial_sizes, align_bytes)
        self.address_manager = self._segmented
        self._expand_thread = threading.Thread(
            target=self._expand_worker,
            args=(expansion_sizes,),
            daemon=True,
            name="segmented-l1-expand",
        )
        self._expand_thread.start()

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used bytes, published capacity in bytes)``."""
        total = self.address_manager.get_heap_size()
        return total - self.address_manager.get_free_size(), total

    def memory_region_count(self) -> int:
        """Return the number of published segments."""
        return len(self._segments)

    def close(self) -> None:
        """Stop expansion, synchronize the device and free every segment once.

        Waits for an in-flight segment allocation. Idempotent. Objects still
        allocated become invalid; callers free them first.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._expand_thread.join()
        if torch_dev.is_available():
            torch_dev.synchronize()
        segments, self._segments = self._segments, []
        self.address_manager = AddressManager(0, self._align)
        for segment in segments:
            self._free_segment(segment.data_ptr())

    def __str__(self) -> str:
        """Return the allocator name."""
        return "SegmentedMemoryAllocator"

    # Helper functions
    def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
        index, offset = self._segmented.locate(start)
        return self._segments[index][offset : offset + size]

    def _allocate_segment(self, size: int) -> torch.Tensor:
        """Allocate one segment and return a flat uint8 view of it."""
        ptr = self._alloc_segment(size)
        if ptr % self._align:
            self._free_segment(ptr)
            raise RuntimeError(
                f"Segment at {ptr:#x} is not aligned to {self._align} bytes"
            )
        return torch.frombuffer(
            (ctypes.c_uint8 * size).from_address(ptr), dtype=torch.uint8
        )

    def _expand_worker(self, sizes: list[int]) -> None:
        """Allocate and publish the background segments in order."""
        for size in sizes:
            if self._closed:
                return
            start = time.monotonic()
            try:
                segment = self._allocate_segment(size)
            except RuntimeError:
                logger.exception(
                    "Segmented L1 expansion stopped at %d of %d bytes",
                    self._segmented.get_heap_size(),
                    self._final_size,
                )
                return
            with self._lock:
                if self._closed:
                    self._free_segment(segment.data_ptr())
                    return
                self._segments.append(segment)
                self._segmented.add_segment(size)
            logger.info(
                "Segmented L1: published a %d-byte segment in %.1f s, "
                "capacity %d of %d bytes",
                size,
                time.monotonic() - start,
                self._segmented.get_heap_size(),
                self._final_size,
            )
