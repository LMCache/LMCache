# SPDX-License-Identifier: Apache-2.0
"""L1 allocator over several independently allocated host memory segments.

Some page-locked host memory sources cap the size of one allocation (a ROCm
GTT ``hipHostMalloc`` must stay below 512 GiB), so a large L1 is built from
several segments whose virtual addresses are not contiguous. The allocator
keeps one logical address space for metadata (``MemoryObjMetadata.address``
is unique across segments) and guarantees that every allocated block lies
inside exactly one segment; no tensor ever spans two segments.
"""

# Standard
from dataclasses import dataclass
from typing import Callable, Union
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
from lmcache.utils import get_size_bytes
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    FreeBlock,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
)

logger = init_logger(__name__)


def _split_capacity(total: int, segment_size: int) -> list[int]:
    """Split ``total`` bytes into ``segment_size`` pieces plus a remainder."""
    pieces = [segment_size] * (total // segment_size)
    if total % segment_size:
        pieces.append(total % segment_size)
    return pieces


def plan_segments(
    init_size: int, final_size: int, segment_size: int, granule: int
) -> tuple[list[int], list[int]]:
    """Plan the segment sizes of a segmented L1.

    The initial capacity is split into ``segment_size`` pieces (the last one
    holds the remainder); the capacity between initial and final size is
    split the same way.

    Args:
        init_size: Bytes that must be usable when the constructor returns.
        final_size: Total configured bytes.
        segment_size: Maximum bytes per segment, a multiple of ``granule``.
        granule: Size granularity; both sizes are rounded up to it.

    Returns:
        ``(initial_segments, expansion_segments)`` as lists of byte sizes,
        each a positive multiple of ``granule`` and at most ``segment_size``.
    """
    final = -(-final_size // granule) * granule
    init = min(-(-init_size // granule) * granule, final)
    return (
        _split_capacity(init, segment_size),
        _split_capacity(final - init, segment_size),
    )


@dataclass
class _Segment:
    """One published host memory segment."""

    ptr: int
    """Host pointer returned by the segment allocation function."""

    size: int
    """Size of the segment in bytes."""

    buffer: torch.Tensor
    """Flat ``torch.uint8`` CPU view of ``[ptr, ptr + size)``."""


class _SegmentedAddressManager(AddressManager):
    """An :class:`AddressManager` whose free blocks never span two segments.

    Segment ``k`` owns the logical range ``[base_k, base_k + size_k)`` with
    ``base_k`` the sum of the sizes of all earlier segments. Free blocks are
    only coalesced within a segment, so ``allocate`` and ``batched_allocate``
    can only return blocks that lie inside one segment.

    Thread safety: segment bases are only appended; a new base lies beyond all
    existing blocks, so concurrent ``free`` calls resolve segments correctly.
    """

    def __init__(self, segment_sizes: list[int], align_bytes: int) -> None:
        """
        Args:
            segment_sizes: Sizes of the initial segments, in logical order.
                Each must be a positive multiple of ``align_bytes``.
            align_bytes: The alignment requirement for allocations.
        """
        super().__init__(segment_sizes[0], align_bytes)
        self._segment_bases: list[int] = [0]
        for size in segment_sizes[1:]:
            self.add_segment(size)

    def add_segment(self, size: int) -> None:
        """Append a segment of ``size`` bytes at the end of the address space.

        Args:
            size: Segment size, a positive multiple of the alignment.
        """
        self._segment_bases.append(self.get_heap_size())
        self.sbrk(size)

    def segment_index(self, address: int) -> int:
        """Return the index of the segment containing logical ``address``."""
        return bisect.bisect_right(self._segment_bases, address) - 1

    def segment_base(self, index: int) -> int:
        """Return the first logical address of segment ``index``."""
        return self._segment_bases[index]

    def free(self, address: int, size: int) -> None:
        """Free a logical range, splitting it at segment boundaries.

        ``TensorMemoryAllocator.batched_free`` coalesces logically adjacent
        objects, which may belong to two different segments.

        Args:
            address: Start of the range to free.
            size: Size of the range in bytes.
        """
        end = address + size
        while address < end:
            index = self.segment_index(address)
            piece_end = min(end, self._segment_end(index))
            super().free(address, piece_end - address)
            address = piece_end

    def check_consistency(self) -> bool:
        """Check coalescing within segments, boundaries and the size total.

        Returns:
            True if consistent, False otherwise.
        """
        blocks = list(self._explicit_list)
        for block in blocks:
            index = self.segment_index(block.start)
            if block.start + block.size > self._segment_end(index):
                return False
        for prev, succ in zip(blocks[:-1], blocks[1:], strict=False):
            if self._can_merge_with_succ(prev, succ):
                return False
        total_free_size = sum(block.size for block in blocks)
        return total_free_size + self.total_allocated_size == self.get_heap_size()

    def _segment_end(self, index: int) -> int:
        if index + 1 < len(self._segment_bases):
            return self._segment_bases[index + 1]
        return self.get_heap_size()

    def _same_segment(self, first: FreeBlock, second: FreeBlock) -> bool:
        return self.segment_index(first.start) == self.segment_index(second.start)

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        return super()._can_merge_with_prev(
            curr_block, prev_block
        ) and self._same_segment(prev_block, curr_block)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        return super()._can_merge_with_succ(
            curr_block, succ_block
        ) and self._same_segment(curr_block, succ_block)


class SegmentedMemoryAllocator(TensorMemoryAllocator):
    """Lazily expanded L1 allocator over non-contiguous host memory segments.

    The constructor allocates the initial capacity synchronously; a daemon
    thread allocates the rest segment by segment and publishes each segment
    only after it is fully usable. Allocation, batching (all-or-nothing),
    freeing, coalescing and accounting are those of
    :class:`TensorMemoryAllocator` over a :class:`_SegmentedAddressManager`.

    Thread safety: allocation and free are as thread-safe as
    :class:`TensorMemoryAllocator`; ``_lock`` serializes segment publication
    with ``close``.
    """

    EXPANSION_RETRY_DELAYS_S = (10.0, 60.0)
    """Back-off before each retry of a failed background segment allocation."""

    def __init__(
        self,
        init_size: int,
        final_size: int,
        segment_size: int,
        alloc_segment: Callable[[int], int],
        free_segment: Callable[[int], None],
        align_bytes: int = AddressManager.ALIGN_BYTES,
        name: str = "segmented",
    ) -> None:
        """
        Args:
            init_size: Bytes allocated before the constructor returns.
            final_size: Total configured bytes, reached by background
                expansion.
            segment_size: Maximum bytes per segment; a positive multiple of
                ``max(align_bytes, page size)``.
            alloc_segment: Allocates one segment of the given size and returns
                its host pointer, or raises. Must return memory aligned to
                ``align_bytes`` that stays valid until ``free_segment``.
            free_segment: Frees a pointer returned by ``alloc_segment``.
            align_bytes: Allocation alignment, a positive power of two.
            name: Label of the memory source used in log messages.

        Raises:
            ValueError: If a size or the alignment is invalid.
            RuntimeError: If an initial segment cannot be allocated or is
                misaligned; segments allocated so far are freed first.
        """
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")
        granule = max(align_bytes, os.sysconf("SC_PAGE_SIZE"))
        if segment_size <= 0 or segment_size % granule != 0:
            raise ValueError(
                f"segment_size must be a positive multiple of {granule} bytes"
            )
        if init_size <= 0 or final_size <= 0:
            raise ValueError("init_size and final_size must be positive")

        self._name = name
        self._align = align_bytes
        self._alloc_segment = alloc_segment
        self._free_segment = free_segment
        initial_sizes, expansion_sizes = plan_segments(
            init_size, final_size, segment_size, granule
        )
        self._final_size = sum(initial_sizes) + sum(expansion_sizes)
        """Total bytes once every planned segment is published."""
        self._max_segment_size = max(initial_sizes + expansion_sizes)
        """Largest segment that will ever exist; bounds a single object."""
        self._segments: list[_Segment] = []
        """Published segments in logical order; appended before their address
        range is added to the address manager."""
        self._lock = threading.Lock()
        self._closed = False

        start = time.monotonic()
        try:
            for size in initial_sizes:
                self._segments.append(self._allocate_segment(size))
        except BaseException:
            self._release(self._segments)
            raise
        init_seconds = time.monotonic() - start

        super().__init__(self._segments[0].buffer, align_bytes=align_bytes)
        self.address_manager = _SegmentedAddressManager(initial_sizes, align_bytes)
        # There is no contiguous buffer; slices come from the segment views.
        self.buffer = torch.empty(0, dtype=torch.uint8)

        logger.info(
            "%s L1: configured %d bytes, segment size %d bytes, initial %d "
            "bytes in %d segment(s) allocated in %.1f s, background expansion "
            "%d bytes in %d segment(s)",
            name,
            self._final_size,
            segment_size,
            sum(initial_sizes),
            len(initial_sizes),
            init_seconds,
            sum(expansion_sizes),
            len(expansion_sizes),
        )

        self._stop_expand = threading.Event()
        self._expand_thread = threading.Thread(
            target=self._expand_worker,
            args=(expansion_sizes,),
            daemon=True,
            name="segmented-l1-expand",
        )
        if expansion_sizes:
            self._expand_thread.start()

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
    ) -> TensorMemoryObj | None:
        """Allocate one object inside a single segment.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional parent allocator identifier.

        Returns:
            The memory object, or ``None`` if no published segment has room.

        Raises:
            ValueError: If the object is larger than the largest segment, so
                it can never be allocated.
        """
        self._check_fits_one_segment(shapes, dtypes)
        return super().allocate(shapes, dtypes, fmt, allocator_type)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
    ) -> list[MemoryObj] | None:
        """Allocate ``batch_size`` equal objects, all or nothing.

        Objects may land in different segments; each one lies inside a single
        segment.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in each object's metadata.
            allocator_type: Optional parent allocator identifier.

        Returns:
            The memory objects, or ``None`` (nothing allocated) if the
            published segments cannot hold the whole batch.

        Raises:
            ValueError: If one object is larger than the largest segment.
        """
        self._check_fits_one_segment(shapes, dtypes)
        objs = super().batched_allocate(shapes, dtypes, batch_size, fmt, allocator_type)
        return None if objs is None else list[MemoryObj](objs)

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used bytes, published capacity in bytes)``.

        Like the lazy allocator, the capacity grows as segments are published.
        """
        total = self.address_manager.get_heap_size()
        return total - self.address_manager.get_free_size(), total

    def segments(self) -> list[tuple[int, int]]:
        """Return ``(host pointer, size)`` of each published segment.

        Returns:
            One entry per segment in logical address order.
        """
        with self._lock:
            return [(segment.ptr, segment.size) for segment in self._segments]

    def memory_region_count(self) -> int:
        """Return the number of published segments."""
        with self._lock:
            return len(self._segments)

    def close(self) -> None:
        """Stop expansion and free every segment exactly once.

        Waits for an in-flight segment allocation to finish (it cannot be
        interrupted), synchronizes the device so no queued transfer outlives a
        mapping, then frees the segments. Idempotent. Objects still allocated
        become invalid; callers free them first.

        Raises:
            RuntimeError: The first error raised by ``free_segment``, after all
                segments were attempted.
        """
        with self._lock:
            if self._closed:
                return
        self._stop_expand.set()
        if self._expand_thread.ident is not None:
            self._expand_thread.join()
        if self.num_active_allocations:
            logger.warning(
                "%s L1 closed with %d active allocations",
                self._name,
                self.num_active_allocations,
            )
        if torch_dev.is_available():
            torch_dev.synchronize()
        with self._lock:
            self._closed = True
            segments, self._segments = self._segments, []
            self.address_manager = AddressManager(0, self._align)
        self._release(segments)

    def __str__(self) -> str:
        """Return the allocator name."""
        return "SegmentedMemoryAllocator"

    # Helper functions
    def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
        address_manager = self.address_manager
        assert isinstance(address_manager, _SegmentedAddressManager)
        index = address_manager.segment_index(start)
        offset = start - address_manager.segment_base(index)
        segment = self._segments[index]
        assert offset + size <= segment.size, "block crosses a segment boundary"
        return segment.buffer[offset : offset + size]

    def _check_fits_one_segment(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> None:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size = self.address_manager.compute_aligned_size(get_size_bytes(shapes, dtypes))
        if size > self._max_segment_size:
            raise ValueError(
                f"An L1 object of {size} bytes cannot fit in one {self._name} "
                f"segment (largest segment: {self._max_segment_size} bytes)"
            )

    def _allocate_segment(self, size: int) -> _Segment:
        """Allocate and wrap one segment; frees it again if wrapping fails."""
        ptr = self._alloc_segment(size)
        try:
            if ptr % self._align != 0:
                raise RuntimeError(
                    f"{self._name} segment at {ptr:#x} is not aligned to "
                    f"{self._align} bytes"
                )
            buffer = torch.frombuffer(
                (ctypes.c_uint8 * size).from_address(ptr), dtype=torch.uint8
            )
        except BaseException:
            self._free_segment(ptr)
            raise
        return _Segment(ptr=ptr, size=size, buffer=buffer)

    def _release(self, segments: list[_Segment]) -> None:
        """Free ``segments``; re-raise the first failure after trying all."""
        errors: list[RuntimeError] = []
        for segment in segments:
            segment.buffer = torch.empty(0, dtype=torch.uint8)
            try:
                self._free_segment(segment.ptr)
            except RuntimeError as e:
                logger.exception(
                    "Failed to free %s segment at %#x", self._name, segment.ptr
                )
                errors.append(e)
        if errors:
            raise errors[0]

    def _allocate_with_retry(self, size: int) -> _Segment | None:
        """Allocate one background segment with bounded retries.

        Returns:
            The segment, or ``None`` if every attempt failed or close was
            requested.
        """
        delays = (0.0, *self.EXPANSION_RETRY_DELAYS_S)
        for attempt, delay in enumerate(delays):
            if self._stop_expand.wait(delay):
                return None
            start = time.monotonic()
            try:
                segment = self._allocate_segment(size)
            except Exception as e:
                published, _ = self._published_and_final()
                if attempt == 0:
                    logger.error(
                        "%s L1: allocating a %d-byte segment failed: %s "
                        "(usable %d of %d configured bytes); retrying up to "
                        "%d times",
                        self._name,
                        size,
                        e,
                        published,
                        self._final_size,
                        len(self.EXPANSION_RETRY_DELAYS_S),
                    )
                else:
                    logger.debug(
                        "%s L1: segment retry %d failed: %s", self._name, attempt, e
                    )
                continue
            logger.info(
                "%s L1: allocated a %d-byte segment in %.1f s",
                self._name,
                size,
                time.monotonic() - start,
            )
            return segment
        return None

    def _published_and_final(self) -> tuple[int, int]:
        return self.address_manager.get_heap_size(), self._final_size

    def _expand_worker(self, sizes: list[int]) -> None:
        """Allocate and publish the background segments in order."""
        for size in sizes:
            segment = self._allocate_with_retry(size)
            if segment is None:
                if not self._stop_expand.is_set():
                    published, final = self._published_and_final()
                    logger.error(
                        "%s L1 expansion stopped: %d of %d configured bytes "
                        "usable in %d segment(s)",
                        self._name,
                        published,
                        final,
                        self.memory_region_count(),
                    )
                return
            with self._lock:
                if self._closed:
                    self._release([segment])
                    return
                self._segments.append(segment)
                address_manager = self.address_manager
                assert isinstance(address_manager, _SegmentedAddressManager)
                address_manager.add_segment(segment.size)
                published = address_manager.get_heap_size()
            logger.info(
                "%s L1: published segment %d (%d bytes), capacity %d / %d bytes",
                self._name,
                len(self._segments) - 1,
                segment.size,
                published,
                self._final_size,
            )
