# SPDX-License-Identifier: Apache-2.0
"""Lazy memory allocator with async progressive expansion and zero-copy."""

# Standard
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
import threading

# Third Party
import sortedcontainers
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    BufferAllocator,
    FreeBlock,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    TensorMemoryAllocator,
    _allocate_cpu_memory,
)
from lmcache.v1.system_detection import NUMAMapping

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops
else:
    # First Party
    import lmcache.non_cuda_equivalents as lmc_ops

logger = init_logger(__name__)


class CompositeBuffer:
    """Manages multiple memory segments with unified view (zero-copy)."""

    def __init__(self, initial_buffer: torch.Tensor):
        self.segments: List[torch.Tensor] = [initial_buffer]
        self.segment_offsets: List[int] = [0]
        self.total_size = initial_buffer.numel()
        self.lock = threading.Lock()

    def add_segment(self, new_buffer: torch.Tensor) -> int:
        """Add a new memory segment to the composite buffer.

        Thread-safe: Protected by self.lock.

        Returns:
            int: The offset of the new segment in the unified address space.
        """
        with self.lock:
            offset = self.total_size
            self.segments.append(new_buffer)
            self.segment_offsets.append(offset)
            self.total_size += new_buffer.numel()
            logger.info(
                f"Added segment: {new_buffer.numel()} bytes, total: {self.total_size}"
            )
            return offset

    def get_slice(self, start: int, size: int) -> torch.Tensor:
        with self.lock:
            segment_idx = self._find_segment(start)
            if segment_idx == -1:
                raise ValueError(f"Invalid offset: {start}")

            segment_start = self.segment_offsets[segment_idx]
            segment = self.segments[segment_idx]
            end = start + size

            if end <= segment_start + segment.numel():
                local_start = start - segment_start
                return segment[local_start : local_start + size]
            else:
                raise ValueError(
                    f"Slice spans segments (start={start}, size={size}). "
                    "Bug in segment-aware coalescing."
                )

    def _find_segment(self, offset: int) -> int:
        for i in range(len(self.segments) - 1, -1, -1):
            if offset >= self.segment_offsets[i]:
                if offset < self.segment_offsets[i] + self.segments[i].numel():
                    return i
        return -1

    def numel(self) -> int:
        return self.total_size


class CompositeTensorMemoryAllocator(TensorMemoryAllocator):
    """TensorMemoryAllocator with segment-aware coalescing for CompositeBuffer."""

    def __init__(
        self,
        composite_buffer: CompositeBuffer,
        align_bytes: int = TensorMemoryAllocator.ALIGN_BYTES,
    ):
        self.composite_buffer = composite_buffer
        self.buffer = composite_buffer.segments[0].view(torch.uint8).flatten()
        self.align_bytes = align_bytes
        self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)
        self.explicit_list.add(FreeBlock(start=0, size=self.buffer.numel()))
        self.num_active_allocations = 0
        self.total_allocated_size = 0
        self.segment_boundaries = [composite_buffer.segments[0].numel()]
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    def expand_with_new_segment(self, new_buffer: torch.Tensor):
        """Expand the allocator with a new memory segment.

        Thread Safety:
        ==============
        This method modifies shared data structures (explicit_list, segment_boundaries)
        that are also accessed by allocate/free operations in the main thread.

        The caller MUST hold the host_mem_lock before calling this method to prevent
        race conditions.
        """
        offset = self.composite_buffer.add_segment(new_buffer)
        new_size = new_buffer.numel()
        self.segment_boundaries.append(offset + new_size)

        new_free_block = FreeBlock(start=offset, size=new_size)
        prev_block = self.explicit_list[-1] if len(self.explicit_list) > 0 else None
        succ_block = None

        if not self._coalesce(new_free_block, prev_block, succ_block):
            self.explicit_list.add(new_free_block)

        logger.info(
            f"Expanded: {new_size} bytes, total: {self.composite_buffer.numel()}"
        )

    def _is_segment_boundary(self, offset: int) -> bool:
        return offset in self.segment_boundaries

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        """Override: Add segment boundary check for prev merge."""
        return super()._can_merge_with_prev(
            curr_block, prev_block
        ) and not self._is_segment_boundary(prev_block.start + prev_block.size)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        """Override: Add segment boundary check for succ merge."""
        return super()._can_merge_with_succ(
            curr_block, succ_block
        ) and not self._is_segment_boundary(curr_block.start + curr_block.size)

    def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
        """Override: Use composite buffer for multi-segment access."""
        return self.composite_buffer.get_slice(start, size)


class AsyncMemoryExpander:
    """Asynchronously expands memory in background.

    Design Philosophy:
    ==================
    This is a ONE-WAY, EXPANSION-ONLY mechanism designed to:
    1. Reduce startup latency: Start with a small initial allocation
    2. Minimize initial memory footprint: Avoid allocating full capacity upfront
    3. Progressive growth: Expand memory as needed in the background

    Key Characteristics:
    - NO SHRINKING: Once memory is allocated, it is never released back to
      the system
    - ONE-TIME EXPANSION: The expander thread runs until target size is
      reached, then stops
    - LAZY ALLOCATION: Memory is allocated progressively, not all at once

    This design is optimal for workloads with monotonically increasing memory
    needs, where the memory will eventually be fully utilized and doesn't need
    to be reclaimed.

    Thread Safety Overview:
    =======================
    This class manages a background daemon thread (_expansion_worker) that
    progressively allocates and adds new memory segments to the allocator.

    Concurrency Model:
    - Main thread: Performs allocate/free operations on the allocator
    - Expander thread: Adds new memory segments via expand_with_new_segment()
    """

    def __init__(
        self,
        composite_buffer: CompositeBuffer,
        allocator: CompositeTensorMemoryAllocator,
        total_size: int,
        step_ratio: float,
        host_mem_lock: threading.Lock,
        numa_mapping: Optional[NUMAMapping] = None,
        memory_limit_callback=None,
    ):
        self.composite_buffer = composite_buffer
        self.allocator = allocator
        self.total_size = total_size
        self.step_ratio = step_ratio
        self.numa_mapping = numa_mapping
        self.memory_limit_callback = memory_limit_callback
        self.host_mem_lock = host_mem_lock
        self.expansion_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.expansion_lock = threading.Lock()
        self.is_expanding = False

    def start_expansion(self):
        with self.expansion_lock:
            if self.is_expanding:
                return
            self.is_expanding = True
            self.stop_flag.clear()
            self.expansion_thread = threading.Thread(
                target=self._expansion_worker, daemon=True, name="MemoryExpander"
            )
            self.expansion_thread.start()
            logger.info("Started async expansion")

    def _get_effective_limit(self, current_size: int) -> Optional[int]:
        """Calculate the effective memory limit based on callback.

        Args:
            current_size: Current allocated memory size in bytes

        Returns:
            Effective memory limit in bytes, or None if expansion should stop
        """
        if not self.memory_limit_callback:
            return self.total_size

        try:
            limit_bytes = self.memory_limit_callback()
            if limit_bytes <= 0:
                return self.total_size

            effective_limit = min(self.total_size, limit_bytes)
            if current_size >= effective_limit:
                logger.warning(
                    f"Expansion stopped: {current_size} >= {effective_limit}"
                )
                return None

            return effective_limit
        except Exception as e:
            logger.warning(f"Memory limit callback failed: {e}")
            return self.total_size

    def _expansion_worker(self):
        """Background worker that progressively expands memory to target size.

        Runs in daemon thread. Allocates memory in steps (step_ratio at a time)
        until total_size is reached or memory limit is hit. Never shrinks.
        """
        try:
            current_size = self.composite_buffer.numel()
            while current_size < self.total_size and not self.stop_flag.is_set():
                effective_limit = self._get_effective_limit(current_size)
                if effective_limit is None:
                    break

                next_size = min(
                    int(self.total_size * self.step_ratio),
                    effective_limit - current_size,
                )
                if next_size <= 0:
                    break

                logger.info(
                    f"Expanding: +{next_size}, current={current_size}, "
                    f"target={self.total_size}"
                )

                try:
                    new_buffer = _allocate_cpu_memory(next_size, self.numa_mapping)
                except Exception as e:
                    logger.error(f"Allocation failed: {e}")
                    break

                with self.host_mem_lock:
                    self.allocator.expand_with_new_segment(new_buffer)

                current_size += next_size

            logger.info(f"Expansion completed: {self.composite_buffer.numel()} bytes")
        except Exception as e:
            logger.error(f"Expansion error: {e}", exc_info=True)
        finally:
            with self.expansion_lock:
                self.is_expanding = False

    def stop(self):
        self.stop_flag.set()
        if self.expansion_thread and self.expansion_thread.is_alive():
            self.expansion_thread.join(timeout=5.0)


class LazyMixedMemoryAllocator(MixedMemoryAllocator):
    """Lazy allocator: starts small, expands async when needed (zero-copy).

    Starts with initial_ratio of target size, triggers one-time background
    expansion when usage exceeds expand_trigger_ratio. Ideal for fast startup
    with low initial memory footprint.

    See AsyncMemoryExpander for detailed design philosophy.
    """

    def __init__(
        self,
        size: int,
        config: "LMCacheEngineConfig",
        use_paging: bool = False,
        memory_limit_callback: Optional[Callable] = None,
        **kwargs,
    ):
        # Extract configuration values from config
        initial_ratio = config.lazy_memory_initial_ratio
        expand_trigger_ratio = config.lazy_memory_expand_trigger_ratio
        step_ratio = config.lazy_memory_step_ratio

        self.total_size = size
        self.initial_ratio = initial_ratio
        self.expand_trigger_ratio = expand_trigger_ratio
        self.step_ratio = step_ratio
        self.memory_limit_callback = memory_limit_callback
        self.expansion_triggered = False
        self.initial_size = int(size * initial_ratio)
        self.numa_mapping = kwargs.get("numa_mapping", None)
        self.size = self.initial_size
        self._unregistered = False
        self.async_expander: Optional[AsyncMemoryExpander]

        if not use_paging:
            initial_buffer = _allocate_cpu_memory(self.initial_size, self.numa_mapping)
            self.composite_buffer = CompositeBuffer(initial_buffer)
            self.buffer = initial_buffer
            self.pin_allocator = CompositeTensorMemoryAllocator(self.composite_buffer)
            self.align_bytes = self.pin_allocator.align_bytes
            self.host_mem_lock = threading.Lock()
            self.buffer_allocator = BufferAllocator("cpu")
            self.async_expander = AsyncMemoryExpander(
                self.composite_buffer,
                self.pin_allocator,
                self.total_size,
                self.step_ratio,
                self.host_mem_lock,
                self.numa_mapping,
                self.memory_limit_callback,
            )
        else:
            logger.warning(
                "Paged allocation with lazy expansion not fully supported. "
                "Using initial size only."
            )
            super().__init__(self.initial_size, use_paging, **kwargs)
            self.async_expander = None

        logger.info(
            f"LazyAllocator: initial={self.initial_size}B "
            f"({initial_ratio * 100:.0f}%), target={self.total_size}B, "
            f"trigger={expand_trigger_ratio * 100:.0f}%, "
            f"step={step_ratio * 100:.0f}%"
        )

    def _check_and_trigger_expansion(self):
        if self.expansion_triggered or not self.async_expander:
            return
        if not isinstance(self.pin_allocator, CompositeTensorMemoryAllocator):
            return

        usage_ratio = (
            self.pin_allocator.total_allocated_size / self.composite_buffer.numel()
        )
        if usage_ratio >= self.expand_trigger_ratio:
            logger.info(
                f"Triggering expansion: usage={usage_ratio * 100:.0f}%, "
                f"threshold={self.expand_trigger_ratio * 100:.0f}%"
            )
            self.async_expander.start_expansion()
            self.expansion_triggered = True

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, Tuple[int, ...], list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        result = super().allocate(shapes, dtypes, fmt, allocator_type)
        if result and fmt != MemoryFormat.BINARY_BUFFER:
            self._check_and_trigger_expansion()
        return result

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, Tuple[int, ...], list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        result = super().batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )
        if result and fmt != MemoryFormat.BINARY_BUFFER:
            self._check_and_trigger_expansion()
        return result

    def close(self):
        if hasattr(self, "async_expander") and self.async_expander:
            self.async_expander.stop()

        if not self._unregistered:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if hasattr(self, "composite_buffer"):
                for segment in self.composite_buffer.segments:
                    ptr = segment.data_ptr()
                    if self.numa_mapping:
                        lmc_ops.free_pinned_numa_ptr(ptr, segment.numel())
                    else:
                        lmc_ops.free_pinned_ptr(ptr)
                logger.info("LazyMixedMemoryAllocator closed and memory freed")
            else:
                # Fall back to parent's close for paging mode
                super().close()
                return
            self._unregistered = True

    def __str__(self):
        return "LazyMixedMemoryAllocator"
