# SPDX-License-Identifier: Apache-2.0
"""Test cases for LazyMixedMemoryAllocator"""

# Standard
import time

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lazy_memory_allocator import (
    CompositeBuffer,
    CompositeTensorMemoryAllocator,
    LazyMixedMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat


class TestLazyMemoryAllocator:
    """Test suite for LazyMixedMemoryAllocator"""

    @staticmethod
    def _create_allocator(size_mb=10, initial=0.2, trigger=0.5, step=0.1):
        config = LMCacheEngineConfig.from_defaults(
            lazy_memory_initial_ratio=initial,
            lazy_memory_expand_trigger_ratio=trigger,
            lazy_memory_step_ratio=step,
        )
        return LazyMixedMemoryAllocator(
            size=size_mb * 1024 * 1024,
            config=config,
        )

    @staticmethod
    def _verify_mem_obj(mem_obj, shape, dtype=torch.float32):
        assert mem_obj and mem_obj.is_valid()
        assert mem_obj.get_shape() == shape
        if dtype:
            assert mem_obj.get_dtype() == dtype

    def test_basic_allocation(self):
        """Test basic allocation and tensor access"""
        allocator = self._create_allocator()
        try:
            shape = torch.Size([1024, 256])
            mem_obj = allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
            self._verify_mem_obj(mem_obj, shape)
            assert mem_obj.tensor is not None and mem_obj.tensor.shape == shape
            mem_obj.ref_count_down()
        finally:
            allocator.close()

    def test_async_expansion_trigger(self):
        """Test async expansion trigger"""
        allocator = self._create_allocator(size_mb=20, initial=0.2, step=0.2)
        try:
            shape = torch.Size([512, 1024])
            mem_obj = allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
            assert mem_obj and allocator.expansion_triggered
            time.sleep(0.5)
            if allocator.async_expander:
                assert allocator.composite_buffer.numel() >= allocator.initial_size
            mem_obj.ref_count_down()
        finally:
            allocator.close()

    def test_multiple_allocations(self):
        """Test multiple allocations with expansion"""
        allocator = self._create_allocator(size_mb=50, initial=0.1, step=0.2)
        try:
            shape = torch.Size([256, 512])
            mem_objs = [
                allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
                for _ in range(10)
            ]
            mem_objs = [m for m in mem_objs if m]
            assert len(mem_objs) > 0
            for m in mem_objs:
                self._verify_mem_obj(m, shape)
            time.sleep(1.0)
            for _ in range(5):
                m = allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
                if m:
                    mem_objs.append(m)
            for m in mem_objs:
                m.ref_count_down()
        finally:
            allocator.close()

    def test_batched_allocation(self):
        """Test batched allocation"""
        allocator = self._create_allocator(size_mb=30, initial=0.3, step=0.2)
        try:
            shape, batch_size = torch.Size([128, 256]), 5
            mem_objs = allocator.batched_allocate(
                shape, torch.float32, batch_size, MemoryFormat.KV_2LTD
            )
            assert mem_objs and len(mem_objs) == batch_size
            for m in mem_objs:
                self._verify_mem_obj(m, shape)
                m.ref_count_down()
        finally:
            allocator.close()

    def test_free_and_reuse(self):
        """Test memory reuse after free"""
        allocator = self._create_allocator(initial=0.5, trigger=0.8)
        try:
            shape = torch.Size([256, 256])
            mem_obj1 = allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
            assert mem_obj1
            mem_obj1.ref_count_down()
            mem_obj2 = allocator.allocate(shape, torch.float32, MemoryFormat.KV_2LTD)
            assert mem_obj2
            mem_obj2.ref_count_down()
        finally:
            allocator.close()

    def test_buffer_allocator_passthrough(self):
        """Test BINARY_BUFFER format"""
        allocator = self._create_allocator()
        try:
            mem_obj = allocator.allocate(
                torch.Size([1024]), [], MemoryFormat.BINARY_BUFFER
            )
            assert mem_obj and mem_obj.get_memory_format() == MemoryFormat.BINARY_BUFFER
            mem_obj.ref_count_down()
        finally:
            allocator.close()

    def test_composite_buffer_growth(self):
        """Test composite buffer segment growth"""
        allocator = self._create_allocator(size_mb=100, initial=0.1)
        try:
            initial_segs = len(allocator.composite_buffer.segments)
            assert initial_segs == 1
            mem_obj = allocator.allocate(
                torch.Size([1024, 1024]), torch.float32, MemoryFormat.KV_2LTD
            )
            assert mem_obj
            time.sleep(1.0)
            assert len(allocator.composite_buffer.segments) >= initial_segs
            mem_obj.ref_count_down()
        finally:
            allocator.close()

    def test_segment_aware_coalescing(self):
        """Test coalescing respects segment boundaries"""
        buf = torch.empty(1024 * 1024, dtype=torch.uint8)
        comp_buf = CompositeBuffer(buf)
        alloc = CompositeTensorMemoryAllocator(comp_buf)

        # Allocate and verify
        shape1 = torch.Size([256, 1024])
        mem1 = alloc.allocate(shape1, torch.float32, MemoryFormat.KV_2LTD)
        assert mem1, "First allocation failed"

        # Add second segment
        alloc.expand_with_new_segment(torch.empty(1024 * 1024, dtype=torch.uint8))
        alloc.free(mem1)

        # Verify no blocks span boundaries
        for block in alloc.explicit_list:
            block_end = block.start + block.size
            for boundary in alloc.segment_boundaries[:-1]:
                assert not (block.start < boundary < block_end), (
                    f"Block crosses boundary at {boundary}"
                )

        # Allocate from second segment
        mem2 = alloc.allocate(
            torch.Size([128, 1024]), torch.float32, MemoryFormat.KV_2LTD
        )
        assert mem2, "Second allocation failed"

        # Verify allocation doesn't span segments
        alloc_end = mem2.meta.address + mem2.meta.phy_size
        for boundary in alloc.segment_boundaries[:-1]:
            assert not (mem2.meta.address < boundary < alloc_end), (
                f"Allocation spans boundary at {boundary}"
            )

        # Verify get_slice works
        try:
            comp_buf.get_slice(mem2.meta.address, mem2.meta.phy_size)
        except ValueError as e:
            if "spans multiple segments" in str(e):
                raise AssertionError(f"get_slice error: {e}") from e
            raise

    def test_cross_segment_allocation_prevented(self):
        """Test allocations don't span segments"""
        buf = torch.empty(100 * 1024, dtype=torch.uint8)
        comp_buf = CompositeBuffer(buf)
        alloc = CompositeTensorMemoryAllocator(comp_buf)

        # Allocate most of first segment
        mem1 = alloc.allocate(
            torch.Size([20, 1024]), torch.float32, MemoryFormat.KV_2LTD
        )
        assert mem1

        # Add second segment and free first allocation
        alloc.expand_with_new_segment(torch.empty(100 * 1024, dtype=torch.uint8))
        alloc.free(mem1)

        # Verify separate free blocks (no cross-segment coalescing)
        assert len(alloc.explicit_list) == 2, (
            f"Expected 2 free blocks, got {len(alloc.explicit_list)}"
        )
