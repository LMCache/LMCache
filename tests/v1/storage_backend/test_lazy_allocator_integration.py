# SPDX-License-Identifier: Apache-2.0
"""Integration tests for LazyMixedMemoryAllocator with LocalCPUBackend"""

# Standard
import unittest

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lazy_memory_allocator import LazyMixedMemoryAllocator
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class TestLazyAllocatorIntegration(unittest.TestCase):
    """Test LazyMixedMemoryAllocator integration with LocalCPUBackend"""

    def setUp(self):
        self.config = LMCacheEngineConfig.from_defaults()
        self.config.local_cpu = True
        self.config.max_local_cpu_size = 1.0
        self.config.chunk_size = 256
        self.config.enable_lazy_memory_allocator = True
        self.config.lazy_memory_initial_ratio = 0.2
        self.config.lazy_memory_expand_trigger_ratio = 0.5
        self.config.lazy_memory_step_ratio = 0.1

        self.metadata = LMCacheEngineMetadata(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            fmt="vllm",
            kv_shape=(32, 2, 256, 32, 128),
            kv_dtype=torch.float16,
        )

    def _create_backend(self, config=None):
        return LocalCPUBackend(
            config=config or self.config, metadata=self.metadata, dst_device="cpu"
        )

    def test_lazy_allocator_enabled(self):
        """Test lazy allocator is used when enabled"""
        backend = self._create_backend()
        self.assertIsInstance(backend.memory_allocator, LazyMixedMemoryAllocator)
        expected_size = int(self.config.max_local_cpu_size * 1024**3 * 0.2)
        self.assertEqual(backend.memory_allocator.initial_size, expected_size)
        backend.close()

    def test_lazy_allocator_disabled(self):
        """Test regular allocator when disabled"""
        self.config.enable_lazy_memory_allocator = False
        backend = self._create_backend()
        self.assertNotIsInstance(backend.memory_allocator, LazyMixedMemoryAllocator)
        backend.close()

    def test_memory_limit_callback(self):
        """Test memory limit callback is set"""
        backend = self._create_backend()
        allocator = backend.memory_allocator
        self.assertIsInstance(allocator, LazyMixedMemoryAllocator)
        self.assertIsNotNone(allocator.async_expander.memory_limit_callback)
        self.assertGreater(allocator.async_expander.memory_limit_callback(), 0)
        backend.close()

    def test_allocation_with_lazy_allocator(self):
        """Test basic allocation"""
        backend = self._create_backend()
        shape, dtype = torch.Size([256, 2, 4096]), torch.float16
        mem_obj = backend.allocate(shape, dtype, eviction=False, busy_loop=False)
        self.assertIsNotNone(mem_obj)
        self.assertEqual(mem_obj.meta.shape, shape)
        self.assertEqual(mem_obj.meta.dtype, dtype)
        backend.memory_allocator.free(mem_obj)
        backend.close()

    def test_config_parameters(self):
        """Test config parameters are applied"""
        cfg = LMCacheEngineConfig.from_defaults()
        cfg.local_cpu = True
        cfg.max_local_cpu_size = 2.0
        cfg.enable_lazy_memory_allocator = True
        cfg.lazy_memory_initial_ratio = 0.3
        cfg.lazy_memory_expand_trigger_ratio = 0.6
        cfg.lazy_memory_step_ratio = 0.15

        backend = self._create_backend(cfg)
        alloc = backend.memory_allocator
        self.assertEqual(alloc.initial_ratio, 0.3)
        self.assertEqual(alloc.expand_trigger_ratio, 0.6)
        self.assertEqual(alloc.step_ratio, 0.15)
        backend.close()

    def test_memcheck_before_expansion(self):
        """Test memcheck passes before any memory expansion"""
        backend = self._create_backend()
        allocator = backend.memory_allocator
        self.assertIsInstance(allocator, LazyMixedMemoryAllocator)

        # memcheck should pass with initial allocation
        self.assertTrue(allocator.pin_allocator.memcheck())
        backend.close()

    def test_memcheck_after_allocation(self):
        """Test memcheck passes after allocations"""
        backend = self._create_backend()
        allocator = backend.memory_allocator

        # Allocate some memory
        shape, dtype = torch.Size([256, 2, 4096]), torch.float16
        mem_objs = []
        for _ in range(3):
            mem_obj = backend.allocate(shape, dtype, eviction=False, busy_loop=False)
            if mem_obj:
                mem_objs.append(mem_obj)

        # memcheck should pass after allocations
        self.assertTrue(allocator.pin_allocator.memcheck())

        # Free all objects
        for mem_obj in mem_objs:
            allocator.free(mem_obj)

        # memcheck should still pass after freeing
        self.assertTrue(allocator.pin_allocator.memcheck())
        backend.close()

    def test_memcheck_after_expansion(self):
        """Test memcheck passes after memory expansion with multiple segments"""
        # Standard
        import time

        # Configure for quick expansion
        cfg = LMCacheEngineConfig.from_defaults()
        cfg.local_cpu = True
        cfg.max_local_cpu_size = 0.1  # 100MB total
        cfg.chunk_size = 256
        cfg.enable_lazy_memory_allocator = True
        cfg.lazy_memory_initial_ratio = 0.2  # Start with 20MB
        cfg.lazy_memory_expand_trigger_ratio = 0.3  # Trigger at 30%
        cfg.lazy_memory_step_ratio = 0.2  # Add 20% each step

        backend = self._create_backend(cfg)
        allocator = backend.memory_allocator
        self.assertIsInstance(allocator, LazyMixedMemoryAllocator)

        initial_segments = len(allocator.composite_buffer.segments)
        self.assertEqual(initial_segments, 1)

        # Allocate memory to trigger expansion
        shape, dtype = torch.Size([256, 2, 8192]), torch.float16
        mem_objs = []
        for _ in range(5):
            mem_obj = backend.allocate(shape, dtype, eviction=False, busy_loop=False)
            if mem_obj:
                mem_objs.append(mem_obj)

        # Wait for async expansion
        time.sleep(0.5)

        # If expansion was triggered, we should have more segments
        if allocator.expansion_triggered:
            final_segments = len(allocator.composite_buffer.segments)
            self.assertGreaterEqual(final_segments, initial_segments)

        # memcheck should pass even with multiple segments
        self.assertTrue(allocator.pin_allocator.memcheck())

        # Free all objects
        for mem_obj in mem_objs:
            allocator.free(mem_obj)

        # memcheck should pass after freeing with multiple segments
        self.assertTrue(allocator.pin_allocator.memcheck())
        backend.close()

    def test_memcheck_segment_boundary_no_coalesce(self):
        """
        Test that free blocks at segment boundaries are not
        flagged as non-coalesced.
        """
        # Standard
        import time

        cfg = LMCacheEngineConfig.from_defaults()
        cfg.local_cpu = True
        cfg.max_local_cpu_size = 0.05  # 50MB total
        cfg.chunk_size = 256
        cfg.enable_lazy_memory_allocator = True
        cfg.lazy_memory_initial_ratio = 0.3
        cfg.lazy_memory_expand_trigger_ratio = 0.4
        cfg.lazy_memory_step_ratio = 0.3

        backend = self._create_backend(cfg)
        allocator = backend.memory_allocator

        # Allocate to trigger expansion
        shape, dtype = torch.Size([256, 2, 4096]), torch.float16
        mem_objs = []
        for _ in range(3):
            mem_obj = backend.allocate(shape, dtype, eviction=False, busy_loop=False)
            if mem_obj:
                mem_objs.append(mem_obj)

        time.sleep(0.5)

        # Free all - this creates free blocks at segment boundaries
        for mem_obj in mem_objs:
            allocator.free(mem_obj)

        # memcheck should NOT report non-coalesced blocks at segment boundaries
        self.assertTrue(allocator.pin_allocator.memcheck())
        backend.close()
