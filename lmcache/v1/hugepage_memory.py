"""
Hugepage Memory Management Module for LMCache

This module provides memory allocation using Linux hugepages for better performance
and reduced TLB pressure in large memory applications.
"""

import ctypes
import threading
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import lmcache.c_ops as lmc_ops

from .memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
    MemoryFormat,
    _lmcache_nvtx_annotate,
)


class HugepageMemoryAllocator(MemoryAllocatorInterface):
    """
    Memory allocator that uses Linux hugepages for pinned memory allocation.
    
    This allocator provides better performance for large memory operations
    by reducing TLB pressure and improving memory access patterns.
    """
    
    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        Initialize the hugepage memory allocator.
        
        Args:
            size: Size of memory to allocate in bytes
            use_paging: Whether to use paged memory allocation
            **kwargs: Additional arguments for paged allocation
        """
        if not lmc_ops.is_hugepage_available():
            raise RuntimeError("Hugepages are not available on this system")
        
        self.hugepage_size = lmc_ops.get_hugepage_size()
        self.available_hugepages = lmc_ops.get_available_hugepage_count()
        
        # Check if we have enough hugepages
        required_hugepages = (size + self.hugepage_size - 1) // self.hugepage_size
        if required_hugepages > self.available_hugepages:
            raise RuntimeError(
                f"Not enough hugepages available. "
                f"Required: {required_hugepages}, Available: {self.available_hugepages}"
            )
        
        # Allocate memory using hugepages
        ptr = lmc_ops.alloc_pinned_hugepage_ptr(size, 0)
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        self.buffer = torch.frombuffer(buf, dtype=torch.uint8)
        
        self._unregistered = False
        self._ptr = ptr
        self._size = size
        
        if use_paging:
            assert "shape" in kwargs, "shape must be specified for paged memory allocator"
            assert "dtype" in kwargs, "dtype must be specified for paged memory allocator"
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            from .memory_management import PagedTensorMemoryAllocator
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            from .memory_management import TensorMemoryAllocator
            self.allocator = TensorMemoryAllocator(self.buffer)
        
        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()
    
    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, self)
    
    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(shape, dtype, batch_size, fmt, self)
    
    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)
    
    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)
    
    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()
    
    def close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            lmc_ops.free_pinned_hugepage_ptr(self._ptr, self._size)
            self._unregistered = True
    
    def get_hugepage_info(self):
        """Get information about hugepage configuration."""
        return {
            "hugepage_size": self.hugepage_size,
            "available_hugepages": self.available_hugepages,
            "allocated_size": self._size,
            "required_hugepages": (self._size + self.hugepage_size - 1) // self.hugepage_size
        }


class NumaHugepageMemoryAllocator(MemoryAllocatorInterface):
    """
    Memory allocator that uses Linux hugepages with NUMA affinity.
    
    This allocator provides better performance for large memory operations
    by reducing TLB pressure and ensuring memory is allocated on the correct NUMA node.
    """
    
    def __init__(self, size: int, numa_mapping, use_paging: bool = False, **kwargs):
        """
        Initialize the NUMA-aware hugepage memory allocator.
        
        Args:
            size: Size of memory to allocate in bytes
            numa_mapping: NUMA mapping configuration
            use_paging: Whether to use paged memory allocation
            **kwargs: Additional arguments for paged allocation
        """
        if not lmc_ops.is_hugepage_available():
            raise RuntimeError("Hugepages are not available on this system")
        
        self.numa_mapping = numa_mapping
        self.hugepage_size = lmc_ops.get_hugepage_size()
        self.available_hugepages = lmc_ops.get_available_hugepage_count()
        
        # Check if we have enough hugepages
        required_hugepages = (size + self.hugepage_size - 1) // self.hugepage_size
        if required_hugepages > self.available_hugepages:
            raise RuntimeError(
                f"Not enough hugepages available. "
                f"Required: {required_hugepages}, Available: {self.available_hugepages}"
            )
        
        # Get current GPU and map to NUMA node
        current_device_id = torch.cuda.current_device()
        gpu_to_numa_mapping = self.numa_mapping.gpu_to_numa_mapping
        assert current_device_id in gpu_to_numa_mapping, (
            f"Current device {current_device_id} is not in the GPU NUMA mapping."
        )
        numa_id = gpu_to_numa_mapping[current_device_id]
        
        # Allocate memory using NUMA-aware hugepages
        ptr = lmc_ops.alloc_pinned_numa_hugepage_ptr(size, numa_id)
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        self.buffer = torch.frombuffer(buf, dtype=torch.uint8)
        
        self._unregistered = False
        self._ptr = ptr
        self._size = size
        self._numa_id = numa_id
        
        if use_paging:
            assert "shape" in kwargs, "shape must be specified for paged memory allocator"
            assert "dtype" in kwargs, "dtype must be specified for paged memory allocator"
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            from .memory_management import PagedTensorMemoryAllocator
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            from .memory_management import TensorMemoryAllocator
            self.allocator = TensorMemoryAllocator(self.buffer)
        
        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()
    
    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shape, dtype, fmt, self)
    
    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: Union[torch.Size, Tuple[int, ...]],
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(shape, dtype, batch_size, fmt, self)
    
    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)
    
    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)
    
    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()
    
    def close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            lmc_ops.free_pinned_numa_hugepage_ptr(self._ptr, self._size)
            self._unregistered = True
    
    def get_hugepage_info(self):
        """Get information about hugepage configuration and NUMA binding."""
        return {
            "hugepage_size": self.hugepage_size,
            "available_hugepages": self.available_hugepages,
            "allocated_size": self._size,
            "required_hugepages": (self._size + self.hugepage_size - 1) // self.hugepage_size,
            "numa_id": self._numa_id
        }


def get_hugepage_info():
    """Get system-wide hugepage information."""
    if not lmc_ops.is_hugepage_available():
        return {"available": False}
    
    return {
        "available": True,
        "hugepage_size": lmc_ops.get_hugepage_size(),
        "available_count": lmc_ops.get_available_hugepage_count()
    }


def create_hugepage_allocator(size: int, numa_mapping=None, use_paging: bool = False, **kwargs):
    """
    Factory function to create appropriate hugepage allocator.
    
    Args:
        size: Size of memory to allocate in bytes
        numa_mapping: Optional NUMA mapping configuration
        use_paging: Whether to use paged memory allocation
        **kwargs: Additional arguments for the allocator
    
    Returns:
        HugepageMemoryAllocator or NumaHugepageMemoryAllocator instance
    """
    if numa_mapping:
        return NumaHugepageMemoryAllocator(size, numa_mapping, use_paging, **kwargs)
    else:
        return HugepageMemoryAllocator(size, use_paging, **kwargs) 