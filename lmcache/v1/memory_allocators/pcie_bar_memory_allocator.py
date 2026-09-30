# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from typing import Union
import ctypes
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)


class PcieBarTensorMemoryAllocator(TensorMemoryAllocator):
    """Thin subclass of TensorMemoryAllocator that carries the BAR sentinel.

    TensorMemoryObj.parent() returns this allocator, so gpu_ops._is_bar_backed()
    correctly identifies the object as BAR-backed via the is_pcie_bar_memory flag.
    """

    is_pcie_bar_memory: bool = True


class PcieBarMemoryAllocator(MemoryAllocatorInterface):
    """KV-cache buffer backed by a PCIe BAR memory region (e.g. CXL DDR,
    NVMe CMB, FPGA SRAM).

    Initialization (once at startup):
      1. Open /sys/bus/pci/devices/<bdf>/resourceN via open(O_RDWR|O_SYNC).
      2. mmap(MAP_SHARED) -> CPU VA window into the BAR physical address.
      3. cudaHostRegister(cudaHostRegisterMapped | cudaHostRegisterIoMemory)
         -> CUDA driver maps the BAR PA into the GPU UVA space.
      4. cudaHostGetDevicePointer() (inside get_kernel_ptr in the CUDA kernels)
         returns the GPU VA that aliases the BAR PA.

    Hot path (GPU -> BAR, D2H):
      multi_layer_kv_transfer / single_layer_kv_transfer call get_kernel_ptr()
      which calls cudaHostGetDevicePointer() on the BAR CPU address. The
      kernel writes to the returned GPU VA -> PCIe Write TLPs -> BAR.
      Host DRAM is never touched.

    Hot path (BAR -> GPU, H2D):
      bar_memcpy_async() calls cudaHostGetDevicePointer() then issues
      cudaMemcpyDeviceToDevice between the two GPU VA regions.

    Constraints (caller responsibility):
      - Linux only, IOMMU disabled or in passthrough mode.
      - bar_offset must be page-aligned.
      - size must not exceed the actual BAR window.
    """

    is_pcie_bar_memory: bool = True

    def __init__(
        self,
        size: int,
        bar_path: str,
        bar_offset: int = 0,
        use_paging: bool = False,
        **kwargs,
    ):
        self.size = size
        self.bar_path = bar_path
        self.bar_offset = bar_offset
        self._released = False

        logger.info(
            "PcieBarMemoryAllocator: registering %.1f GB of PCIe BAR memory  "
            "path=%s  offset=%d  cpu_ptr=pending",
            size / 1024**3,
            bar_path,
            bar_offset,
        )
        try:
            ptr = device_ops.alloc_pcie_bar_ptr(bar_path, size, bar_offset)
        except Exception as e:
            logger.error(
                "PcieBarMemoryAllocator: FAILED to register BAR memory  "
                "path=%s  offset=%d  size=%d  error=%s",
                bar_path,
                bar_offset,
                size,
                e,
            )
            raise
        logger.info(
            "PcieBarMemoryAllocator: BAR memory ready  path=%s  cpu_ptr=0x%x",
            bar_path,
            ptr,
        )

        array_type = ctypes.c_uint8 * size
        self._ctypes_buf = array_type.from_address(ptr)
        self.buffer = torch.frombuffer(self._ctypes_buf, dtype=torch.uint8)

        self._inner: MemoryAllocatorInterface
        if use_paging:
            if "shapes" not in kwargs:
                raise ValueError("shapes required for paged allocator")
            if "dtypes" not in kwargs:
                raise ValueError("dtypes required for paged allocator")
            if "fmt" not in kwargs:
                raise ValueError("fmt required for paged allocator")
            self._inner = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            align = kwargs.get("align_bytes", AddressManager.ALIGN_BYTES)
            self._inner = PcieBarTensorMemoryAllocator(
                self.buffer, align_bytes=align
            )

        self._lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
    ) -> "MemoryObj | None":
        """Allocate a memory object from the BAR-backed buffer.

        Args:
            shapes: Tensor shape(s).
            dtypes: Tensor dtype(s).
            fmt: Memory format.
            allocator_type: Ignored; present for interface compatibility.

        Returns:
            A MemoryObj on success, or None if the buffer is full.
        """
        with self._lock:
            return self._inner.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: str | None = None,
    ) -> "list[MemoryObj] | None":
        """Allocate ``batch_size`` equal-sized memory objects from the BAR buffer.

        Args:
            shapes: Tensor shape(s) for each slot.
            dtypes: Tensor dtype(s).
            batch_size: Number of slots to allocate.
            fmt: Memory format.
            allocator_type: Ignored; present for interface compatibility.

        Returns:
            A list of MemoryObj on success, or None if insufficient space.
        """
        with self._lock:
            return self._inner.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: "MemoryObj",
             allocator_type: str | None = None) -> None:
        """Free a single memory object back to the BAR buffer.

        Args:
            memory_obj: The object to free.
            allocator_type: Ignored; present for interface compatibility.
        """
        with self._lock:
            self._inner.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: list["MemoryObj"],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch of memory objects back to the BAR buffer.

        Args:
            memory_objs: Objects to free.
            allocator_type: Ignored; present for interface compatibility.
            update_stats: Ignored; present for interface compatibility.
        """
        with self._lock:
            self._inner.batched_free(memory_objs)

    def memcheck(self) -> bool:
        """Return True if the inner allocator passes its consistency check."""
        with self._lock:
            return self._inner.memcheck()

    def close(self) -> None:
        """Unregister the BAR region from CUDA and release the mmap."""
        if not self._released:
            if self.buffer.numel() > 0:
                if torch_dev.is_available():
                    torch_dev.synchronize()
                logger.info(
                    "PcieBarMemoryAllocator: releasing BAR memory  path=%s  cpu_ptr=0x%x",
                    self.bar_path,
                    self.buffer.data_ptr(),
                )
                device_ops.free_pcie_bar_ptr(self.buffer.data_ptr(), self.size)
            self._released = True

    def __str__(self):
        return f"PcieBarMemoryAllocator(path={self.bar_path})"

    @property
    def align_bytes(self) -> int:
        return self._inner.address_manager._align
