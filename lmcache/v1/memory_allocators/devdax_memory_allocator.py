# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List, Optional, Union
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
    logger,
)
import lmcache.v1.memory_management as memory_management


class DevDaxMemoryAllocator(MemoryAllocatorInterface):
    """Allocates L1 objects from DRAM and/or an mmap-backed Device-DAX arena.

    The mapped bytes are exposed as a flat CPU ``torch.uint8`` tensor so the
    existing L1 state machine and tensor slicing logic can run unchanged while
    the overflow backing storage is the configured Device-DAX device. When a
    local allocator is provided, local DRAM is tried first and Device-DAX is
    used as overflow.
    """

    def __init__(
        self,
        size: int,
        device_path: str,
        *,
        local_allocator: MixedMemoryAllocator | None = None,
        local_size: int = 0,
        shm_name: str | None = None,
        align_bytes: int = AddressManager.ALIGN_BYTES,
    ) -> None:
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        if size <= 0:
            raise ValueError("size must be > 0")
        if local_size < 0:
            raise ValueError("local_size must be >= 0")
        if local_size and local_allocator is not None:
            raise ValueError("local_size cannot be used with local_allocator")
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        self.size = size
        self.device_path = device_path
        self.align_bytes = align_bytes
        self.local_allocator = local_allocator
        if local_size:
            self.local_allocator = MixedMemoryAllocator(
                local_size,
                align_bytes=self.align_bytes,
                shm_name=shm_name,
            )
        self.host_mem_lock = threading.Lock()
        self.buffer_allocator = BufferAllocator("cpu")
        self.devdax_allocator: TensorMemoryAllocator | None = None
        self._fd: int | None = None
        self._mmap_obj: mmap.mmap | None = None
        self._mmap_buffer: Any | None = None
        self._unregistered = False
        self._host_memory_pinned_ptr: int | None = None

        self.devdax_buffer = self._map_devdax()
        self.devdax_allocator = TensorMemoryAllocator(
            self.devdax_buffer, align_bytes=self.align_bytes
        )
        self.address_manager = self.devdax_allocator.address_manager
        self._register_cuda_host_memory()

    @property
    def buffer(self) -> torch.Tensor:
        """Return the primary L1 buffer exposed by this allocator."""
        if self.local_allocator is not None:
            return self.local_allocator.buffer
        return self.devdax_buffer

    def _open_devdax_mapping(
        self,
        device_path: str,
        size: int,
    ) -> tuple[int, mmap.mmap, Any, torch.Tensor]:
        fd: int | None = None
        mmap_obj: mmap.mmap | None = None
        try:
            fd = os.open(device_path, os.O_RDWR)
            capacity = os.fstat(fd).st_size
            if capacity > 0 and size > capacity:
                raise RuntimeError(
                    f"l1 devdax size ({size} bytes) exceeds "
                    f"{device_path} capacity ({capacity} bytes)"
                )

            mmap_obj = mmap.mmap(
                fd,
                size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            array_type = ctypes.c_uint8 * size
            mmap_buffer = array_type.from_buffer(mmap_obj)
            buffer = torch.frombuffer(mmap_buffer, dtype=torch.uint8)
            return fd, mmap_obj, mmap_buffer, buffer
        except Exception:
            if mmap_obj is not None:
                mmap_obj.close()
            if fd is not None:
                os.close(fd)
            raise

    def _map_devdax(self) -> torch.Tensor:
        fd, mmap_obj, mmap_buffer, buffer = self._open_devdax_mapping(
            self.device_path,
            self.size,
        )
        self._fd = fd
        self._mmap_obj = mmap_obj
        self._mmap_buffer = mmap_buffer
        return buffer

    def _close_devdax_mapping_locked(self) -> None:
        self._unregister_cuda_host_memory()
        self.devdax_allocator = None
        self.devdax_buffer = torch.empty(0, dtype=torch.uint8)
        self._mmap_buffer = None

        if self._mmap_obj is not None:
            self._mmap_obj.close()
            self._mmap_obj = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _register_cuda_host_memory(self) -> None:
        if not memory_management.current_device_spec.is_pin_supported:
            return
        if not memory_management.current_device_spec.pin_memory(
            self.devdax_buffer.data_ptr(), self.size
        ):
            logger.warning(
                "pin_memory failed for Device-DAX L1 mapping; "
                "falling back to pageable host copies"
            )
            return
        self._host_memory_pinned_ptr = self.devdax_buffer.data_ptr()

    def _unregister_cuda_host_memory(self) -> None:
        if self._host_memory_pinned_ptr is None:
            return
        memory_management.current_device_spec.unpin_memory(self._host_memory_pinned_ptr)
        self._host_memory_pinned_ptr = None

    def _is_local_obj(self, memory_obj: MemoryObj) -> bool:
        return (
            self.local_allocator is not None
            and memory_obj.parent() is self.local_allocator
        )

    def _is_devdax_obj(self, memory_obj: MemoryObj) -> bool:
        return memory_obj.parent() is self

    def _local_available_count(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> int:
        if self.local_allocator is None:
            return 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        unit_raw_size = get_size_bytes(shapes, dtypes)
        unit_aligned_size = local_pin_allocator.address_manager.compute_aligned_size(
            unit_raw_size
        )
        return local_pin_allocator.address_manager.get_free_size() // unit_aligned_size

    def _local_memory_usage(self) -> tuple[int, int]:
        if self.local_allocator is None:
            return 0, 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        local_total = local_pin_allocator.address_manager.get_heap_size()
        local_used = local_total - local_pin_allocator.address_manager.get_free_size()
        return local_used, local_total

    def _allocate_from_devdax(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            assert self.devdax_allocator is not None
            obj = self.devdax_allocator.allocate(shapes, dtypes, fmt, str(self))
            if isinstance(obj, TensorMemoryObj):
                obj.parent_allocator = self
            return obj

    def _batched_allocate_from_devdax(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            assert self.devdax_allocator is not None
            objs = self.devdax_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )
            if objs is not None:
                for obj in objs:
                    if isinstance(obj, TensorMemoryObj):
                        obj.parent_allocator = self
            return objs

    def _free_devdax_obj(self, memory_obj: MemoryObj) -> None:
        with self.host_mem_lock:
            assert self.devdax_allocator is not None
            self.devdax_allocator.free(memory_obj)
            if isinstance(memory_obj, TensorMemoryObj):
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)

    def _batched_free_devdax_objs(
        self,
        memory_objs: List[MemoryObj],
        update_stats: bool,
    ) -> None:
        with self.host_mem_lock:
            assert self.devdax_allocator is not None
            self.devdax_allocator.batched_free(memory_objs, update_stats=update_stats)
            for memory_obj in memory_objs:
                if isinstance(memory_obj, TensorMemoryObj):
                    memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one object from local DRAM first, then Device-DAX.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format to allocate.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if both tiers are full.

        Raises:
            ValueError: If ``fmt`` is unsupported.
        """
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shapes, dtypes, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self.local_allocator is not None:
                obj = self.local_allocator.allocate(shapes, dtypes, fmt, str(self))
                if obj is not None:
                    return obj
            if self.devdax_allocator is None:
                return None
            return self._allocate_from_devdax(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate multiple objects from local DRAM first, then Device-DAX.

        Args:
            shapes: Logical tensor shape or shapes for each allocation.
            dtypes: Logical tensor dtype or dtypes for each allocation.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format to allocate.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects, or ``None`` if the full batch cannot be allocated.

        Raises:
            ValueError: If ``fmt`` is unsupported.
        """
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs: list[MemoryObj] = []
            if self.local_allocator is not None:
                local_count = min(
                    batch_size, self._local_available_count(shapes, dtypes)
                )
                if local_count:
                    local_objs = (
                        self.local_allocator.batched_allocate(
                            shapes, dtypes, local_count, fmt, str(self)
                        )
                        or []
                    )

            remaining = batch_size - len(local_objs)
            if remaining == 0:
                return local_objs

            if self.devdax_allocator is None:
                if local_objs and self.local_allocator is not None:
                    self.local_allocator.batched_free(local_objs, update_stats=False)
                return None

            dax_objs = self._batched_allocate_from_devdax(
                shapes, dtypes, remaining, fmt
            )
            if dax_objs is None:
                if local_objs and self.local_allocator is not None:
                    self.local_allocator.batched_free(local_objs, update_stats=False)
                return None
            return local_objs + dax_objs
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """Free one Device-DAX allocator memory object.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Optional allocator type string.

        Raises:
            ValueError: If the object format is unsupported or the object does
                not belong to this allocator.
        """
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self._is_local_obj(memory_obj):
                assert self.local_allocator is not None
                self.local_allocator.free(memory_obj)
                return
            if self._is_devdax_obj(memory_obj):
                self._free_devdax_obj(memory_obj)
                return
            raise ValueError("Memory object does not belong to DevDaxMemoryAllocator")
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free multiple Device-DAX allocator memory objects.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.

        Raises:
            ValueError: If the object format is unsupported or any object does
                not belong to this allocator.
        """
        if not memory_objs:
            return

        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self._is_local_obj(memory_obj)
            ]
            devdax_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self._is_devdax_obj(memory_obj)
            ]
            if len(local_objs) + len(devdax_objs) != len(memory_objs):
                raise ValueError(
                    "One or more memory objects do not belong to DevDaxMemoryAllocator"
                )
            if local_objs:
                assert self.local_allocator is not None
                self.local_allocator.batched_free(local_objs, update_stats=update_stats)
            if devdax_objs:
                self._batched_free_devdax_objs(devdax_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def memcheck(self) -> bool:
        """Return whether local and Device-DAX allocator state is consistent."""
        local_ok = True
        if self.local_allocator is not None:
            local_ok = self.local_allocator.memcheck()
        with self.host_mem_lock:
            if self.devdax_allocator is None:
                return local_ok
            return local_ok and self.devdax_allocator.memcheck()

    def get_memory_usage(self) -> tuple[int, int]:
        """Return used and total bytes across local and Device-DAX tiers."""
        local_used, local_total = self._local_memory_usage()

        if self.devdax_allocator is None:
            return local_used, local_total
        dax_total = self.devdax_allocator.address_manager.get_heap_size()
        dax_used = dax_total - self.devdax_allocator.address_manager.get_free_size()
        return local_used + dax_used, local_total + dax_total

    def close(self) -> None:
        """Release mapped Device-DAX and optional local DRAM resources.

        Raises:
            BufferError: If Device-DAX allocations are still active.
        """
        if self._unregistered:
            return
        if memory_management.torch_dev.is_available():
            memory_management.torch_dev.synchronize()

        with self.host_mem_lock:
            if (
                self.devdax_allocator is not None
                and self.devdax_allocator.num_active_allocations > 0
            ):
                raise BufferError(
                    "cannot close DevDaxMemoryAllocator with active allocations"
                )
            self._close_devdax_mapping_locked()

        if self.local_allocator is not None:
            self.local_allocator.close()
            self.local_allocator = None
        self._unregistered = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __str__(self) -> str:
        return "DevDaxMemoryAllocator"
