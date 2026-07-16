# SPDX-License-Identifier: Apache-2.0

"""Device memory allocator that registers memory with Phoenix phxfs for DMA.

This allocator is the Phoenix equivalent of ``CuFileMemoryAllocator``. It
pre-allocates a contiguous device memory buffer and registers it with
``phxfs_regmem`` so that ``phxfs_read`` / ``phxfs_read_async`` can DMA directly
between NVMe (via Phoenix) and device memory, bypassing the host CPU.

The alignment and page size are queried at runtime from the Phoenix device
(``phx_cache.page_size``), making this allocator vendor-neutral: it works with
NVIDIA (64 KiB page), AMD HIP, Huawei NPU, or any backend registered with
phxfs via ``devconn_ops``.
"""

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator


class PhxDeviceMemoryAllocator(GPUMemoryAllocator):
    """Device memory allocator backed by Phoenix ``phxfs_regmem``.

    Differences from ``CuFileMemoryAllocator``:

    * Uses ``phxfs_regmem`` (via the ``phxcache`` pybind11 extension) instead
      of ``cuFileBufRegister``.
    * ``phxfs_regmem`` returns a *target_addr* (DMA-mapped address) that is
      stored for diagnostics but **not** passed to I/O calls.  All
      ``phxfs_read`` calls use the original ``tensor.data_ptr()`` (the raw
      device virtual address) with ``buf_offset`` for the slab offset.
    * Alignment is queried at runtime from ``phx_cache.page_size`` (e.g.
      64 KiB for NVIDIA) rather than hard-coded.
    * The ``device`` parameter must match the physical device that the
      ``PhxCache`` instance maps to (e.g. device 4 → phxfs_dev2).

    :param size: Size of the device memory pool in bytes.
    :param device: Torch device string (e.g. ``"cuda:4"``).  Must match the
        device that ``phx_cache`` is bound to.
    :param phx_cache: A ``phxcache.PhxCache`` instance with an open Phoenix
        device connection.  Required.
    """

    def __init__(self, size: int, device=None, phx_cache=None) -> None:
        assert phx_cache is not None, (
            "PhxDeviceMemoryAllocator requires a phxcache.PhxCache instance"
        )

        if device is None:
            if torch_dev.is_available():
                device = f"{torch_device_type}:{torch_dev.current_device()}"
            else:
                device = "cpu:0"

        # The device buffer must reside on the same physical device that the
        # PhxCache's phxfs device maps to (e.g. device 4 -> phxfs_dev2).
        # phxfs_regmem performs a P2P mapping that fails with
        # "p2p_map->vaddrs _phxfs_regmem fail" (-14) when the device memory is
        # on a different (cross-NUMA) device than the phxfs device.

        # Query the device page size at runtime (vendor-specific, e.g. 64 KiB
        # for NVIDIA).  This is the single source of truth for alignment.
        align_bytes = phx_cache.page_size

        super().__init__(size, device, align_bytes=align_bytes)

        self.phx_cache = phx_cache
        self.base_pointer = self.tensor.data_ptr()
        self.size = size

        # Register the entire pre-allocated device memory block with Phoenix.
        # target_base is the DMA-mapped address used by phxfs_read/read_async.
        self.target_base = phx_cache.regmem(self.base_pointer, size)

    def __del__(self) -> None:
        if hasattr(self, "base_pointer") and hasattr(self, "phx_cache"):
            try:
                self.phx_cache.deregmem(self.base_pointer, self.size)
            except Exception:
                pass

    def __str__(self) -> str:
        return "PhxDeviceMemoryAllocator"


# Deprecated alias for backward compatibility.
PhxFileMemoryAllocator = PhxDeviceMemoryAllocator
