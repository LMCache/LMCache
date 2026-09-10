# SPDX-License-Identifier: Apache-2.0

"""SPDK FFI bindings for C++ implementation.

This module provides Python bindings to the SPDK C++ library (liblmcache_spdk.so)
using ctypes. It wraps the SpdkIoEngineCore class and provides methods for:
- Initializing/deinitializing SPDK
- Registering/unregistering external memory
- Performing read/write operations via NVMe-oF
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
import ctypes
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

IO_READ = 0
IO_WRITE = 1


class SpdkIoEngineFFI:
    """FFI wrapper for SpdkIoEngineCore C++ class.

    This class provides Python bindings to the SPDK C++ implementation
    using ctypes. It handles library loading, function signature setup,
    and method calls.
    """

    # Default library name
    LIB_NAME = os.environ.get("LMCACHE_SPDK_LIB", "liblmcache_spdk.so")

    @staticmethod
    def _resolve_library_path() -> str:
        """Resolve the full path to the SPDK shared library"""
        return os.path.join(os.path.dirname(__file__), "liblmcache_spdk.so")

    def __init__(self):
        """Initialize SPDK FFI wrapper"""
        lib_path = self._resolve_library_path()
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(
                f"Failed to load SPDK library {lib_path}: {e}\n"
                "Ensure the library is built and "
                "LMCACHE_SPDK_LIB or LD_LIBRARY_PATH is set correctly."
            ) from e

        logger.debug("Loaded SPDK library from %s", lib_path)

        self._obj = None
        try:
            make_core_fn = self._lib.make_SpdkIoEngineCore
            make_core_fn.restype = ctypes.c_void_p
            self._obj = make_core_fn()
            if self._obj is None:
                raise RuntimeError("Failed to create SpdkIoEngineCore instance")
        except Exception as e:
            raise RuntimeError(
                f"Error creating SpdkIoEngineCore: {e}\n"
                "Ensure the SPDK library exports the required symbols."
            ) from e

        self._init_spdk = self._lib.core_init_spdk
        self._init_spdk.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._init_spdk.restype = ctypes.c_int

        self._deinit_spdk = self._lib.core_deinit_spdk
        self._deinit_spdk.argtypes = [ctypes.c_void_p]
        self._deinit_spdk.restype = None

        self._register_external = self._lib.core_register_external_memory
        self._register_external.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._register_external.restype = ctypes.c_int

        self._unregister_external = self._lib.core_unregister_external_memory
        self._unregister_external.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._unregister_external.restype = ctypes.c_int

        self._spdk_io = self._lib.core_spdk_io
        self._spdk_io.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._spdk_io.restype = ctypes.c_int

        self._launch_spdk_workers = self._lib.core_launch_spdk_workers
        self._launch_spdk_workers.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._launch_spdk_workers.restype = ctypes.c_int

        self._shutdown_spdk_workers = self._lib.core_shutdown_spdk_workers
        self._shutdown_spdk_workers.argtypes = [ctypes.c_void_p]
        self._shutdown_spdk_workers.restype = None

        self._get_device_size = self._lib.core_get_device_size
        self._get_device_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._get_device_size.restype = ctypes.c_int

        self._allocate_spdk_memory = self._lib.core_allocate_spdk_memory
        self._allocate_spdk_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._allocate_spdk_memory.restype = ctypes.c_uint64

        self._free_spdk_memory = self._lib.core_free_spdk_memory
        self._free_spdk_memory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._free_spdk_memory.restype = None

        self._batch_io_submit = self._lib.core_spdk_batch_io_submit
        self._batch_io_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self._batch_io_submit.restype = ctypes.c_int

        self._wait_spdk_batch = self._lib.core_spdk_wait_batch
        self._wait_spdk_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._wait_spdk_batch.restype = ctypes.c_int

    def allocate_spdk_memory(
        self, size: int, align: int = 4096, numa_id: int = -1
    ) -> int:
        """Allocate DMA-safe memory using SPDK's spdk_dma_zmalloc"""
        ptr = self._allocate_spdk_memory(
            self._obj,
            ctypes.c_size_t(size),
            ctypes.c_size_t(align),
            ctypes.c_int(numa_id),
        )
        if ptr != 0:
            logger.debug("Allocated %d bytes of SPDK DMA memory at ptr=0x%x", size, ptr)
        else:
            logger.error("Failed to allocate %d bytes of SPDK DMA memory", size)
        return ptr

    def free_spdk_memory(self, ptr: int) -> None:
        """Free SPDK DMA-safe memory.

        Args:
            ptr: Pointer returned by allocate_spdk_memory.
        """
        if ptr != 0:
            try:
                self._free_spdk_memory(self._obj, ctypes.c_uint64(ptr))
                logger.debug("Freed SPDK DMA memory at ptr=0x%x", ptr)
            except Exception as e:
                logger.error("Error freeing SPDK DMA memory: %s", e)

    def init(self, core_mask: str = "", mem_size_mb: int = 0) -> int:
        """Initialize the SPDK environment.

        The DPDK/SPDK core mask and memory size are passed here instead of via
        separate setters. The I/O worker and admin worker cores are
        automatically derived from the core mask:
        - If only 1 core: All workers use that single core
        - If 2+ cores: I/O worker uses the highest core, admin uses the second-highest

        Args:
            core_mask: Hex string representing available cores
                      (e.g., "0x3f" for cores 0-5). Must be non-empty.
            mem_size_mb: Hugepage memory size in MB to reserve for the SPDK
                         environment.

        Returns:
            0 on success, non-zero on failure.
        """
        core_mask_bytes = core_mask.encode("utf-8") if core_mask else b""

        rc = self._init_spdk(self._obj, core_mask_bytes, ctypes.c_int(mem_size_mb))
        if rc == 0:
            logger.debug("SPDK environment initialized successfully")
        else:
            logger.error("SPDK initialization failed with rc=%d", rc)
        return rc

    def deinit(self) -> None:
        """Deinitialize SPDK environment."""
        try:
            self._deinit_spdk(self._obj)
            self._obj = None
            logger.debug("SPDK environment deinitialized")
        except Exception as e:
            logger.error("Error deinitializing SPDK: %s", e)

    def register_external_memory(self, ptr: int, size: int) -> int:
        """Register external memory with SPDK"""
        rc = self._register_external(
            self._obj, ctypes.c_size_t(ptr), ctypes.c_size_t(size)
        )
        if rc == 0:
            logger.debug("Registered %d bytes at ptr=0x%x with SPDK", size, ptr)
        else:
            logger.error(
                "Failed to register memory: ptr=0x%x, size=%d, rc=%d", ptr, size, rc
            )
        return rc

    def unregister_external_memory(self, ptr: int, size: int) -> int:
        """Unregister external memory from SPDK"""
        rc = self._unregister_external(
            self._obj, ctypes.c_size_t(ptr), ctypes.c_size_t(size)
        )
        if rc == 0:
            logger.debug("Unregistered %d bytes at ptr=0x%x from SPDK", size, ptr)
        else:
            logger.error(
                "Failed to unregister memory: ptr=0x%x, size=%d, rc=%d", ptr, size, rc
            )
        return rc

    def spdk_io(self, byte_offset: int, byte_count: int, buffer_ptr, op: int) -> int:
        """Write or read data using SPDK.

        The byte offset/count are converted to LBA internally using the
        attached NVMe namespace sector size.

        Args:
            byte_offset: Starting byte offset on the device.
            byte_count: Number of bytes to transfer.
            buffer_ptr: Pointer to the data buffer.
            op: IO_READ or IO_WRITE.

        Returns:
            0 on success, non-zero on failure.
        """
        op_name = "read" if op == IO_READ else "write"
        rc = self._spdk_io(
            self._obj,
            ctypes.c_uint64(byte_offset),
            ctypes.c_uint64(byte_count),
            buffer_ptr,
            ctypes.c_int(op),
        )
        if rc != 0:
            logger.error(
                "SPDK %s failed: byte_offset=%d, byte_count=%d, rc=%d",
                op_name,
                byte_offset,
                byte_count,
                rc,
            )
        return rc

    def launch_spdk_workers(
        self,
        transport_type: str = "tcp",
        addr: str = "127.0.0.1",
        port: str = "4420",
        nqn: str = "nqn.2016-06.io.spdk:cnode1",
    ) -> int:
        """Launch the SPDK I/O and admin worker threads.

        Args:
            transport_type: Transport type - "pcie" for local NVMe, "tcp" or
                            "rdma" for NVMe-oF.
                            For PCIe: addr = "0000:01:00.0", port and nqn are ignored.
                            For TCP/RDMA: addr = IP, port = port, nqn = NQN.
            addr: For PCIe: device address (e.g., "0000:01:00.0").
                  For TCP: IP address of the NVMe-oF target.
            port: Port number of the NVMe-oF target (ignored for PCIe).
            nqn: NVMe Qualified Name of the target subsystem (ignored for PCIe).

        Returns:
            0 on success, negative error code on failure.
        """
        if transport_type == "pcie":
            logger.debug("Launching SPDK I/O worker: PCIe device=%s", addr)
        elif transport_type == "rdma":
            logger.debug(
                "Launching SPDK I/O worker: RDMA IP=%s, Port=%s, NQN=%s",
                addr,
                port,
                nqn,
            )
        else:
            logger.debug(
                "Launching SPDK I/O worker: TCP IP=%s, Port=%s, NQN=%s", addr, port, nqn
            )

        rc = self._launch_spdk_workers(
            self._obj,
            transport_type.encode("utf-8"),
            addr.encode("utf-8"),
            port.encode("utf-8"),
            nqn.encode("utf-8"),
        )
        if rc == 0:
            logger.debug("SPDK workers launched successfully")
        else:
            logger.error("SPDK worker launch failed with rc=%d", rc)
        return rc

    def shutdown_spdk_workers(self) -> None:
        """Signal the SPDK I/O and admin worker threads to shut down."""
        try:
            self._shutdown_spdk_workers(self._obj)
            logger.debug("SPDK workers shutdown successfully")
        except Exception as e:
            logger.error("Error shutting down SPDK workers: %s", e)

    def get_device_size(self) -> int:
        """Get the NVMe device size in bytes"""
        size = ctypes.c_uint64(0)
        rc = self._get_device_size(
            self._obj,
            ctypes.byref(size),
        )
        if rc == 0:
            logger.info(
                "NVMe device size: %d bytes (%.2f GB)",
                size.value,
                size.value / (1024 * 1024 * 1024),
            )
        else:
            logger.error("Failed to get device size with rc=%d", rc)
        return -1 if rc != 0 else size.value

    def __del__(self) -> None:
        """Cleanup SPDK resources."""
        if self._obj is not None:
            try:
                self.shutdown_spdk_workers()
                self.deinit()
            except Exception:
                pass

    def batch_io_submit(
        self,
        offsets: Sequence[int],
        total_lens: Sequence[int],
        buf_ptrs: list[int],
        count: int,
        op: int,
    ) -> tuple[int, int]:
        """Submit a batch of read or write operations.

        Args:
            offsets: Sequence of byte offsets for each operation.
            total_lens: Sequence of byte counts for each operation.
            buf_ptrs: List of device pointers (zero-copy registered or DMA-allocated)
                for each operation. Each entry is used as the transfer buffer.
            count: Number of operations in the batch.
            op: IO_READ or IO_WRITE.

        Returns:
            A tuple ``(rc, batch_id)`` where ``rc`` is 0 on success and -1 on
            failure, and ``batch_id`` is the assigned batch identifier (valid
            when ``rc == 0``).

        Raises:
            RuntimeError: If ``count == 0``, since an empty batch cannot be
            submitted to the SPDK io_worker.
        """
        count = len(offsets)
        if count == 0:
            raise RuntimeError("batch_io_submit called with count == 0")
        if count != len(total_lens) or count != len(buf_ptrs):
            raise ValueError(
                "offsets, total_lens, and buf_ptrs must all have the same length"
            )

        op_name = "read" if op == IO_READ else "write"
        offset_arr = (ctypes.c_uint64 * count)(*offsets)
        len_arr = (ctypes.c_uint64 * count)(*total_lens)
        ptr_arr = (ctypes.c_uint64 * count)(*buf_ptrs)

        batch_id = ctypes.c_uint(0)
        rc = self._batch_io_submit(
            self._obj,
            offset_arr,
            len_arr,
            ptr_arr,
            ctypes.c_int(count),
            ctypes.c_int(op),
            ctypes.byref(batch_id),
        )
        if rc != 0:
            logger.error("SPDK batched %s submission failed: rc=%d", op_name, rc)
        else:
            logger.debug(
                "Submitted SPDK batched %s: count=%d, batch_id=%d",
                op_name,
                batch_id.value,
            )
        return rc, batch_id.value

    def wait_batch(self, batch_id: int) -> int:
        """Wait for a batch of I/O operations to complete.

        Args:
            batch_id: The batch ID returned by batch_io_submit.

        Returns:
            0 if every I/O in the batch succeeded, -1 if any I/O failed or the
            batch_id is unknown.
        """
        rc = self._wait_spdk_batch(self._obj, ctypes.c_uint(batch_id))
        if rc == 0:
            logger.debug("Batch %d completed successfully", batch_id)
        else:
            logger.error(
                "SPDK wait_batch failed for batch_id=%d (status=%d)", batch_id, rc
            )
        return rc
