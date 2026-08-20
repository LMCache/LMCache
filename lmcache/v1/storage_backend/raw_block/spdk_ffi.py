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
import ctypes
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


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
        """Resolve the full path to the SPDK shared library.

        Only loads from lmcache/storage_backends/raw_block/liblmcache_spdk.so
        (next to this Python file).

        Returns:
            Path to the library file.
        """
        # Only load from the installed package location
        return os.path.join(os.path.dirname(__file__), "liblmcache_spdk.so")

    def __init__(self):
        """Initialize SPDK FFI wrapper.

        Loads the SPDK C++ library and sets up function signatures.

        Raises:
            RuntimeError: If the library cannot be loaded.
        """
        # Resolve and load the library
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

        # Create C++ object using factory function
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

        # Set up function signatures for core mask
        # configuration (must be called before init)
        self._set_dpdk_core_mask = self._lib.core_set_dpdk_core_mask
        self._set_dpdk_core_mask.argtypes = [ctypes.c_char_p]
        self._set_dpdk_core_mask.restype = ctypes.c_int

        # Set up function signatures for init/deinit (C wrapper functions)
        self._init_spdk = self._lib.core_init_spdk
        self._init_spdk.argtypes = [ctypes.c_void_p]
        self._init_spdk.restype = ctypes.c_int

        # Set up function signature for mem_size configuration
        self._set_mem_size = self._lib.core_set_mem_size
        self._set_mem_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._set_mem_size.restype = None

        self._deinit_spdk = self._lib.core_deinit_spdk
        self._deinit_spdk.argtypes = [ctypes.c_void_p]
        self._deinit_spdk.restype = None

        # Set up function signatures for memory registration
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

        # Set up function signatures for I/O operations (byte offset based)
        self._spdk_write = self._lib.core_spdk_write_external
        self._spdk_write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._spdk_write.restype = ctypes.c_int

        self._spdk_read = self._lib.core_spdk_read_external
        self._spdk_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._spdk_read.restype = ctypes.c_int

        # Set up function signatures for connection parameter setting
        self._set_connection_params = self._lib.core_set_connection_params
        self._set_connection_params.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._set_connection_params.restype = ctypes.c_int

        # Set up function signatures for thread management
        self._launch_io_worker = self._lib.core_launch_io_worker
        self._launch_io_worker.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._launch_io_worker.restype = ctypes.c_int

        self._shutdown_io_worker = self._lib.core_shutdown_io_worker
        self._shutdown_io_worker.argtypes = [ctypes.c_void_p]
        self._shutdown_io_worker.restype = None

        # Set up function signatures for device size query
        self._get_device_size = self._lib.core_get_device_size
        self._get_device_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._get_device_size.restype = ctypes.c_int

        # Set up function signatures for SPDK DMA memory allocation
        self._allocate_spdk_memory = self._lib.core_allocate_spdk_memory
        self._allocate_spdk_memory.argtypes = [
            ctypes.c_void_p,  # core_ptr
            ctypes.c_size_t,  # size
            ctypes.c_size_t,  # align
            ctypes.c_int,  # numa_id
        ]
        self._allocate_spdk_memory.restype = ctypes.c_uint64

        self._free_spdk_memory = self._lib.core_free_spdk_memory
        self._free_spdk_memory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._free_spdk_memory.restype = None

    def allocate_spdk_memory(
        self, size: int, align: int = 4096, numa_id: int = -1
    ) -> int:
        """Allocate DMA-safe memory using SPDK's spdk_dma_zmalloc.

        This allocates memory that is already registered with SPDK for DMA
        operations, avoiding the need for separate memory registration.

        Args:
            size: Number of bytes to allocate.
            align: Alignment in bytes (default 4096).
            numa_id: NUMA socket ID (-1 for any).

        Returns:
            Pointer to the allocated memory, or 0 on failure.
        """
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

    def set_dpdk_core_mask(self, core_mask: str) -> int:
        """Set the DPDK/SPDK core mask before initialization.

        This controls which CPU cores SPDK can use. The I/O worker and admin
        worker cores are automatically derived from this mask:
        - If only 1 core: All workers use that single core
        - If 2+ cores: I/O worker uses the highest core, admin uses the second-highest

        Must be called before init().

        Args:
            core_mask: Hex string representing available cores
                      (e.g., "0x3f" for cores 0-5).
                      Empty string means SPDK uses all available cores.

        Returns:
            0 on success.

        Example:
            >>> ffi = SpdkIoEngineFFI()
            >>> ffi.set_dpdk_core_mask("0x3f")  # Cores 0-5
            >>> ffi.init()
        """
        if not core_mask:
            core_mask_bytes = b""
        else:
            core_mask_bytes = core_mask.encode("utf-8")

        rc = self._set_dpdk_core_mask(core_mask_bytes)
        if rc == 0:
            logger.debug("DPDK core mask set to: %s", core_mask)
        else:
            logger.error("Failed to set DPDK core mask: %s", core_mask)
        return rc

    def set_mem_size(self, mem_size_mb: int) -> None:
        """Set the SPDK memory size in MB for hugepage allocation.

        This controls how much hugepage memory SPDK reserves during
        initialization via DPDK's -m flag. Must be called before init().

        Args:
            mem_size_mb: Memory size in megabytes (0 = use SPDK default).
        """
        if self._set_mem_size is not None:
            self._set_mem_size(self._obj, ctypes.c_int(mem_size_mb))

    def init(self) -> int:
        """Initialize SPDK environment.

        Returns:
            0 on success, negative error code on failure.
        """
        rc = self._init_spdk(self._obj)
        if rc == 0:
            logger.debug("SPDK environment initialized successfully")
        else:
            logger.error("SPDK initialization failed with rc=%d", rc)
        return rc

    def deinit(self) -> None:
        """Deinitialize SPDK environment."""
        try:
            self._deinit_spdk(self._obj)
            logger.debug("SPDK environment deinitialized")
        except Exception as e:
            logger.error("Error deinitializing SPDK: %s", e)

    def register_external_memory(self, ptr: int, size: int) -> int:
        """Register external memory with SPDK.

        Args:
            ptr: Pointer to the memory buffer.
            size: Size of the memory region in bytes.

        Returns:
            0 on success, negative error code on failure.
        """
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
        """Unregister external memory from SPDK.

        Args:
            ptr: Pointer to the memory buffer.
            size: Size of the memory region in bytes.

        Returns:
            0 on success, negative error code on failure.
        """
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

    def spdk_write_external(self, byte_offset: int, byte_count: int, buffer_ptr) -> int:
        """Write data using SPDK.

        The byte offset and byte count are converted to LBA and lba_count
        internally by the C++ SPDK code using the device's sector size.

        Args:
            byte_offset: Starting byte offset on the device.
            byte_count: Number of bytes to write.
            buffer_ptr: Pointer to the data buffer (must be registered).

        Returns:
            0 on success, negative error code on failure.
        """
        rc = self._spdk_write(
            self._obj,
            ctypes.c_uint64(byte_offset),
            ctypes.c_uint64(byte_count),
            buffer_ptr,
        )
        if rc != 0:
            logger.error(
                "SPDK write failed: byte_offset=%d, byte_count=%d, rc=%d",
                byte_offset,
                byte_count,
                rc,
            )
        return rc

    def spdk_read_external(self, byte_offset: int, byte_count: int, buffer_ptr) -> int:
        """Read data using SPDK.

        The byte offset and byte count are converted to LBA and lba_count
        internally by the C++ SPDK code using the device's sector size.

        Args:
            byte_offset: Starting byte offset on the device.
            byte_count: Number of bytes to read.
            buffer_ptr: Pointer to the destination buffer (must be registered).

        Returns:
            0 on success, negative error code on failure.
        """
        rc = self._spdk_read(
            self._obj,
            ctypes.c_uint64(byte_offset),
            ctypes.c_uint64(byte_count),
            buffer_ptr,
        )
        if rc != 0:
            logger.error(
                "SPDK read failed: byte_offset=%d, byte_count=%d, rc=%d",
                byte_offset,
                byte_count,
                rc,
            )
        return rc

    def launch_io_worker(
        self,
        transport_type: str = "tcp",
        addr: str = "127.0.0.1",
        port: str = "4420",
        nqn: str = "nqn.2019-04.pos:subsystem1",
    ) -> int:
        """Launch SPDK I/O worker thread.

        Args:
            transport_type: Transport type - "pcie" for local NVMe, "tcp" for NVMe-oF.
                           For PCIe: addr = "0000:01:00.0", port and nqn are ignored.
                           For TCP: addr = IP, port = port, nqn = NQN.
            addr: For PCIe: device address (e.g., "0000:01:00.0").
                  For TCP: IP address of the NVMe-oF target.
            port: Port number of the NVMe-oF target (ignored for PCIe).
            nqn: NVMe Qualified Name of the target subsystem (ignored for PCIe).

        Returns:
            0 on success, negative error code on failure.

        Examples:
            # PCIe (local NVMe):
            ffi.launch_io_worker("pcie", "0000:01:00.0")

            # TCP (NVMe-oF):
            ffi.launch_io_worker(
                "tcp", "107.99.41.188", "1158",
                "nqn.2019-04.pos:subsystem1"
            )
        """
        if transport_type == "pcie":
            logger.debug("Launching SPDK I/O worker: PCIe device=%s", addr)
        else:
            logger.debug(
                "Launching SPDK I/O worker: TCP IP=%s, Port=%s, NQN=%s", addr, port, nqn
            )

        # Set connection parameters first
        rc = self._set_connection_params(
            self._obj,
            transport_type.encode("utf-8"),
            addr.encode("utf-8"),
            port.encode("utf-8"),
            nqn.encode("utf-8"),
        )
        if rc != 0:
            logger.error("Failed to set connection parameters with rc=%d", rc)
            return rc

        # Launch I/O worker with connection parameters
        rc = self._launch_io_worker(
            self._obj,
            transport_type.encode("utf-8"),
            addr.encode("utf-8"),
            port.encode("utf-8"),
            nqn.encode("utf-8"),
        )
        if rc == 0:
            logger.debug("SPDK I/O worker launched successfully")
        else:
            logger.error("SPDK I/O worker launch failed with rc=%d", rc)
        return rc

    def shutdown_io_worker(self) -> None:
        """Shutdown SPDK I/O worker thread."""
        try:
            self._shutdown_io_worker(self._obj)
            logger.debug("SPDK I/O worker shutdown successfully")
        except Exception as e:
            logger.error("Error shutting down SPDK I/O worker: %s", e)

    def get_device_size(self) -> int:
        """Get the NVMe device size in bytes.

        This queries the NVMe namespace for capacity information after
        the I/O worker has been launched and the device is connected.

        Returns:
            Device size in bytes, or -1 on failure.

        Example:
            >>> ffi = SpdkIoEngineFFI()
            >>> ffi.init()
            >>> ffi.launch_io_worker("pcie", "0000:01:00.0")
            >>> size = ffi.get_device_size()
            >>> print(f"Device size: {size} bytes ({size / (1024**3):.2f} GB)")
        """
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
                self.shutdown_io_worker()
                self.deinit()
            except Exception:
                pass  # Ignore errors during cleanup
