# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Union
import ctypes
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform import current_device_spec
from lmcache.v1.system_detection import NUMAMapping

logger = init_logger(__name__)


# Helper functions
def get_numa_id(numa_mapping: NUMAMapping) -> int:
    """
    Get the NUMA ID for the current GPU

    Args:
        numa_mapping (NUMAMapping): The NUMA mapping object.

    Returns:
        int: The NUMA ID for the current GPU.

    Raises:
        KeyError: If GPU id is not detected in the numa mapping.
    """
    gpu_id = torch_dev.current_device() if torch_dev.is_available() else 0
    return numa_mapping.gpu_to_numa_mapping[gpu_id]


def align_to(size: int, align_size: int) -> int:
    """
    Align the given size to the nearest multiple of align_size.

    Args:
        size (int): The size to align.
        align_size (int): The alignment size, MUST BE a power of two.

    Returns:
        int: The aligned size.
    """
    return (size + align_size - 1) & (~(align_size - 1))


# Main class
class LazyMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates CPU (numa) pinned memory with a initial size and expand
    the size to the required size in the background.

    Background expansion logic:
    - After registering X GB memory, we call sbrk and updates _curr_size
    - Once everything is registered, the background thread stops

    Deferred pinning:
    - Pinning (``cudaHostRegister``) creates a CUDA context on the calling
      thread's current device. ``__init__`` runs before any worker's device
      is known, so pinning there would bind that context to the default
      device. Pinning therefore must stay out of ``__init__``: it is bound
      to a device by :meth:`ensure_pinning`, triggered at the latest by the
      first :meth:`allocate` / :meth:`batched_allocate`.
    """

    PIN_CHUNK_SIZE = 1 << 26  # 64 MB pin chunk
    COMMIT_SIZE = 1 << 30  # Do a commit every 1 GB
    LOG_INTERVAL = 10 << 30  # Log expansion progress every 10 GB

    def __init__(
        self,
        init_size: int,
        final_size: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_mapping: NUMAMapping | None = None,
    ) -> None:
        """
        Args:
            init_size (int): Initial size of the memory allocation in bytes.
            final_size (int): Final size of the memory allocation in bytes.
            align_bytes (int, optional): Alignment for the underlying allocations.
                Must be a positive power of two. The buffer's base address is
                aligned to this value, not merely the offsets within it.

        Raises:
            ValueError: If ``align_bytes`` is not a positive power of two.
            RuntimeError: If the platform does not support memory pinning, or if
                the allocated buffer could not be aligned to ``align_bytes``.
        """
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # Whether using NUMA allocation
        self._use_numa = numa_mapping is not None
        # Currently pinned size, only accessed by the expansion thread
        self._curr_size = align_to(init_size, self.PIN_CHUNK_SIZE)
        # Final size of the allocation, only accessed by the expansion thread
        self._final_size = align_to(final_size, self.PIN_CHUNK_SIZE)
        # Underlying buffer for the memory allocation
        self._buffer: torch.Tensor
        if not current_device_spec.is_pin_supported:
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support memory "
                "pinning. LazyMemoryAllocator requires pinned memory."
            )

        # List of (ptr, size) for pinned memory chunks
        self._pin_record: list[tuple[int, int]] = []

        # Detect numa mapping
        if numa_mapping is not None:
            numa_id = get_numa_id(numa_mapping)
            ptr = device_ops.alloc_numa_ptr(self._final_size, numa_id)
            arr_type = ctypes.c_uint8 * self._final_size
            buf = arr_type.from_address(ptr)
            self._buffer = torch.frombuffer(buf, dtype=torch.uint8)
        else:
            # torch.empty() only guarantees 64-byte alignment, but consumers of
            # get_l1_memory_desc() (O_DIRECT, RDMA/GDS) need the buffer base
            # itself aligned to align_bytes.
            backing = torch.empty(
                self._final_size + align_bytes - 1,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=False,
            )
            offset = (-backing.data_ptr()) % align_bytes
            # Slice shares storage with `backing`; no separate reference needed.
            self._buffer = backing[offset : offset + self._final_size]

        # Fail loudly here rather than let a misaligned buffer surface as an
        # O_DIRECT EINVAL somewhere downstream.
        base_ptr = self._buffer.data_ptr()
        if base_ptr % align_bytes != 0:
            raise RuntimeError(
                f"LazyMemoryAllocator buffer base {base_ptr:#x} is not aligned "
                f"to align_bytes={align_bytes} (remainder "
                f"{base_ptr % align_bytes})."
            )

        # Create the tensor memory allocator
        self._allocator = TensorMemoryAllocator(
            tensor=self._buffer,
            align_bytes=align_bytes,
            init_address_space=self._curr_size,
        )

        # Get the address manager
        # NOTE(ApostaC): this assumes the tensor memory allocator owns the address
        # manager, which creates extra coupling in the code.
        # NOTE(ApostaC): this also assumes that the behavior of the allocation is
        # completely determined by the address manager.
        self._address_manager = self._allocator.address_manager

        # Deferred-pinning state, all guarded by _init_lock:
        # _pin_device is bound once by the first ensure_pinning() and never
        # rebound; _pinning_started/_closed order pinning against close().
        self._pin_device: int | torch.device | None = None
        self._pinning_started = False
        self._closed = False
        self._init_lock = threading.Lock()

        # The expansion thread is created here but started only after the
        # initial pinning is bound to a device.
        self._stop_expand = threading.Event()
        self._expand_thread = threading.Thread(
            target=self._expand_worker, daemon=True, name="lazy-mem-expand-thread"
        )

    # Public methods
    def ensure_pinning(self, device: int | torch.device) -> None:
        """
        Pin the initial chunk on ``device`` and start background expansion.

        Idempotent and thread-safe: only the first call pins and binds the
        device; subsequent calls (and calls after :meth:`close`) are no-ops.
        A call racing the initial pin blocks until it completes.

        Args:
            device (int | torch.device): Device whose CUDA context the pinned
                host pool is bound to. Typically the worker's current device.
        """
        with self._init_lock:
            if self._pinning_started or self._closed:
                return
            self._pin_device = device
            self._pin_memory_chunk(0, self._curr_size)
            self._pinning_started = True
            self._expand_thread.start()

    def warm_up(self, device: int | torch.device) -> None:
        """Start pinning on ``device`` in a background thread.

        Non-blocking counterpart of :meth:`ensure_pinning` for callers on
        latency-sensitive paths (e.g. worker registration): the calling
        thread returns immediately while the initial chunk is pinned and
        background expansion starts. Idempotent: once pinning has started
        (or after :meth:`close`), no thread is spawned and this is a no-op.
        An allocation racing the background pin blocks until it completes.

        Args:
            device (int | torch.device): Device whose CUDA context the
                pinned host pool is bound to.
        """
        with self._init_lock:
            if self._pinning_started or self._closed:
                return
        threading.Thread(
            target=self.ensure_pinning,
            args=(device,),
            name="lazy-allocator-warm-up",
            daemon=True,
        ).start()

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            A memory object, or ``None`` if the committed address space is full.

        Note:
            The first allocation (across all threads) pins the initial chunk,
            which blocks the caller and creates a CUDA context on the calling
            thread's current device unless :meth:`ensure_pinning` already ran.
        """
        self._ensure_pinned_for_use()
        obj = self._allocator.allocate(shapes, dtypes, fmt, allocator_type)
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        if obj is not None:
            obj.parent_allocator = self
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate a batch of objects from the lazily pinned memory pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate for each object.
            dtypes: Logical tensor dtype or dtypes to allocate for each object.
            batch_size: Number of memory objects to allocate.
            fmt: Memory format stored in the returned metadata.
            allocator_type: Optional allocator type string.

        Returns:
            Memory objects for the batch, or ``None`` if allocation fails.

        Note:
            The first allocation (across all threads) pins the initial chunk,
            which blocks the caller and creates a CUDA context on the calling
            thread's current device unless :meth:`ensure_pinning` already ran.
        """
        self._ensure_pinned_for_use()
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        ret = self._allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

        if ret is None:
            return ret

        for obj in ret:
            obj.parent_allocator = self
        return ret

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free one memory object back to the lazy allocator.

        Args:
            memory_obj: The memory object to free.
            allocator_type: Optional allocator type string.
        """
        self._allocator.free(memory_obj, allocator_type)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free a batch of memory objects back to the lazy allocator.

        Args:
            memory_objs: Memory objects to free.
            allocator_type: Optional allocator type string.
            update_stats: Whether to update allocator statistics.
        """
        self._allocator.batched_free(memory_objs, allocator_type, update_stats)

    def close(self) -> None:
        """Stop background expansion and release pinned or NUMA memory.

        Thread-safe and idempotent. Holds the init lock so a close racing a
        concurrent first :meth:`ensure_pinning` is serialized against it, and
        marks the allocator closed so any later pinning attempt is a no-op.
        If pinning never happened there is nothing to stop or unpin.

        Note:
            Allocations issued after close are not blocked; they proceed
            against the unpinned (and, for NUMA, freed) buffer.
        """
        with self._init_lock:
            self._closed = True

            # Stop the expansion thread and unpin only if pinning started.
            # The expansion thread never takes _init_lock, so joining it
            # while holding the lock cannot deadlock.
            if self._pinning_started:
                self._stop_expand.set()
                self._expand_thread.join()

                # Unpin in the same device context the chunks were pinned in
                with torch_dev.device(self._pin_device):
                    for ptr, size in self._pin_record:
                        current_device_spec.unpin_memory(ptr)
                self._pin_record.clear()

            # Free the underlying buffer if using NUMA allocation
            if self._use_numa:
                device_ops.free_numa_ptr(self._buffer.data_ptr(), self._final_size)
                self._use_numa = False

    def memcheck(self) -> bool:
        """Return whether the delegated tensor allocator is consistent."""
        return self._allocator.memcheck()

    def get_underlying_buffer(self) -> torch.Tensor:
        """
        Get the underlying buffer tensor. Will be used by RDMA registrations.
        """
        return self._buffer

    def get_address_manager(self) -> AddressManager:
        """
        Get the address manager used by this allocator.
        """
        return self._address_manager

    # Helper functions
    def _ensure_pinned_for_use(self) -> None:
        """
        Pin the pool on the calling thread's current device if not yet pinned.

        Fallback for callers that never invoke :meth:`ensure_pinning`
        explicitly: the first allocation binds pinning to whatever device is
        current on the calling thread.
        """
        if not self._pinning_started:
            device = torch_dev.current_device() if torch_dev.is_available() else 0
            self.ensure_pinning(device)

    def _pin_memory_chunk(self, offset: int, size: int) -> None:
        """
        Pin a chunk of memory inside the bound device's context.

        Caller must ensure ``_pin_device`` is already bound (i.e. only call
        this from ``ensure_pinning`` or the expansion thread it starts).

        Args:
            offset (int): Offset in the buffer to pin.
            size (int): Size of the memory chunk in bytes.
        """
        assert offset & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Offset must be aligned to PIN_CHUNK_SIZE"
        )
        assert size & (self.PIN_CHUNK_SIZE - 1) == 0, (
            "Size must be aligned to PIN_CHUNK_SIZE"
        )
        assert offset + size <= self._final_size, "Pinning exceeds buffer size"

        ptr = self._buffer.data_ptr() + offset
        # Pin inside the bound device's context so the CUDA context the
        # registration creates lands on that device, not the thread default.
        # Use flag: cudaHostRegisterMapped (0x02)
        with torch_dev.device(self._pin_device):
            pinned = current_device_spec.pin_memory(ptr, size, 2)
        if not pinned:
            logger.warning(
                "pin_memory failed for chunk at ptr=%#x size=%d; "
                "DMA performance may be degraded",
                ptr,
                size,
            )
        else:
            self._pin_record.append((ptr, size))

    def _commit_expansion(self, expand_size: int) -> None:
        """
        Call sbrk in the address manager to commit the expansion.
        """
        self._address_manager.sbrk(expand_size)

    def _log_expansion_progress(self, expanded_since_last_log: int) -> None:
        """
        Log the cumulative expansion progress since the last log.
        """
        percent = 100.0 * self._curr_size / self._final_size
        logger.info(
            "LazyMemoryAllocator: Expanded %s MB pinned memory, "
            "now total is %s MB / %s MB (%.1f%%)",
            expanded_since_last_log >> 20,
            self._curr_size >> 20,
            self._final_size >> 20,
            percent,
        )

    def _expand_worker(self) -> None:
        """
        Background worker to expand the pinned memory.
        """
        last_commit_size = self._curr_size
        last_log_size = self._curr_size
        while self._curr_size < self._final_size and not self._stop_expand.is_set():
            # Expand chunk by chunk and commit
            for i in range(self.COMMIT_SIZE // self.PIN_CHUNK_SIZE):
                if self._curr_size >= self._final_size:
                    break
                self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                self._curr_size += self.PIN_CHUNK_SIZE

            expand_size = self._curr_size - last_commit_size
            self._commit_expansion(expand_size)
            last_commit_size = self._curr_size

            # Log every LOG_INTERVAL bytes, and always on the final commit.
            expanded_since_last_log = self._curr_size - last_log_size
            if (
                expanded_since_last_log >= self.LOG_INTERVAL
                or self._curr_size >= self._final_size
            ):
                self._log_expansion_progress(expanded_since_last_log)
                last_log_size = self._curr_size
