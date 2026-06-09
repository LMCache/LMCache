# SPDX-License-Identifier: Apache-2.0
"""Global cuFile data-path context for the GDS L1 tier.

A single :class:`GDSContext` per worker process owns the slab file, its
``cuFileHandleRegister`` handle, the registered GPU staging buffer(s), and
the stream-ordered ``cuFileReadAsync`` / ``cuFileWriteAsync`` submissions.
It is created once at server startup via :func:`initialize_gds_context` and
reached elsewhere via :func:`get_gds_context`:

- :meth:`GDSContext.register_gpu_buffer` is called from
  ``GPUCacheContext.__init__`` (once per staging buffer).
- :meth:`GDSContext.write_async` / :meth:`GDSContext.read_async` are called
  from ``lmcache_memcpy_async`` (per chunk), moving bytes between a slice of
  the registered GPU buffer and the chunk's ``(offset, size)`` in the slab.

There is no POSIX fallback: if GDS / cuFile is unavailable, construction
fails loudly. The slab file is created and cleared on init, so its contents
do not survive a restart (GDS L1 is treated like DRAM).
"""

# Standard
from typing import Optional
import bisect
import functools
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import _cufile_async as ca
from lmcache.v1.memory_management import GDSMemoryObject

logger = init_logger(__name__)

_SLAB_FILENAME = "lmcache_gds_slab.bin"
_CUFILE_ALIGNMENT = 4096
# cuFile submissions to accumulate before recording a completion event and
# draining finished ones (keeps the live submission set bounded).
_SUBMISSION_CHECKPOINT_EVERY = 64


class GDSContext:
    """Per-process cuFile context owning the slab file and its DMA path.

    Constructed empty (the global singleton always exists); :meth:`initialize`
    does the heavy setup (create/clear the slab, register the cuFile handle)
    and flips :attr:`initialized`. Until then GDS L1 is off and only
    :attr:`initialized` should be consulted: ``register_gpu_buffer`` is a no-op
    on an uninitialized context, and ``read_async`` / ``write_async`` raise.
    """

    #: Whether :meth:`initialize` has completed (GDS L1 is active).
    initialized: bool = False

    def __init__(self) -> None:
        # ``initialized`` defaults to False via the class attribute; it is
        # flipped to True by ``initialize``.
        self._slab_size = 0
        self._slab_path = ""
        self._slab_handle: Optional[ca.AsyncHandle] = None
        # The CUDA stream registered once with cuFile and held (as its raw
        # ``CUstream`` int) for the context's lifetime; deregistered on close.
        self._registered_stream_handle: Optional[int] = None
        self._stream_lock = threading.Lock()
        # In-flight submissions, released once a CUDA event recorded after them
        # on the cuFile stream completes (see ``_checkpoint_submissions_locked``).
        self._uncommitted_submissions: list[ca.Submission] = []
        self._inflight_submissions: list[tuple[torch.Event, list[ca.Submission]]] = []
        self._ops_since_checkpoint = 0
        self._submissions_lock = threading.Lock()
        # One entry per cuFile-registered region, parallel lists kept sorted by
        # ``_base_ptrs`` ascending for ``_resolve_buffer``'s bisect.
        self._buffers: list[torch.Tensor] = []
        self._base_ptrs: list[int] = []
        self._nbytes: list[int] = []

    def initialize(self, config: GdsL1Config) -> None:
        """Create + clear the slab and register it with cuFile.

        Called once at server startup; not safe to call twice (a second call
        re-opens the slab and leaks the prior fd + cuFile handle).

        Args:
            config: The GDS tier config. ``size_in_bytes`` sizes the
                preallocated slab file (rounded up to the 4 KiB cuFile/O_DIRECT
                alignment); the slab lives at
                ``<file_location>/lmcache_gds_slab.bin`` (one shared slab per
                process, used by all GPU instances); and ``use_direct_io`` opens
                the slab with ``O_DIRECT``.

        Raises:
            Exception: Whatever ``cufile`` raises if GDS is unavailable --
                there is no POSIX fallback. (``config`` is already validated
                by :class:`GdsL1Config`.)
        """
        self._slab_size = (config.size_in_bytes + _CUFILE_ALIGNMENT - 1) & ~(
            _CUFILE_ALIGNMENT - 1
        )

        # One shared slab per process (the GDSContext is a process-global
        # singleton used by every GPU instance).
        selected = config.file_location
        os.makedirs(selected, exist_ok=True)
        self._slab_path = os.path.join(selected, _SLAB_FILENAME)

        self._open_and_register_slab(config.use_direct_io)
        self.initialized = True

    # --- Public API ---------------------------------------------------

    def register_gpu_buffer(self, buffer: torch.Tensor, slot_bytes: int) -> None:
        """Register a GPU staging buffer with cuFile, split into slots.

        No-op when the context is uninitialized (GDS L1 is off), so callers can
        invoke it unconditionally from ``GPUCacheContext.__init__``.

        ``buffer`` is split into ``slot_bytes``-sized regions, each registered
        separately. The split keeps every registration under nvidia-fs's 16 MiB
        ``cuFileBufRegister`` cap and -- because callers later hand per-slot
        slices to :meth:`read_async` / :meth:`write_async` -- guarantees each
        such slice lies within a single registration (so :meth:`_resolve_buffer`
        can map it).

        Args:
            buffer: The contiguous CUDA staging buffer (chunk-major), whose
                size is a multiple of ``slot_bytes``.
            slot_bytes: Size of one chunk slot in bytes; each slot must be
                4 KiB-aligned and <= 16 MiB.
        """
        if not self.initialized:
            return
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        for start in range(0, nbytes, slot_bytes):
            self._register_region(buf[start : start + slot_bytes])

    def write_async(
        self, memory_obj: GDSMemoryObject, gpu_buffer: torch.Tensor
    ) -> None:
        """DMA ``gpu_buffer`` into ``memory_obj``'s slab region (no per-call sync).

        Args:
            memory_obj: Target chunk; its ``slab_offset`` / ``get_size()`` give
                the file offset and transfer length.
            gpu_buffer: A slice of a registered staging buffer holding the
                bytes to write.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` is outside every registered region or
                smaller than the chunk.
        """
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.get_size()
        capacity = gpu_buffer.numel() * gpu_buffer.element_size()
        if nbytes > capacity:
            raise ValueError(
                f"GDSContext.write_async: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {capacity}"
            )
        self._slab_write(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)

    def read_async(self, memory_obj: GDSMemoryObject, gpu_buffer: torch.Tensor) -> None:
        """DMA ``memory_obj``'s slab region into ``gpu_buffer`` (no per-call sync).

        Args:
            memory_obj: Source chunk; its ``slab_offset`` / ``get_size()`` give
                the file offset and transfer length.
            gpu_buffer: A slice of a registered staging buffer to receive the
                bytes.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` is outside every registered region or
                smaller than the chunk.
        """
        base_ptr, dev_offset = self._resolve_buffer(gpu_buffer)
        nbytes = memory_obj.get_size()
        capacity = gpu_buffer.numel() * gpu_buffer.element_size()
        if nbytes > capacity:
            raise ValueError(
                f"GDSContext.read_async: chunk size {nbytes} exceeds gpu_buffer "
                f"capacity {capacity}"
            )
        self._slab_read(memory_obj.slab_offset, nbytes, dev_offset, base_ptr)

    def close(self) -> None:
        """Sync the stream, deregister cuFile state, and close the slab handle."""
        if self._buffers:
            torch_dev.synchronize(device=self._buffers[0].device)
        with self._submissions_lock:
            self._uncommitted_submissions = []
            self._inflight_submissions = []
            self._ops_since_checkpoint = 0
        if self._buffers:
            for buf in self._buffers:
                try:
                    ca.deregister_buffer(buf)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_buffer: %s", e)
            self._buffers.clear()
            self._base_ptrs.clear()
            self._nbytes.clear()
        with self._stream_lock:
            if self._registered_stream_handle is not None:
                try:
                    ca.deregister_stream(self._registered_stream_handle)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_stream: %s", e)
                self._registered_stream_handle = None
        if self._slab_handle is not None:
            try:
                self._slab_handle.close()
            except Exception as e:
                logger.warning("GDSContext.close: slab handle close failed: %s", e)
            self._slab_handle = None

    # --- Internal -----------------------------------------------------

    def _open_and_register_slab(self, use_direct_io: bool) -> None:
        """Create + clear the slab file and register it with cuFile.

        The file is truncated to empty then preallocated to ``self._slab_size``
        so its contents never survive a restart and ``cuFileWriteAsync`` never
        has to grow it.

        Args:
            use_direct_io: If ``True``, open with ``O_DIRECT`` (required for the
                cuFile GDS DMA fast path on ext4).
        """
        # Create, truncate, and fallocate via a regular (non-O_DIRECT) fd.
        creator_fd = os.open(
            self._slab_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644
        )
        try:
            os.posix_fallocate(creator_fd, 0, self._slab_size)
        finally:
            os.close(creator_fd)
        flags = os.O_RDWR
        if use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(self._slab_path, flags)
        try:
            # Third Party
            from cufile.bindings import cuFileHandleRegister

            handle = cuFileHandleRegister(fd)
        except Exception:
            os.close(fd)
            raise
        self._slab_handle = ca.AsyncHandle.from_fd(
            fd, handle, self._slab_path, writable=True
        )
        logger.info(
            "GDSContext: slab created at %s (%.1f GiB, O_DIRECT=%s), cuFile "
            "handle registered",
            self._slab_path,
            self._slab_size / (1 << 30),
            use_direct_io,
        )

    def _register_region(self, buffer: torch.Tensor) -> None:
        """Register one <=16 MiB region with cuFile and the CUDA stream.

        Args:
            buffer: A CUDA staging-buffer slot, 4 KiB-aligned in size and no
                larger than 16 MiB.

        Raises:
            ValueError: If the region is not 4 KiB-aligned or exceeds 16 MiB.
        """
        nbytes = buffer.numel() * buffer.element_size()
        if nbytes % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                f"_register_region: region size {nbytes} is not a multiple "
                f"of {_CUFILE_ALIGNMENT} (cuFile requires 4 KiB alignment)."
            )
        if nbytes > 16 * 1024 * 1024:
            raise ValueError(
                f"_register_region: a single cuFileBufRegister is capped at "
                f"16 MiB on the standard nvidia-fs config; got {nbytes} bytes. "
                "Reduce the chunk size."
            )
        base = buffer.data_ptr()
        ca.register_buffer(buffer)
        self._ensure_stream_registered()
        idx = bisect.bisect_left(self._base_ptrs, base)
        self._buffers.insert(idx, buffer)
        self._base_ptrs.insert(idx, base)
        self._nbytes.insert(idx, nbytes)
        logger.info(
            "GDSContext: registered %d bytes at 0x%x via cuFile "
            "(total registrations: %d)",
            nbytes,
            base,
            len(self._buffers),
        )

    def _ensure_stream_registered(self) -> None:
        """Register the caller's current CUDA stream with cuFile (idempotent).

        The first call records the stream that ``cuFileReadAsync`` /
        ``cuFileWriteAsync`` run on; later calls are no-ops. cuFile mishandles
        repeated register/deregister cycles on the same stream, so we register
        exactly once and keep it for the context's lifetime.
        """
        with self._stream_lock:
            if self._registered_stream_handle is not None:
                return
            raw_stream = torch_dev.current_stream().cuda_stream
            ca.register_stream(raw_stream)
            self._registered_stream_handle = raw_stream

    def _resolve_buffer(self, gpu_buffer: torch.Tensor) -> tuple[int, int]:
        """Find which registered region ``gpu_buffer`` belongs to.

        Returns:
            ``(base_ptr, dev_offset)`` where ``base_ptr`` is the matching
            registration's base pointer and ``dev_offset`` is
            ``gpu_buffer.data_ptr() - base_ptr``.

        Raises:
            RuntimeError: If no buffer has been registered.
            ValueError: If ``gpu_buffer`` does not lie entirely inside any
                single registered region.
        """
        ptr = gpu_buffer.data_ptr()
        idx = bisect.bisect_right(self._base_ptrs, ptr) - 1
        if idx < 0:
            raise ValueError(
                f"GDSContext: gpu_buffer pointer 0x{ptr:x} is below every "
                f"registered region"
            )
        base = self._base_ptrs[idx]
        nbytes = self._nbytes[idx]
        offset = ptr - base
        if offset < 0 or offset >= nbytes:
            raise ValueError(
                f"GDSContext: gpu_buffer pointer 0x{ptr:x} is outside every "
                f"registered region (closest: [0x{base:x}, 0x{base + nbytes:x}))"
            )
        return base, offset

    def _slab_read(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one ``cuFileReadAsync`` against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_read: slab handle not open")
        # Submit on the caller's current stream
        stream_handle = torch_dev.current_stream().cuda_stream
        sub = self._slab_handle.read_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )
        self._record_submission(sub)

    def _slab_write(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one ``cuFileWriteAsync`` against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_write: slab handle not open")
        # Submit on the caller's current stream
        stream_handle = torch_dev.current_stream().cuda_stream
        sub = self._slab_handle.write_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )
        self._record_submission(sub)

    def _record_submission(self, sub: "ca.Submission") -> None:
        """Track one in-flight cuFile submission, draining completed batches.

        The submission's ctypes storage must outlive the stream op, so
        submissions are accumulated and only released once a CUDA event
        recorded after them on the cuFile stream reports complete. A checkpoint
        is taken every ``_SUBMISSION_CHECKPOINT_EVERY`` ops to keep the live set
        bounded.
        """
        with self._submissions_lock:
            self._uncommitted_submissions.append(sub)
            self._ops_since_checkpoint += 1
            if self._ops_since_checkpoint >= _SUBMISSION_CHECKPOINT_EVERY:
                self._checkpoint_submissions_locked()

    def _checkpoint_submissions_locked(self) -> None:
        """Close the current submission batch and release completed ones.

        Records a CUDA event on the current (transfer) stream marking the point
        after every uncommitted submission, then drops any earlier batch whose
        event has completed (non-blocking ``query()``). Callers submit on the
        current stream, so the event orders correctly behind those submissions.

        Must be called while holding ``self._submissions_lock``.
        """
        if self._uncommitted_submissions:
            event = torch_dev.Event()
            event.record()
            self._inflight_submissions.append((event, self._uncommitted_submissions))
            self._uncommitted_submissions = []
        self._ops_since_checkpoint = 0
        self._inflight_submissions = [
            (event, subs)
            for (event, subs) in self._inflight_submissions
            if not event.query()
        ]


@functools.cache
def get_gds_context() -> GDSContext:
    """Return the process-global :class:`GDSContext` singleton.

    The singleton always exists; it is created empty on first access (memoized
    by ``functools.cache``). Callers that run in both the GDS and non-GDS
    configurations should consult :attr:`GDSContext.initialized` to tell whether
    GDS L1 is active (it is ``False`` until :func:`initialize_gds_context` runs).
    """
    return GDSContext()


def initialize_gds_context(config: Optional[GdsL1Config]) -> GDSContext:
    """Set up the process-global :class:`GDSContext`.

    Called once at server startup. ``config=None`` (GDS L1 disabled) leaves the
    singleton uninitialized; otherwise the slab is created and registered.

    Args:
        config: The GDS tier config (slab size, file locations, DMA mode), or
            ``None`` when GDS L1 is disabled.

    Returns:
        The process-global :class:`GDSContext` (initialized only when ``config``
        is not ``None``).
    """
    context = get_gds_context()
    if config is not None:
        context.initialize(config)
    return context
