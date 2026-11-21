# SPDX-License-Identifier: Apache-2.0
"""GPUDirect Storage data path owned by the active GDS L1 manager.

The manager opens its slab before GPU cache contexts register staging buffers.
Transfers resolve the context from the memory object's owner, and manager
shutdown drains submissions before closing the slab. One slab per process is
supported because native stream registration is process-wide. Unowned legacy
objects retain the separately initialized singleton path.

The backend clears the slab at startup, so GDS L1 does not survive a restart.
Construction fails if the requested GDS backend is unavailable.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
import bisect
import enum
import functools
import threading

# Third Party
import torch

# First Party
from lmcache.v1.common.buffer_registration import SlottedBufferRegistration
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.platform import stream as platform_stream

logger = init_logger(__name__)

_GDS_ALIGNMENT = 4096
# LMCache layout choice for registration bases. The cuFileStreamRegister 0x8
# contract requires 4 KiB-aligned offsets and lengths (NVIDIA cuFile API §3.4.1).
GDS_SLOT_ALIGNMENT = 64 * 1024


def gds_group_stride(nbytes: int) -> int:
    """Return the space occupied by a group before the next aligned group."""
    return (nbytes + GDS_SLOT_ALIGNMENT - 1) // GDS_SLOT_ALIGNMENT * GDS_SLOT_ALIGNMENT


# This context registers staging buffers in 16 MiB regions. This is our
# registration geometry, not a universal limit of the backend API.
_MAX_GDS_REGION = 16 * 1024 * 1024
# GDS submissions to accumulate before recording a completion event and
# draining finished ones (keeps the live submission set bounded).
_SUBMISSION_CHECKPOINT_EVERY = 64

# L1 topology is established before GPU cache contexts register their buffers.
_owner_contexts: dict[int, "GDSContext"] = {}
_owner_contexts_lock = threading.Lock()


def initialize_l1_gds_context(owner: int, config: GdsL1Config) -> None:
    """Open the slab owned by an L1 before GPU staging buffers are registered.

    Args:
        owner: Process-local L1 identity stamped on its memory objects.
        config: Slab location and backend settings.

    Raises:
        ValueError: Another GDS slab is already active in this process.
        RuntimeError: The backend cannot initialize its slab or driver.
        OSError: The backing storage cannot be opened.
    """
    with _owner_contexts_lock:
        # cuFile stream registration is process-wide, not per slab.
        if _owner_contexts or get_gds_context().initialized:
            raise ValueError("Only one GDS L1 slab is supported per process")
        context = GDSContext()
        context.initialize(config)
        _owner_contexts[owner] = context


def get_l1_gds_context(owner: int | None) -> "GDSContext":
    """Resolve an object's slab by owner; unowned legacy objects use the singleton.

    Args:
        owner: Identity returned by MemoryObj.get_l1_manager().

    Returns:
        The owning slab context.

    Raises:
        ValueError: A tagged object has no registered GDS owner.
    """
    if owner is None:
        return get_gds_context()
    with _owner_contexts_lock:
        if owner not in _owner_contexts:
            raise ValueError(f"Unknown GDS L1 owner: {owner}")
        return _owner_contexts[owner]


def close_l1_gds_context(owner: int) -> None:
    """Drain and close an L1's slab before unregistering its owner.

    Args:
        owner: Process-local L1 identity. Repeated closes are harmless.

    Raises:
        RuntimeError: Device synchronization fails; the context remains registered.
    """
    with _owner_contexts_lock:
        entry = _owner_contexts.get(owner)
        if entry is not None:
            entry.close()
            del _owner_contexts[owner]


def register_gds_gpu_buffer(
    buffer: torch.Tensor,
    *,
    object_sizes: tuple[int, ...] = (),
    stream_owner: object | None = None,
) -> None:
    """Register a GPU staging buffer with each live slab on its current stream.

    Args:
        buffer: Contiguous, aligned staging buffer, after L1 initialization.
        object_sizes: Logical bytes per group in the aligned layout, or empty.
    """
    with _owner_contexts_lock:
        contexts = [get_gds_context(), *_owner_contexts.values()]
    for context in contexts:
        context.register_gpu_buffer(
            buffer, object_sizes=object_sizes, stream_owner=stream_owner
        )


def deregister_gds_gpu_buffer(buffer: torch.Tensor) -> None:
    """Drain and unregister a staging buffer from every slab before freeing it.

    Args:
        buffer: Buffer previously passed to register_gds_gpu_buffer.
    """
    with _owner_contexts_lock:
        contexts = [get_gds_context(), *_owner_contexts.values()]
    for context in contexts:
        context.deregister_gpu_buffer(buffer)


class GDSBufferRegistration(SlottedBufferRegistration):
    """Connect cache-context staging buffers to the server's live GDS slabs."""

    def slot_alignment(self, device_type: str) -> int:
        if device_type != "cuda":
            return 0
        with _owner_contexts_lock:
            contexts = [get_gds_context(), *_owner_contexts.values()]
            return GDS_SLOT_ALIGNMENT if any(c.io_concurrency for c in contexts) else 0

    def register(
        self,
        buffer: torch.Tensor,
        *,
        object_sizes: tuple[int, ...] = (),
        stream_owner: object | None = None,
    ) -> None:
        register_gds_gpu_buffer(
            buffer, object_sizes=object_sizes, stream_owner=stream_owner
        )

    def deregister(self, buffer: torch.Tensor) -> None:
        deregister_gds_gpu_buffer(buffer)

    def deregister_stream(self, device: object, stream: object) -> None:
        with _owner_contexts_lock:
            contexts = [get_gds_context(), *_owner_contexts.values()]
        for context in contexts:
            context.deregister_stream(device, stream)


class SlabDirection(enum.Enum):
    """Direction of a GDS slab transfer. GPUDirect DMAs run straight between GPU
    memory and slab storage (no host buffer), so directions are storage I/O
    (READ/WRITE), not host<->device (H2D/D2H)."""

    READ = enum.auto()  # slab storage -> GPU buffer
    WRITE = enum.auto()  # GPU buffer -> slab storage


@dataclass
class _StreamSubmissions:
    """Per-stream GDS submissions, kept alive until their DMA has run.

    Submissions accumulate in ``uncommitted``, move to ``inflight`` behind a
    GPU event on the stream, and drop once it completes. Per-stream because an
    event only orders work on its own stream.
    """

    uncommitted: list[Submission] = field(default_factory=list)
    inflight: list[tuple[platform_stream.CompletionEvent, list[Submission]]] = field(
        default_factory=list
    )
    ops_since_checkpoint: int = 0


@dataclass(frozen=True)
class _SlotIO:
    """Physical DMA plan and owned write padding for a fixed-size staging group."""

    logical_size: int
    physical_size: int
    tail: torch.Tensor
    # Registered GPU base, offset within that registration, and transfer length.
    segments: tuple[tuple[int, int, int], ...]


class GDSContext:
    """Per-process GDS context owning the slab file and its DMA path.

    A context stays inert until :meth:`initialize` opens the slab and registers
    its GDS handle. While off, ``register_gpu_buffer`` is a no-op.
    """

    #: Whether :meth:`initialize` has completed (GDS L1 is active).
    initialized: bool = False

    def __init__(self, backend: Optional[GDSBackend] = None) -> None:
        # ``initialized`` defaults to False via the class attribute; it is
        # flipped to True by ``initialize``.
        self._slot_io: dict[int, _SlotIO] = {}
        self._slot_streams: dict[int, object] = {}
        self._slab_size = 0
        self._slab_path = ""
        self._backend = backend
        self._slab_handle: Optional[GDSHandle] = None
        # Per-stream in-flight submissions (keyed by raw GPU stream), released
        # once a GPU event recorded on that stream completes. Guarded by
        # ``_submissions_lock`` (see ``_record_submission``).
        self._submissions_lock = threading.Lock()
        self._submissions: dict[int, _StreamSubmissions] = {}
        # Registry of GDS-registered GPU regions and the streams they run on
        self._registry_lock = threading.Lock()
        self._buffers: list[torch.Tensor] = []
        self._base_ptrs: list[int] = []
        self._nbytes: list[int] = []
        self._main_streams: dict[int, object | None] = {}
        self._io_lock = threading.RLock()
        self._io_concurrency = 0

    def initialize(self, config: GdsL1Config) -> None:
        """Create + clear the slab and register it with the GDS library.

        Args:
            config: GDS tier config. ``size_in_bytes`` sizes the preallocated
                slab (rounded up to 4 KiB). The backend interprets
                ``file_location`` and prepares the corresponding file or device.

        Raises:
            ValueError: If the backend rejects the environment, location, or size.
            RuntimeError: If this context already owns an initialized slab.
            Exception: Whatever the GDS library raises if GDS is unavailable.
        """
        if self.initialized:
            raise RuntimeError("GDS context is already initialized")
        self._slab_size = (config.size_in_bytes + _GDS_ALIGNMENT - 1) & ~(
            _GDS_ALIGNMENT - 1
        )
        if self._backend is None:
            self._backend = create_backend(config.backend)
        else:
            self._backend.validate_environment()
        try:
            self._slab_handle = self._backend.open_slab(
                config.file_location, self._slab_size, config.use_direct_io
            )
        except Exception:
            try:
                self._backend.close_driver()
            except Exception as cleanup_error:
                logger.warning("GDS driver cleanup failed: %s", cleanup_error)
            raise
        if (
            self.backend.name == "cufile"
            and config.use_direct_io
            and config.align_bytes > 0
            and config.align_bytes % _GDS_ALIGNMENT == 0
        ):
            self._io_concurrency = config.io_concurrency
        self._slab_path = self._slab_handle.path
        logger.info(
            "GDSContext: %s slab opened at %s (%.1f GiB)",
            self._backend.name,
            self._slab_path,
            self._slab_size / (1 << 30),
        )
        self.initialized = True

    # --- Public API ---------------------------------------------------

    @property
    def backend(self) -> GDSBackend:
        """Return this context's backend, or raise before one is configured."""
        if self._backend is None:
            raise RuntimeError("GDS context has no backend")
        return self._backend

    @property
    def io_concurrency(self) -> int:
        """Return the enabled asynchronous slot count, or zero for the default path."""
        return self._io_concurrency

    def has_aligned_slot(self, buffer: torch.Tensor) -> bool:
        return buffer.data_ptr() in self._slot_io

    @contextmanager
    def io_scope(self) -> Iterator[None]:
        """Serialize bounded asynchronous groups across this slab's cache contexts.

        The caller holds its cache-context lease first and drains every submitted
        stream before leaving this scope. Close takes the same lock, so native
        buffers and the slab handle survive until the group has finished.

        Raises:
            RuntimeError: Slab teardown finished before this group acquired its scope.
        """
        with self._io_lock:
            if not self.initialized:
                raise RuntimeError("GDS context is closed")
            yield

    def register_stream(self, device: object, stream: object) -> None:
        """Register an owned aligned slot stream inside its transfer scope."""
        raw = platform_stream.stream_handle(device, stream)
        with self._registry_lock:
            if raw in self._main_streams:
                raise ValueError(
                    "A main stream cannot be registered as an aligned slot"
                )
            if raw in self._slot_streams:
                if self._slot_streams[raw] is not stream:
                    raise ValueError("GDS slot stream belongs to another owner")
                return
            self.backend.register_aligned_stream(raw)
            self._slot_streams[raw] = stream

    def deregister_stream(self, device: object, stream: object) -> None:
        """Drain an auxiliary stream before releasing its native registration.

        Args:
            device: Device owning the stream.
            stream: Stream no longer used by the cache context.

        The cache-context lease excludes new submissions. Close uses io_scope's
        lock, so slab teardown cannot overlap this native cleanup. Native
        deregistration failures are logged and retained for slab-close retry,
        allowing other drained resources to be released.

        Raises:
            RuntimeError: Synchronization or deferred I/O validation failed; the
                caller must retain buffers whose completion is not established.
        """
        with self._io_lock:
            if not self.initialized:
                return
            self.synchronize_and_check(device, stream)
            raw_stream = platform_stream.stream_handle(device, stream)
            with self._registry_lock:
                if raw_stream in self._slot_streams:
                    try:
                        self.backend.deregister_stream(raw_stream)
                    except RuntimeError:
                        logger.exception(
                            "GDSContext: deregister_stream failed after stream drain"
                        )
                    else:
                        del self._slot_streams[raw_stream]

    def synchronize_and_check(self, device: object, stream: object) -> None:
        """Drain this stream and validate every retained deferred byte count.

        Args:
            device: Device owning the submitted staging buffer.
            stream: Issuing stream, called by its serialized affinity handler.

        Raises:
            RuntimeError: A completed I/O reports an error or short byte count.

        Call before scatter or successful publication. This scoped wait must
        not run in the MP transport event loop or in a CUDA host callback.
        """
        platform_stream.synchronize_stream(device, stream)
        raw_stream = platform_stream.stream_handle(device, stream)
        with self._submissions_lock:
            submissions = self._submissions.pop(raw_stream, None)
        if submissions is not None:
            for sub in submissions.uncommitted:
                if sub.bytes_done != sub.size.value:
                    raise RuntimeError(
                        f"GDS I/O at {sub.file_offset.value}: expected "
                        f"{sub.size.value} bytes, completed {sub.bytes_done}"
                    )

    def register_gpu_buffer(
        self,
        buffer: torch.Tensor,
        *,
        object_sizes: tuple[int, ...] = (),
        stream_owner: object | None = None,
    ) -> None:
        """Register a staging buffer (and its stream) with the GDS library.

        Registered as contiguous <=16 MiB regions; :meth:`transfer_async`
        splits transfers at these boundaries. For an aligned layout,
        concurrent I/O also splits at slot boundaries so independently owned
        slots do not share a registered region.

        Args:
            buffer: Contiguous GPU staging buffer, 4 KiB-aligned in size.
            object_sizes: Logical group sizes for aligned cuFile slots, or empty.
            stream_owner: Keeps a privately allocated main stream alive.

        Raises:
            ValueError: The supplied group sizes do not partition the buffer.
        """
        if not self.initialized:
            return
        slot_bytes = sum(map(gds_group_stride, object_sizes)) or None
        plans = (
            self._plan_slot_io(buffer, slot_bytes, object_sizes) if object_sizes else {}
        )
        stream = platform_stream.current_stream(buffer.device)
        raw_stream = platform_stream.stream_handle(buffer.device, stream)
        with self._registry_lock:
            stream_registered = False
            registered_regions: list[torch.Tensor] = []
            try:
                if raw_stream in self._slot_streams:
                    raise ValueError(
                        "An aligned slot cannot be registered as a main stream"
                    )
                if (
                    stream_owner is not None
                    and raw_stream in self._main_streams
                    and self._main_streams[raw_stream] is not stream_owner
                ):
                    raise ValueError("GDS main stream belongs to another owner")
                if raw_stream not in self._main_streams:
                    self.backend.register_stream(raw_stream)
                    self._main_streams[raw_stream] = stream_owner
                    stream_registered = True
                for region in self._buffer_regions(buffer, slot_bytes):
                    self._register_region_locked(region)
                    registered_regions.append(region)
                self._slot_io.update(plans)
            except BaseException:
                for region in reversed(registered_regions):
                    try:
                        self._deregister_region_locked(region)
                    except Exception:
                        logger.exception(
                            "GDSContext: failed to roll back buffer registration"
                        )
                if stream_registered:
                    self._deregister_main_stream_locked(raw_stream)
                raise

    def deregister_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Reverse of :meth:`register_gpu_buffer`: deregister its regions + stream.

        Args:
            buffer: The buffer passed to :meth:`register_gpu_buffer`.
        """
        if not self.initialized:
            return
        stream = platform_stream.current_stream(buffer.device)
        raw_stream = platform_stream.stream_handle(buffer.device, stream)
        # No in-flight DMA on this stream may still reference the buffer.
        platform_stream.synchronize_stream(buffer.device, stream)
        with self._registry_lock:
            lo = buffer.data_ptr()
            for ptr in list(self._slot_io):
                if lo <= ptr < lo + buffer.nbytes:
                    del self._slot_io[ptr]
            start = bisect.bisect_left(self._base_ptrs, lo)
            end = bisect.bisect_left(self._base_ptrs, lo + buffer.nbytes)
            if buffer.nbytes and (start == end or self._base_ptrs[start] != lo):
                raise ValueError(f"GDS buffer at 0x{lo:x} is not registered")
            for region in self._buffers[start:end]:
                self._deregister_region_locked(region)
            if raw_stream in self._main_streams:
                self._deregister_main_stream_locked(raw_stream)
        # Stream is synced above, so its submissions' DMAs are done -- drop them.
        with self._submissions_lock:
            self._submissions.pop(raw_stream, None)

    def transfer_async(
        self,
        memory_obj: GDSMemoryObject,
        gpu_buffer: torch.Tensor,
        direction: SlabDirection,
    ) -> None:
        """DMA a chunk between ``gpu_buffer`` and its slab region.

        ``READ`` pulls slab -> ``gpu_buffer``; ``WRITE`` pushes the reverse.
        Split at registered-region boundaries (each GDS DMA must stay within
        one <=16 MiB region), so any chunk size works. Stream-ordered, no sync.

        Args:
            memory_obj: The chunk; ``slab_offset`` / ``get_size()`` give the
                file offset and length.
            gpu_buffer: A slice of a registered staging buffer; its first
                ``get_size()`` bytes are transferred.
            direction: :attr:`SlabDirection.READ` or ``.WRITE``.
        """
        stream = platform_stream.current_stream(gpu_buffer.device)
        raw_stream = platform_stream.stream_handle(gpu_buffer.device, stream)
        slab_op = (
            self._slab_read if direction is SlabDirection.READ else self._slab_write
        )
        nbytes = memory_obj.get_size()
        buf = gpu_buffer.view(torch.uint8)
        if raw_stream in self._slot_streams:
            plan = self._slot_io.get(buf.data_ptr())
            if (
                plan is None
                or nbytes != plan.logical_size
                or buf.nbytes != nbytes
                or not buf.is_contiguous()
            ):
                raise ValueError("Unexpected object layout on aligned GDS stream")
            if (
                memory_obj.slab_offset % _GDS_ALIGNMENT
                or memory_obj.slab_offset < 0
                or memory_obj.get_physical_size() < plan.physical_size
                or memory_obj.slab_offset + plan.physical_size > self._slab_size
            ):
                raise ValueError("Invalid slab reservation for aligned GDS transfer")
            if direction is SlabDirection.WRITE and plan.physical_size != nbytes:
                plan.tail.zero_()
            file_offset = memory_obj.slab_offset
            for base, offset, length in plan.segments:
                sub = slab_op(file_offset, length, offset, base, raw_stream)
                self._record_submission(sub, gpu_buffer.device, stream, raw_stream)
                file_offset += length
            return

        pos = 0
        while pos < nbytes:
            base_ptr, dev_offset, region_nbytes = self._resolve_buffer(buf[pos:])
            seg_len = min(nbytes - pos, region_nbytes - dev_offset)
            submission = slab_op(
                memory_obj.slab_offset + pos,
                seg_len,
                dev_offset,
                base_ptr,
                raw_stream,
            )
            self._record_submission(
                submission,
                gpu_buffer.device,
                stream,
                raw_stream,
            )
            pos += seg_len

    def close(self) -> None:
        """Wait for the active group before deregistering buffers and the slab."""
        with self._io_lock:
            self._close_slab()

    def _close_slab(self) -> None:
        devices = {str(buffer.device): buffer.device for buffer in self._buffers}
        for device in devices.values():
            platform_stream.synchronize_device(device)
        with self._submissions_lock:
            self._submissions.clear()
        # Deregister any regions/streams still live (per-instance teardown via
        # ``deregister_gpu_buffer`` normally clears these first; this is the
        # shutdown sweep for anything left).
        with self._registry_lock:
            for buf in self._buffers:
                try:
                    self.backend.deregister_buffer(buf)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_buffer: %s", e)
            self._buffers.clear()
            self._base_ptrs.clear()
            self._nbytes.clear()
            for raw_stream in (*self._slot_streams, *self._main_streams):
                try:
                    self.backend.deregister_stream(raw_stream)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_stream: %s", e)
            self._slot_io.clear()
        if self._slab_handle is not None:
            try:
                self._slab_handle.close()
            except Exception as e:
                logger.warning("GDSContext.close: slab handle close failed: %s", e)
            self._slab_handle = None
        if self.initialized:
            try:
                self.backend.close_driver()
            except Exception as e:
                logger.warning("GDSContext.close: driver close failed: %s", e)
                if self.io_concurrency:
                    raise
        self._main_streams.clear()
        self._slot_streams.clear()
        self.initialized = False
        self._io_concurrency = 0

    # --- Internal -----------------------------------------------------

    def _plan_slot_io(
        self,
        buffer: torch.Tensor,
        slot_bytes: int | None,
        object_sizes: tuple[int, ...],
    ) -> dict[int, _SlotIO]:
        """Resolve the DMA segments for each group at registration."""
        if (
            not self.io_concurrency
            or not slot_bytes
            or buffer.dtype != torch.uint8
            or buffer.ndim != 1
            or not buffer.is_contiguous()
            or buffer.data_ptr() % GDS_SLOT_ALIGNMENT
            or buffer.numel() % slot_bytes
            or any(size <= 0 for size in object_sizes)
        ):
            raise ValueError("Invalid aligned GDS staging layout")
        plans = {}
        for slot in range(0, buffer.numel(), slot_bytes):
            group = 0
            for logical in object_sizes:
                physical = (
                    (logical + _GDS_ALIGNMENT - 1) // _GDS_ALIGNMENT * _GDS_ALIGNMENT
                )
                pos = group
                segments = []
                while pos < group + physical:
                    start = pos // _MAX_GDS_REGION * _MAX_GDS_REGION
                    offset = pos - start
                    size = min(_MAX_GDS_REGION - offset, group + physical - pos)
                    segments.append((buffer.data_ptr() + slot + start, offset, size))
                    pos += size
                begin = slot + group
                plans[buffer.data_ptr() + begin] = _SlotIO(
                    logical,
                    physical,
                    buffer[begin + logical : begin + physical],
                    tuple(segments),
                )
                group += gds_group_stride(logical)
        return plans

    def _buffer_regions(
        self, buffer: torch.Tensor, slot_bytes: int | None
    ) -> Iterator[torch.Tensor]:
        """Use identical region boundaries for registration and deregistration."""
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        if not nbytes:
            return
        stride = slot_bytes or nbytes
        if stride <= 0 or nbytes % stride:
            raise ValueError("GDS slot stride must evenly partition the staging buffer")
        for slot_start in range(0, nbytes, stride):
            slot_end = slot_start + stride
            for start in range(slot_start, slot_end, _MAX_GDS_REGION):
                yield buf[start : min(start + _MAX_GDS_REGION, slot_end)]

    def _register_region_locked(self, buffer: torch.Tensor) -> None:
        """GDS-register one <=16 MiB region (caller holds the lock)."""
        nbytes = buffer.numel() * buffer.element_size()
        base = buffer.data_ptr()
        self.backend.register_buffer(buffer)
        idx = bisect.bisect_left(self._base_ptrs, base)
        self._buffers.insert(idx, buffer)
        self._base_ptrs.insert(idx, base)
        self._nbytes.insert(idx, nbytes)
        logger.info(
            "GDSContext: registered %d bytes at 0x%x via GDS (total registrations: %d)",
            nbytes,
            base,
            len(self._buffers),
        )

    def _deregister_main_stream_locked(self, raw: int) -> None:
        try:
            self.backend.deregister_stream(raw)
        except Exception:
            logger.exception("GDS main stream deregistration failed")
            if self._main_streams[raw] is not None:
                return  # Retain the native owner for slab teardown.
        del self._main_streams[raw]

    def _deregister_region_locked(self, buffer: torch.Tensor) -> None:
        """Deregister one region with the GDS library (caller holds the lock).

        Args:
            buffer: A staging-buffer slot previously registered.
        """
        base = buffer.data_ptr()
        idx = bisect.bisect_left(self._base_ptrs, base)
        if idx >= len(self._base_ptrs) or self._base_ptrs[idx] != base:
            raise ValueError(
                f"GDS buffer at 0x{base:x} is not registered by this context"
            )
        try:
            self.backend.deregister_buffer(self._buffers[idx])
        except Exception as e:
            logger.warning("GDSContext: deregister_buffer: %s", e)
        del self._buffers[idx]
        del self._base_ptrs[idx]
        del self._nbytes[idx]

    def _resolve_buffer(self, gpu_buffer: torch.Tensor) -> tuple[int, int, int]:
        """Locate the registered region ``gpu_buffer`` starts in.

        Returns ``(base_ptr, dev_offset, region_nbytes)``; ``region_nbytes -
        dev_offset`` is the room left in the region, which :meth:`transfer_async`
        uses to cut DMAs at region boundaries. Callers always pass a pointer
        inside a registered region.
        """
        ptr = gpu_buffer.data_ptr()
        # Held briefly so a concurrent deregister can't mutate the parallel
        # lists mid-lookup.
        with self._registry_lock:
            idx = bisect.bisect_right(self._base_ptrs, ptr) - 1
            if idx < 0:
                raise ValueError(f"GDS buffer at 0x{ptr:x} is not registered")
            base = self._base_ptrs[idx]
            nbytes = self._nbytes[idx]
        offset = ptr - base
        if offset >= nbytes:
            raise ValueError(f"GDS buffer at 0x{ptr:x} is not registered")
        return base, offset, nbytes

    def _slab_read(
        self,
        slab_offset: int,
        size: int,
        dev_offset: int,
        buf_base: int,
        stream_handle: int,
    ) -> Submission:
        """Submit one async GDS read against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_read: slab handle not open")
        return self._slab_handle.read_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )

    def _slab_write(
        self,
        slab_offset: int,
        size: int,
        dev_offset: int,
        buf_base: int,
        stream_handle: int,
    ) -> Submission:
        """Submit one async GDS write against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_write: slab handle not open")
        return self._slab_handle.write_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )

    def _record_submission(
        self,
        sub: "Submission",
        device: object,
        stream: object,
        raw_stream: int,
    ) -> None:
        """Track an in-flight submission so its ctypes storage outlives the DMA.

        Accumulated per stream; every ``_SUBMISSION_CHECKPOINT_EVERY`` ops a
        platform completion event is recorded and completed batches are released.
        """
        with self._submissions_lock:
            st = self._submissions.get(raw_stream)
            if st is None:
                st = self._submissions[raw_stream] = _StreamSubmissions()
            st.uncommitted.append(sub)
            # A slot owns one bounded object and explicitly drains its segments
            # before scatter or reuse. Other streams use periodic checkpoints.
            if raw_stream in self._slot_streams:
                return
            st.ops_since_checkpoint += 1
            if st.ops_since_checkpoint >= _SUBMISSION_CHECKPOINT_EVERY:
                self._checkpoint_submissions_locked(st, device, stream)

    def _checkpoint_submissions_locked(
        self, st: _StreamSubmissions, device: object, stream: object
    ) -> None:
        """Close ``st``'s current batch behind a platform event on ``stream`` and
        drop earlier batches whose event has completed. Hold
        ``self._submissions_lock``.
        """
        if st.uncommitted:
            event = platform_stream.record_completion_event(device, stream)
            st.inflight.append((event, st.uncommitted))
            st.uncommitted = []
        st.ops_since_checkpoint = 0
        st.inflight = [
            (event, subs) for (event, subs) in st.inflight if not event.is_complete()
        ]


@functools.cache
def get_gds_context() -> GDSContext:
    """Return the process-global :class:`GDSContext` singleton (created empty on
    first access). Consult :attr:`GDSContext.initialized` to tell whether GDS L1
    is active."""
    return GDSContext()


def initialize_gds_context(config: Optional[GdsL1Config]) -> GDSContext:
    """Set up the process-global :class:`GDSContext` (once, at startup).

    ``config=None`` leaves it uninitialized (GDS L1 disabled); otherwise the
    slab is created and registered. Returns the singleton.
    """
    context = get_gds_context()
    if config is not None:
        with _owner_contexts_lock:
            if _owner_contexts:
                raise ValueError("Only one GDS L1 slab is supported per process")
            context.initialize(config)
    return context
