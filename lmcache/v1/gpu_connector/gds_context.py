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
from dataclasses import dataclass, field
from typing import Optional
import bisect
import enum
import functools
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.platform import stream as platform_stream

logger = init_logger(__name__)

_GDS_ALIGNMENT = 4096
# A single GDS buffer registration / DMA is capped at 16 MiB (both cuFile and
# hipFile); larger buffers and chunks are registered and transferred in
# <=16 MiB regions.
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


def register_gds_gpu_buffer(buffer: torch.Tensor) -> None:
    """Register a GPU staging buffer with each live slab on its current stream.

    Args:
        buffer: Contiguous, aligned staging buffer, after L1 initialization.
    """
    with _owner_contexts_lock:
        contexts = [get_gds_context(), *_owner_contexts.values()]
    for context in contexts:
        context.register_gpu_buffer(buffer)


def deregister_gds_gpu_buffer(buffer: torch.Tensor) -> None:
    """Drain and unregister a staging buffer from every slab before freeing it.

    Args:
        buffer: Buffer previously passed to register_gds_gpu_buffer.
    """
    with _owner_contexts_lock:
        contexts = [get_gds_context(), *_owner_contexts.values()]
    for context in contexts:
        context.deregister_gpu_buffer(buffer)


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
        self._registered_streams: set[int] = set()  # maintained for close()

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

    def register_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Register a staging buffer (and its stream) with the GDS library.

        Registered as contiguous <=16 MiB regions (the GDS buffer-registration
        cap); :meth:`transfer_async` splits transfers at these boundaries.

        Args:
            buffer: Contiguous GPU staging buffer, 4 KiB-aligned in size.
        """
        if not self.initialized:
            return
        stream = platform_stream.current_stream(buffer.device)
        raw_stream = platform_stream.stream_handle(buffer.device, stream)
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        with self._registry_lock:
            stream_registered = False
            registered_regions: list[torch.Tensor] = []
            try:
                if raw_stream not in self._registered_streams:
                    self.backend.register_stream(raw_stream)
                    self._registered_streams.add(raw_stream)
                    stream_registered = True
                for start in range(0, nbytes, _MAX_GDS_REGION):
                    region = buf[start : min(start + _MAX_GDS_REGION, nbytes)]
                    self._register_region_locked(region)
                    registered_regions.append(region)
            except BaseException:
                for region in reversed(registered_regions):
                    try:
                        self._deregister_region_locked(region)
                    except Exception:
                        logger.exception(
                            "GDSContext: failed to roll back buffer registration"
                        )
                if stream_registered:
                    try:
                        self.backend.deregister_stream(raw_stream)
                    except Exception:
                        logger.exception(
                            "GDSContext: failed to roll back stream registration"
                        )
                    self._registered_streams.discard(raw_stream)
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
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        with self._registry_lock:
            for start in range(0, nbytes, _MAX_GDS_REGION):
                self._deregister_region_locked(
                    buf[start : min(start + _MAX_GDS_REGION, nbytes)]
                )
            if raw_stream in self._registered_streams:
                try:
                    self.backend.deregister_stream(raw_stream)
                except Exception as e:
                    logger.warning(
                        "GDSContext.deregister_gpu_buffer: deregister_stream: %s", e
                    )
                self._registered_streams.discard(raw_stream)
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
        """Sync the stream, deregister GDS state, and close the slab handle."""
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
            for raw_stream in list(self._registered_streams):
                try:
                    self.backend.deregister_stream(raw_stream)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_stream: %s", e)
            self._registered_streams.clear()
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
        self.initialized = False

    # --- Internal -----------------------------------------------------

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
