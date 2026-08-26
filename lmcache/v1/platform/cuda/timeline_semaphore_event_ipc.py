# SPDX-License-Identifier: Apache-2.0
"""GPU timeline-semaphore event IPC backend (zero host-side sharing).

CUDA interprocess *event* handles (used by ``DefaultEventIPCBackend``)
only resolve when both processes share a ``/dev/shm`` tmpfs -- on
Kubernetes that means ``hostIPC: true``. Legacy CUDA IPC *memory* handles
have no such dependency: they rendezvous inside the kernel driver and work
across fully isolated containers (only the PID *values* of the two
processes must differ, same constraint as KV-cache tensor IPC).

This backend implements event semantics on memory handles alone, using
timeline semaphores: monotonically increasing 64-bit values where
signaling writes a target value and waiting blocks until the value
reaches it (the same concept as Vulkan/Direct3D 12 timeline semaphores).
Each process lazily allocates a small buffer of semaphore slots and
exports it once with ``cudaIpcGetMemHandle``; an event is a
``(slot, sequence)`` event object against it:

- ``record_event`` -> ``cuStreamWriteValue64(slot, seq)`` on the stream
- ``wait_event``   -> ``cuStreamWaitValue64(slot, seq, GEQ)`` on the stream
- ``query_event``  -> 8-byte device read, compared against ``seq``
- ``export_event`` -> self-contained ``(mem handle, slot offset, seq)``
  bytes, so the importer needs no registration side channel

Semantics vs ``DefaultEventIPCBackend``: a never-recorded event is
complete; an exported handle is a snapshot of the sequence at export time
(LMCache exports once, after the single record); same-process import is
supported.

Slots are assigned per recording stream, so each slot's values are
stream-ordered and monotonic, keeping the GEQ wait race-free.

Only events from this backend's ``create_event`` can be exported; call
sites constructing ``torch_dev.Event(interprocess=True)`` directly must be
migrated before binding this backend to the device spec. See
``docs/design/v1/platform/cuda/timeline_semaphore_event_ipc.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import ctypes
import enum
import struct
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.cuda.utils import (
    _CHECK_CUDA,
    _cuda,
    _raw_stream_handle,
    _resolve_device_index,
    cudaStream_t,
)

logger = init_logger(__name__)

#: 8-byte semaphore slots per buffer; slot 0 is reserved for the
#: ``check_event_support`` probe.
_SLOT_COUNT = 4096
_SLOT_BYTES = 8
_PROBE_SLOT_OFFSET = 0
_FIRST_ALLOCATABLE_SLOT = 1

#: Wire format: version, 64-byte cudaIpcMemHandle, slot offset, sequence.
_HANDLE_VERSION = 1
_EXPORT_STRUCT = struct.Struct("!B64sQQ")

#: Sentinels for a local event that has not been recorded yet.
_UNASSIGNED_SLOT_OFFSET = -1
_UNRECORDED_SEQ = 0


class TimelineSemaphoreEventOrigin(enum.Enum):
    """Which side of the wire a :class:`TimelineSemaphoreEvent` was created on."""

    LOCAL = "local"
    IMPORTED = "imported"


@dataclass
class TimelineSemaphoreEvent:
    """A ``(slot, sequence)`` event object against a semaphore buffer.

    Created by :class:`TimelineSemaphoreEventIPCBackend`; callers treat it as
    opaque.

    Attributes:
        origin: LOCAL events are recordable; IMPORTED are wait/query only. Just
            a safeguard so that we don't make stupid mistakes and fail silently.
        device_index: CUDA device ordinal the event operates on.
        base_ptr: Semaphore buffer pointer in this process (0 until the
            first record assigns a slot).
        slot_offset: Byte offset of the slot (-1 until the first record).
        seq: Target sequence number; 0 means never recorded. Semaphore's value
            >= than a event's seq number means such event is complete.
        handle_bytes: The buffer's ``cudaIpcMemHandle`` bytes (empty until
            the first record).
    """

    origin: TimelineSemaphoreEventOrigin
    device_index: int
    base_ptr: int = 0
    slot_offset: int = _UNASSIGNED_SLOT_OFFSET
    seq: int = _UNRECORDED_SEQ
    handle_bytes: bytes = b""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (``IPCEvent`` duck method).

        Args:
            stream: Stream that should wait; ``None`` for the current one.
        """
        _enqueue_wait(self, stream)


@dataclass
class _TimelineSemaphoreBuffer:
    """Per-device semaphore buffer owned by one backend instance.

    Attributes:
        device_index: CUDA device ordinal the buffer lives on.
        base_ptr: Device pointer of the buffer.
        handle_bytes: ``cudaIpcMemHandle`` bytes exported at allocation.
        buffer_ops_stream: Dedicated non-blocking stream for host-initiated
            slot reads/writes, so they never synchronize with (possibly
            blocked) caller streams.
        lock: Guards the mutable fields below; ``record_event`` holds it
            through the write enqueue (see there).
        slot_by_stream: Raw ``cudaStream_t`` -> slot index.
        next_seq_by_slot: Slot index -> last assigned sequence.
        next_free_slot: Next never-used slot. Slots are not reclaimed.
    """

    device_index: int
    base_ptr: int
    handle_bytes: bytes
    buffer_ops_stream: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    slot_by_stream: dict[cudaStream_t, int] = field(default_factory=dict)
    next_seq_by_slot: dict[int, int] = field(default_factory=dict)
    next_free_slot: int = _FIRST_ALLOCATABLE_SLOT

    def assign_locked(self, raw_stream: cudaStream_t) -> tuple[int, int]:
        """Assign the next sequence for ``raw_stream``'s slot.

        Caller must hold :attr:`lock` through the semaphore write enqueue.

        Args:
            raw_stream: Raw ``cudaStream_t`` of the recording stream.

        Returns:
            A ``(slot_offset, seq)`` pair for this record.

        Raises:
            RuntimeError: If all slots are in use.
        """
        slot: int | None = self.slot_by_stream.get(raw_stream)
        if slot is None:
            if self.next_free_slot >= _SLOT_COUNT:
                raise RuntimeError(
                    f"Semaphore buffer on cuda:{self.device_index} is "
                    f"out of slots ({_SLOT_COUNT}); too many distinct "
                    "recording streams."
                )
            slot = self.next_free_slot
            self.next_free_slot += 1
            self.slot_by_stream[raw_stream] = slot
        seq: int = self.next_seq_by_slot.get(slot, _UNRECORDED_SEQ) + 1
        self.next_seq_by_slot[slot] = seq
        return slot * _SLOT_BYTES, seq


# Semaphore-buffer base pointers created by this process, by handle bytes:
# import_event resolves same-process handles locally, since
# cudaIpcOpenMemHandle cannot open a handle in the exporting process.
_LOCAL_BUFFERS_BY_HANDLE: dict[bytes, int] = {}
# Imported buffers by (handle bytes, importer device index); mappings live
# for the process lifetime (32 KiB each).
_IMPORTED_BUFFERS_BY_HANDLE: dict[tuple[bytes, int], int] = {}
_HANDLE_REGISTRY_LOCK = threading.Lock()


def _enqueue_wait(event: TimelineSemaphoreEvent, stream: object | None) -> None:
    """Enqueue a ``cuStreamWaitValue64`` for ``event`` on ``stream``.

    Never-recorded events are complete; nothing is enqueued.

    Args:
        event: The event to wait for.
        stream: Stream that should wait; ``None`` for the current stream.
    """
    if event.seq == _UNRECORDED_SEQ:
        return
    raw: cudaStream_t = _raw_stream_handle(stream, event.device_index)
    with torch.cuda.device(event.device_index):
        result: tuple[object, ...] = _cuda.driver.cuStreamWaitValue64(
            _cuda.driver.CUstream(raw),
            _cuda.driver.CUdeviceptr(event.base_ptr + event.slot_offset),
            event.seq,
            _cuda.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
        )
    _CHECK_CUDA(result, "cuStreamWaitValue64")


class TimelineSemaphoreEventIPCBackend:
    """CUDA event-IPC backend over semaphore slots in IPC-shared memory.

    Implements the
    :class:`~lmcache.v1.platform.base.event_ipc.EventIPCBackend` protocol
    with the same observable semantics as ``DefaultEventIPCBackend`` but
    without CUDA interprocess event handles (which need a shared
    ``/dev/shm``); see the module docstring.

    Thread safety: backend methods may be called from multiple threads;
    backend and buffer state is locked internally. A single event object,
    however, must not be recorded concurrently with export/wait/query on
    that same event (record mutates its fields) -- the same discipline
    CUDA events require between ``record`` and ``ipc_handle``.
    """

    def __init__(self) -> None:
        self.device_type: str = "cuda"
        self._lock = threading.Lock()
        self._buffers: dict[int, _TimelineSemaphoreBuffer] = {}
        self._probed_devices: set[int] = set()
        # Per-thread dedicated streams for host-blocking waits, by device
        # index. synchronize_event enqueues a semaphore wait here and blocks
        # in cudaStreamSynchronize; a shared stream would serialize
        # concurrent synchronizes behind each other's pending waits, and the
        # buffer's host-ops stream must stay wait-free for query_event reads.
        self._sync_streams = threading.local()

    def check_event_support(self, device: object) -> None:
        """Validate timeline-semaphore event IPC on ``device`` with a live probe."""
        device_index: int = _resolve_device_index(device)
        with self._lock:
            if device_index in self._probed_devices:
                return

        # Allocates the buffer and round-trips a write/wait pair through the
        # reserved probe slot. Probes with real calls on purpose: the
        # ``CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`` attribute reports 0
        # on drivers where the v2 memops (default since CUDA 12) work.
        try:
            buffer: _TimelineSemaphoreBuffer = self._get_buffer(device_index)
            probe_ptr: int = buffer.base_ptr + _PROBE_SLOT_OFFSET
            stream_hdl: cudaStream_t = _raw_stream_handle(
                buffer.buffer_ops_stream, device_index
            )
            with torch.cuda.device(device_index):
                _CHECK_CUDA(
                    _cuda.driver.cuStreamWriteValue64(
                        _cuda.driver.CUstream(stream_hdl),
                        _cuda.driver.CUdeviceptr(probe_ptr),
                        1,
                        _cuda.driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                    ),
                    "cuStreamWriteValue64 (probe)",
                )
                _CHECK_CUDA(
                    _cuda.driver.cuStreamWaitValue64(
                        _cuda.driver.CUstream(stream_hdl),
                        _cuda.driver.CUdeviceptr(probe_ptr),
                        1,
                        _cuda.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
                    ),
                    "cuStreamWaitValue64 (probe)",
                )
                _CHECK_CUDA(
                    _cuda.runtime.cudaStreamSynchronize(buffer.buffer_ops_stream),
                    "cudaStreamSynchronize (probe)",
                )
        except Exception as e:
            raise RuntimeError(
                f"Device backend '{self.device_type}' does not support "
                f"timeline-semaphore event IPC on device {device_index}: {e}"
            ) from e
        with self._lock:
            self._probed_devices.add(device_index)

    def create_event(self, device: object) -> object:
        """Create a new timeline-semaphore event on ``device``."""
        # Slot/flag binding is deferred to the recording time when stream is known.
        return TimelineSemaphoreEvent(
            origin=TimelineSemaphoreEventOrigin.LOCAL,
            device_index=_resolve_device_index(device),
        )

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream``.

        Assigns the stream's slot, bumps its sequence, and enqueues the
        semaphore write so the slot reaches that value exactly when all
        prior work on ``stream`` has completed.

        Args:
            event: A local :class:`TimelineSemaphoreEvent`.
            stream: Recording stream; ``None`` for the current stream.

        Raises:
            RuntimeError: If ``event`` was imported (imported events are
                wait/query only) or the enqueue fails.
        """
        if not isinstance(event, TimelineSemaphoreEvent):
            raise RuntimeError(
                f"record_event expected a TimelineSemaphoreEvent, got {type(event)!r}"
            )
        if event.origin is not TimelineSemaphoreEventOrigin.LOCAL:
            raise RuntimeError(
                "Imported timeline-semaphore events cannot be recorded; only the "
                "exporting process records."
            )
        buffer: _TimelineSemaphoreBuffer = self._get_buffer(event.device_index)
        stream_hdl: cudaStream_t = _raw_stream_handle(stream, event.device_index)
        # Lock held through the enqueue: a second thread landing seq N+1's
        # write before this thread's seq N would leave the slot at N and
        # strand the N+1 event's waiters.
        with buffer.lock:
            slot_offset: int
            seq: int
            slot_offset, seq = buffer.assign_locked(stream_hdl)
            event.base_ptr = buffer.base_ptr
            event.handle_bytes = buffer.handle_bytes
            event.slot_offset = slot_offset
            event.seq = seq
            with torch.cuda.device(event.device_index):
                _CHECK_CUDA(
                    _cuda.driver.cuStreamWriteValue64(
                        _cuda.driver.CUstream(stream_hdl),
                        _cuda.driver.CUdeviceptr(event.base_ptr + event.slot_offset),
                        event.seq,
                        _cuda.driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                    ),
                    "cuStreamWriteValue64 (record)",
                )

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize ``event`` for import by another process.

        Args:
            event: A :class:`TimelineSemaphoreEvent` to export.
            device: Device that owns the event (unused; part of the
                protocol signature).

        Returns:
            The packed handle bytes (snapshot of the current sequence).

        Raises:
            RuntimeError: If ``event`` is not a :class:`TimelineSemaphoreEvent`.
        """
        if not isinstance(event, TimelineSemaphoreEvent):
            raise RuntimeError(
                f"export_event expected a TimelineSemaphoreEvent, got {type(event)!r}"
            )
        handle_bytes: bytes = event.handle_bytes
        slot_offset: int = event.slot_offset
        if event.seq == _UNRECORDED_SEQ:
            # Exports as "already complete"; the wire format still needs a
            # valid handle and offset.
            handle_bytes = self._get_buffer(event.device_index).handle_bytes
            slot_offset = _PROBE_SLOT_OFFSET
        return _EXPORT_STRUCT.pack(
            _HANDLE_VERSION, handle_bytes, slot_offset, event.seq
        )

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a serialized timeline-semaphore event handle on ``device``.

        Opens the exporter's semaphore buffer on first use and caches the
        mapping for the process lifetime; same-process handles resolve to
        the local buffer.

        Args:
            handle: Bytes produced by :meth:`export_event`.
            device: Device on which to import the event.

        Returns:
            An imported :class:`TimelineSemaphoreEvent` (wait/query only).

        Raises:
            RuntimeError: If the payload is malformed, carries an
                out-of-range slot offset, or the semaphore buffer cannot be
                opened.
        """
        version: int
        handle_bytes: bytes
        slot_offset: int
        seq: int
        try:
            version, handle_bytes, slot_offset, seq = _EXPORT_STRUCT.unpack(handle)
        except struct.error as e:
            raise RuntimeError(
                f"Malformed timeline-semaphore event handle ({len(handle)} bytes): {e}"
            ) from e
        if version != _HANDLE_VERSION:
            raise RuntimeError(
                f"Unsupported timeline-semaphore event handle version {version}; "
                f"this process supports version {_HANDLE_VERSION}."
            )
        if (
            slot_offset < 0
            or slot_offset >= _SLOT_COUNT * _SLOT_BYTES
            or slot_offset % _SLOT_BYTES != 0
        ):
            raise RuntimeError(
                f"Timeline-semaphore event handle carries an invalid slot offset "
                f"{slot_offset}; expected an {_SLOT_BYTES}-byte-aligned "
                f"offset below {_SLOT_COUNT * _SLOT_BYTES}."
            )
        device_index = _resolve_device_index(device)

        # Lock held across the open so two threads importing the same new
        # handle cannot race it (opening is rare; imports hit the cache).
        base_ptr: int
        with _HANDLE_REGISTRY_LOCK:
            local_base: int | None = _LOCAL_BUFFERS_BY_HANDLE.get(handle_bytes)
            if local_base is not None:
                base_ptr = local_base
            else:
                cached: int | None = _IMPORTED_BUFFERS_BY_HANDLE.get(
                    (handle_bytes, device_index)
                )
                if cached is not None:
                    base_ptr = cached
                else:
                    ipc_handle = _cuda.runtime.cudaIpcMemHandle_t()
                    ipc_handle.reserved = handle_bytes
                    with torch.cuda.device(device_index):
                        open_result = _cuda.runtime.cudaIpcOpenMemHandle(
                            ipc_handle,
                            _cuda.runtime.cudaIpcMemLazyEnablePeerAccess,
                        )
                        _CHECK_CUDA(
                            open_result, "cudaIpcOpenMemHandle (semaphore buffer)"
                        )
                        _err, opened_ptr = open_result
                        base_ptr = int(opened_ptr)
                    _IMPORTED_BUFFERS_BY_HANDLE[(handle_bytes, device_index)] = base_ptr

        return TimelineSemaphoreEvent(
            origin=TimelineSemaphoreEventOrigin.IMPORTED,
            device_index=device_index,
            base_ptr=base_ptr,
            slot_offset=slot_offset,
            seq=seq,
            handle_bytes=handle_bytes,
        )

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event``.

        Args:
            event: A local or imported :class:`TimelineSemaphoreEvent`.
            stream: Stream that should wait; ``None`` for the current one.

        Raises:
            RuntimeError: If ``event`` is not a :class:`TimelineSemaphoreEvent` or
                the enqueue fails.
        """
        if not isinstance(event, TimelineSemaphoreEvent):
            raise RuntimeError(
                f"wait_event expected a TimelineSemaphoreEvent, got {type(event)!r}"
            )
        _enqueue_wait(event, stream)

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed.

        Args:
            event: A local or imported :class:`TimelineSemaphoreEvent`.

        Returns:
            ``True`` when the slot has reached the event's sequence (or
            the event was never recorded); otherwise ``False``.

        Raises:
            RuntimeError: If ``event`` is not a :class:`TimelineSemaphoreEvent` or
                the device read fails.
        """
        if not isinstance(event, TimelineSemaphoreEvent):
            raise RuntimeError(
                f"query_event expected a TimelineSemaphoreEvent, got {type(event)!r}"
            )
        if event.seq == _UNRECORDED_SEQ:
            return True
        return self._read_slot(event) >= event.seq

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until ``event`` completes.

        Enqueues a semaphore wait on this thread's dedicated sync stream
        and blocks in ``cudaStreamSynchronize``, so the calling thread is
        parked in the driver and woken like a CUDA event synchronize (no
        host-side polling). Blocks indefinitely while the record point is
        unreached, like a CUDA event synchronize.

        Args:
            event: A local or imported :class:`TimelineSemaphoreEvent`.
            device: Device that owns the event (unused; part of the
                protocol signature).

        Raises:
            RuntimeError: If ``event`` is not a :class:`TimelineSemaphoreEvent` or
                a driver call fails.
        """
        if not isinstance(event, TimelineSemaphoreEvent):
            raise RuntimeError(
                f"synchronize_event expected a TimelineSemaphoreEvent, "
                f"got {type(event)!r}"
            )
        if event.seq == _UNRECORDED_SEQ:
            return
        if self._read_slot(event) >= event.seq:
            return  # already complete; skip the stream round trip
        sync_stream: cudaStream_t = self._get_sync_stream(event.device_index)
        _enqueue_wait(event, sync_stream)
        with torch.cuda.device(event.device_index):
            _CHECK_CUDA(
                _cuda.runtime.cudaStreamSynchronize(sync_stream),
                "cudaStreamSynchronize (synchronize_event)",
            )

    def _get_sync_stream(self, device_index: int) -> cudaStream_t:
        """Return this thread's dedicated sync stream for ``device_index``,
        creating it on first use.

        The stream lives for the process lifetime (like the semaphore
        buffers); one per (thread, device) that calls
        :meth:`synchronize_event`.

        Args:
            device_index: CUDA device ordinal.

        Returns:
            The raw ``cudaStream_t`` handle.

        Raises:
            RuntimeError: If stream creation fails.
        """
        streams: dict[int, cudaStream_t] | None = getattr(
            self._sync_streams, "by_device", None
        )
        if streams is None:
            streams = {}
            self._sync_streams.by_device = streams
        stream = streams.get(device_index)
        if stream is None:
            with torch.cuda.device(device_index):
                result = _cuda.runtime.cudaStreamCreateWithFlags(
                    _cuda.runtime.cudaStreamNonBlocking
                )
            _CHECK_CUDA(result, "cudaStreamCreateWithFlags (sync stream)")
            _err, stream = result
            streams[device_index] = stream
        return stream

    def _get_buffer(self, device_index: int) -> _TimelineSemaphoreBuffer:
        """Return the semaphore buffer for ``device_index``, allocating it
        on first use.

        Args:
            device_index: CUDA device ordinal.

        Returns:
            The per-device :class:`_TimelineSemaphoreBuffer` of this backend.

        Raises:
            RuntimeError: If allocation or IPC export fails.
        """
        # Lock-free fast path: query/synchronize poll this on every
        # iteration, and buffers are never removed.
        if (buffer := self._buffers.get(device_index)) is not None:
            return buffer
        with self._lock:
            if (buffer := self._buffers.get(device_index)) is not None:
                return buffer
            nbytes: int = _SLOT_COUNT * _SLOT_BYTES
            with torch.cuda.device(device_index):
                malloc_result = _cuda.runtime.cudaMalloc(nbytes)
                _CHECK_CUDA(malloc_result, "cudaMalloc")
                _err, base = malloc_result
                buffer_stream = None
                try:
                    stream_result = _cuda.runtime.cudaStreamCreateWithFlags(
                        _cuda.runtime.cudaStreamNonBlocking
                    )
                    _CHECK_CUDA(stream_result, "cudaStreamCreateWithFlags")
                    _err, buffer_stream = stream_result
                    _CHECK_CUDA(
                        _cuda.runtime.cudaMemsetAsync(base, 0, nbytes, buffer_stream),
                        "cudaMemsetAsync",
                    )
                    _CHECK_CUDA(
                        _cuda.runtime.cudaStreamSynchronize(buffer_stream),
                        "cudaStreamSynchronize (semaphore buffer init)",
                    )
                    handle_result = _cuda.runtime.cudaIpcGetMemHandle(base)
                    _CHECK_CUDA(handle_result, "cudaIpcGetMemHandle")
                    _err, handle = handle_result
                except Exception:
                    if buffer_stream is not None:
                        _cuda.runtime.cudaStreamDestroy(buffer_stream)
                    _cuda.runtime.cudaFree(base)
                    raise
            buffer = _TimelineSemaphoreBuffer(
                device_index=device_index,
                base_ptr=int(base),
                handle_bytes=bytes(handle.reserved),
                buffer_ops_stream=buffer_stream,
            )
            with _HANDLE_REGISTRY_LOCK:
                _LOCAL_BUFFERS_BY_HANDLE[buffer.handle_bytes] = buffer.base_ptr
            self._buffers[device_index] = buffer
            logger.info(
                "Allocated timeline-semaphore event buffer on cuda:%d (%d slots)",
                device_index,
                _SLOT_COUNT,
            )
            return buffer

    def _read_slot(self, event: TimelineSemaphoreEvent) -> int:
        """Read the current 64-bit value of ``event``'s slot.

        Args:
            event: A recorded or imported event.

        Returns:
            The slot's current value.

        Raises:
            RuntimeError: If the device read fails.
        """
        buffer_stream: object = self._get_buffer(event.device_index).buffer_ops_stream
        value: ctypes.c_uint64 = ctypes.c_uint64(0)
        with torch.cuda.device(event.device_index):
            _CHECK_CUDA(
                _cuda.runtime.cudaMemcpyAsync(
                    ctypes.addressof(value),
                    event.base_ptr + event.slot_offset,
                    _SLOT_BYTES,
                    _cuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    buffer_stream,
                ),
                "cudaMemcpyAsync (semaphore read)",
            )
            _CHECK_CUDA(
                _cuda.runtime.cudaStreamSynchronize(buffer_stream),
                "cudaStreamSynchronize (semaphore read)",
            )
        return value.value
