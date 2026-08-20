# SPDX-License-Identifier: Apache-2.0
"""Tests for the GPU timeline-semaphore event IPC backend.

Parity cases are parametrized over both backends; timeline-semaphore-only cases
cover what CUDA events cannot do (same-process import).
"""

# Standard
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.connection import Connection
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    EventIPCBackend,
)
from lmcache.v1.platform.cuda.timeline_semaphore_event_ipc import (
    TimelineSemaphoreEventIPCBackend,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
    # ROCm reports torch.cuda.is_available() but has no cuda.bindings; the
    # backend correctly fails check_event_support there (fails closed).
    pytest.mark.skipif(
        torch.version.hip is not None,
        reason="timeline-semaphore event IPC is NVIDIA-only (cuda.bindings memops)",
    ),
]

DEVICE = "cuda:0"


def _make_backend(kind: str) -> EventIPCBackend:
    """Construct a backend by parametrization key."""
    if kind == "timeline_semaphore":
        return TimelineSemaphoreEventIPCBackend()
    return DefaultEventIPCBackend()


def _gate_stream(stream: torch.cuda.Stream) -> Callable[[], None]:
    """Block ``stream`` behind a host-released flag; return the release.

    Enqueues a ``cuStreamWaitValue64`` on ``stream``, so it provably has
    pending work until the returned callable writes the flag with
    ``cuStreamWriteValue64`` on a separate stream (a pending memop wait is
    not woken by plain memory writes).

    Busy kernels cannot provide the pending work: the k3 CI runs with
    ``CUDA_LAUNCH_BLOCKING=1``, under which every kernel completes inside
    its launch call, so a stream is never observably busy behind kernels.
    Memops are unaffected by that mode -- but for the same reason, gated
    test bodies must not launch kernels (the launch would block the test
    thread behind the pending gate and deadlock the same-thread release);
    they stick to memop-based backend calls.

    Callers must allocate the semaphore buffer
    (``backend.check_event_support(device)``) and any tensors before
    gating: an allocation's implicit ``cudaMalloc`` device-sync would
    deadlock behind the gate. Always call the release (``try/finally``):
    a stream left gated hangs whatever touches it next.
    """
    # Third Party
    from cuda.bindings import driver

    flag = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    # The allocator may hand back a block still holding a released gate's 1,
    # and the zeroing memset is asynchronous -- make sure it lands before
    # the wait below samples the flag, or the gate never arms.
    torch.cuda.current_stream().synchronize()
    release_stream = torch.cuda.Stream()

    (err,) = driver.cuStreamWaitValue64(
        driver.CUstream(stream.cuda_stream),
        driver.CUdeviceptr(flag.data_ptr()),
        1,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamWaitValue64 (gate) failed: {err}")

    def release() -> None:
        # `flag` and `release_stream` are kept alive by this closure.
        (werr,) = driver.cuStreamWriteValue64(
            driver.CUstream(release_stream.cuda_stream),
            driver.CUdeviceptr(flag.data_ptr()),
            1,
            driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        if werr != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamWriteValue64 (gate release) failed: {werr}")
        release_stream.synchronize()

    return release


def test_timeline_semaphore_backend_satisfies_protocol() -> None:
    assert isinstance(TimelineSemaphoreEventIPCBackend(), EventIPCBackend)


def test_check_event_support_passes_on_cuda() -> None:
    backend = TimelineSemaphoreEventIPCBackend()
    backend.check_event_support(torch.device(DEVICE))  # must not raise
    # Second call exercises the probed-device cache.
    backend.check_event_support(torch.device(DEVICE))


@pytest.mark.parametrize("kind", ["default", "timeline_semaphore"])
def test_unrecorded_event_is_complete(kind: str) -> None:
    """A never-recorded event queries True and never blocks (both backends)."""
    backend = _make_backend(kind)
    device = torch.device(DEVICE)
    event = backend.create_event(device)
    assert backend.query_event(event) is True
    backend.synchronize_event(event, device)  # must return immediately
    stream = torch.cuda.Stream()
    backend.wait_event(event, stream)  # must be a no-op enqueue
    stream.synchronize()


def test_unrecorded_event_exports_as_complete() -> None:
    """Exporting a never-recorded event yields an importable handle that is
    already complete (probe-slot fallback). CUDA event handles cannot be
    self-imported, so this is timeline-semaphore-only coverage.
    """
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    event = backend.create_event(device)
    imported = backend.import_event(backend.export_event(event, device), device)
    assert backend.query_event(imported) is True
    backend.synchronize_event(imported, device)  # must return immediately
    stream = torch.cuda.Stream()
    backend.wait_event(imported, stream)  # must be a no-op enqueue
    stream.synchronize()


def test_query_transitions_with_recording_stream() -> None:
    """query_event flips False -> True when the recording stream drains."""
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    backend.check_event_support(device)  # allocate before gating
    stream = torch.cuda.Stream()
    release = _gate_stream(stream)
    try:
        event = backend.create_event(device)
        backend.record_event(event, stream)
        assert backend.query_event(event) is False
    finally:
        release()

    stream.synchronize()
    assert backend.query_event(event) is True


def test_same_process_export_import_orders_consumer_stream() -> None:
    """Imported event gates a consumer stream behind the producer's record
    point (timeline-semaphore-only: CUDA event handles cannot be
    self-imported). Consumer progress is observed via a second event
    recorded behind the imported wait, since gated test bodies must not
    launch kernels (see ``_gate_stream``).
    """
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    backend.check_event_support(device)  # allocate before gating
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    release = _gate_stream(producer)
    try:
        event = backend.create_event(device)
        backend.record_event(event, producer)

        imported = backend.import_event(backend.export_event(event, device), device)
        assert backend.query_event(imported) is False

        backend.wait_event(imported, consumer)
        tail = backend.create_event(device)
        backend.record_event(tail, consumer)  # sits behind the imported wait
        assert backend.query_event(tail) is False  # consumer genuinely gated
    finally:
        release()
    consumer.synchronize()

    assert backend.query_event(tail) is True  # consumer ran after the release
    assert backend.query_event(imported) is True
    assert producer.query() is True  # consumer only finished after producer


def test_reexport_after_rerecord_uses_higher_sequence() -> None:
    """Re-recording produces a later completion point; both imports drain."""
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    stream = torch.cuda.Stream()

    event = backend.create_event(device)
    backend.record_event(event, stream)
    first = backend.export_event(event, device)
    backend.record_event(event, stream)
    second = backend.export_event(event, device)
    assert first != second

    stream.synchronize()
    assert backend.query_event(backend.import_event(first, device)) is True
    assert backend.query_event(backend.import_event(second, device)) is True


def test_events_on_different_streams_are_independent() -> None:
    """A busy stream's pending event must not delay another stream's event."""
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    backend.check_event_support(device)  # allocate before gating
    busy = torch.cuda.Stream()
    idle = torch.cuda.Stream()

    release = _gate_stream(busy)
    try:
        busy_event = backend.create_event(device)
        backend.record_event(busy_event, busy)

        idle_event = backend.create_event(device)
        backend.record_event(idle_event, idle)
        backend.synchronize_event(idle_event, device)  # must not block on `busy`

        assert backend.query_event(busy_event) is False
    finally:
        release()
    busy.synchronize()
    assert backend.query_event(busy_event) is True


def test_event_object_satisfies_ipc_event_duck_protocol() -> None:
    """Events expose wait(stream) so they fit IPCEvent call sites."""
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    stream = torch.cuda.Stream()

    event = backend.create_event(device)
    backend.record_event(event, stream)
    waiter = torch.cuda.Stream()
    event.wait(waiter)  # type: ignore[attr-defined]
    waiter.synchronize()


def test_exports_from_one_buffer_share_identical_handle_bytes() -> None:
    """Every export from one device's buffer carries byte-identical mem-handle
    bytes (packed at bytes 1..65 of the payload, after the version byte).
    The importer's mapping cache is keyed by these bytes, so two events from
    the same producer must resolve to one cached mapping, not two.
    """
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)

    first_event = backend.create_event(device)
    backend.record_event(first_event, torch.cuda.Stream())
    second_event = backend.create_event(device)
    backend.record_event(second_event, torch.cuda.Stream())

    first = backend.export_event(first_event, device)
    second = backend.export_event(second_event, device)
    assert first[1:65] == second[1:65]


def test_ipc_mem_handle_stable_and_unique_across_live_allocations() -> None:
    """Driver-behavior canary for the assumptions behind handle caching:
    ``cudaIpcGetMemHandle`` bytes are stable across repeated calls on one
    live allocation (including from interior pointers) and unique across
    concurrently live allocations -- the property the importer's
    ``(handle bytes, device)`` mapping cache relies on.

    Deliberately NOT asserted: handle bytes after free + realloc at the
    same address. The driver specifies nothing there, and we have observed
    both a differing and a byte-identical handle for the reused address.
    The identical case means a stale handle can silently alias a later
    allocation, which is why the semaphore buffers are never freed
    (process lifetime, see the design doc's constraints).
    """
    # Third Party
    from cuda.bindings import runtime

    def alloc(nbytes: int) -> int:
        err, base = runtime.cudaMalloc(nbytes)
        if int(err) != 0:
            raise RuntimeError(f"cudaMalloc failed: {err}")
        return int(base)

    def free(base: int) -> None:
        (err,) = runtime.cudaFree(base)
        if int(err) != 0:
            raise RuntimeError(f"cudaFree failed: {err}")

    def handle_bytes_of(ptr: int) -> bytes:
        err, handle = runtime.cudaIpcGetMemHandle(ptr)
        if int(err) != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed: {err}")
        return bytes(handle.reserved)

    torch.cuda.init()
    nbytes = 32768
    first_base = alloc(nbytes)
    first = handle_bytes_of(first_base)

    assert handle_bytes_of(first_base) == first  # stable across repeated calls
    assert handle_bytes_of(first_base + 4096) == first  # interior ptr, same alloc

    second_base = alloc(nbytes)
    assert handle_bytes_of(second_base) != first  # unique across live allocs

    free(first_base)
    free(second_base)


def test_import_rejects_malformed_handles() -> None:
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    with pytest.raises(RuntimeError, match="Malformed timeline-semaphore event handle"):
        backend.import_event(b"junk", device)


def test_import_rejects_unsupported_version_and_bad_offset() -> None:
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    stream = torch.cuda.Stream()
    event = backend.create_event(device)
    backend.record_event(event, stream)
    exported = backend.export_event(event, device)

    wrong_version = bytes([99]) + exported[1:]
    with pytest.raises(RuntimeError, match="version 99"):
        backend.import_event(wrong_version, device)

    # Slot offset lives in bytes 65..72 (big-endian) of the payload.
    huge_offset = exported[:65] + (1 << 20).to_bytes(8, "big") + exported[73:]
    with pytest.raises(RuntimeError, match="invalid slot offset"):
        backend.import_event(huge_offset, device)


def test_stale_slot_value_does_not_satisfy_next_sequence() -> None:
    """A drained earlier event's residue in the shared slot must not
    complete a later event (guards against a GEQ off-by-one).
    """
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    stream = torch.cuda.Stream()

    first = backend.create_event(device)
    backend.record_event(first, stream)
    stream.synchronize()  # slot now holds first's sequence

    release = _gate_stream(stream)
    try:
        second = backend.create_event(device)
        backend.record_event(second, stream)
        imported = backend.import_event(backend.export_event(second, device), device)
        assert backend.query_event(imported) is False
    finally:
        release()

    stream.synchronize()
    assert backend.query_event(imported) is True


def test_record_rejects_imported_event() -> None:
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    stream = torch.cuda.Stream()
    event = backend.create_event(device)
    backend.record_event(event, stream)
    imported = backend.import_event(backend.export_event(event, device), device)
    with pytest.raises(RuntimeError, match="cannot be recorded"):
        backend.record_event(imported, stream)


def _cross_process_consumer(kind: str, conn: Connection) -> None:
    """Child: import the received handle, report the immediate query result,
    then synchronize on the event and report (blocked duration, query after).
    """
    torch.cuda.init()
    torch.cuda.set_device(0)
    backend = _make_backend(kind)
    device = torch.device(DEVICE)
    backend.check_event_support(device)  # allocate before the parent gates
    conn.send("ready")

    handle = conn.recv()
    event = backend.import_event(handle, device)
    conn.send(backend.query_event(event))
    start = time.monotonic()
    backend.synchronize_event(event, device)
    blocked_for = time.monotonic() - start
    conn.send((blocked_for, backend.query_event(event)))


@pytest.mark.parametrize("kind", ["default", "timeline_semaphore"])
def test_cross_process_synchronize_blocks_until_record_point(
    kind: str,
) -> None:
    """A separate process blocks on the imported event until the producer's
    stream reaches the record point (same host, so the default backend
    works and serves as the parity reference).
    """
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    child = ctx.Process(target=_cross_process_consumer, args=(kind, child_conn))
    child.start()
    try:
        assert parent_conn.recv() == "ready"

        backend = _make_backend(kind)
        device = torch.device(DEVICE)
        backend.check_event_support(device)  # allocate before gating
        stream = torch.cuda.Stream()
        release = _gate_stream(stream)
        try:
            event = backend.create_event(device)
            backend.record_event(event, stream)
            parent_conn.send(backend.export_event(event, device))

            assert parent_conn.recv() is False  # incomplete at import
            time.sleep(0.3)  # give the child time to enter synchronize_event
        finally:
            release()
        stream.synchronize()

        blocked_for, query_after = parent_conn.recv()
        assert query_after is True
        # The child must have been genuinely blocked on the pending work (a
        # never-completing wait would hang and trip the join timeout below).
        assert blocked_for >= 0.05
    finally:
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0
