# SPDX-License-Identifier: Apache-2.0
"""Tests for the GPU timeline-semaphore event IPC backend.

Parity cases are parametrized over both backends; timeline-semaphore-only cases
cover what CUDA events cannot do (same-process import).
"""

# Standard
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
]

DEVICE = "cuda:0"

#: Matmul iterations that keep an H100/H200-class GPU busy for hundreds of
#: milliseconds; slower GPUs only get slower, which these tests tolerate.
_BUSY_ITERS_SHORT = 200
_BUSY_ITERS_LONG = 600


def _make_backend(kind: str) -> EventIPCBackend:
    """Construct a backend by parametrization key."""
    if kind == "timeline_semaphore":
        return TimelineSemaphoreEventIPCBackend()
    return DefaultEventIPCBackend()


def _enqueue_busy_work(stream: torch.cuda.Stream, iters: int) -> torch.Tensor:
    """Enqueue ``iters`` chained matmuls on ``stream``.

    Tensors are allocated inside the stream context (no cross-stream
    allocator hazards). Keep the returned tensor alive until sync.
    """
    with torch.cuda.stream(stream):
        a = torch.randn(4096, 4096, device=DEVICE)
        b = torch.randn(4096, 4096, device=DEVICE)
        for _ in range(iters):
            a = a @ b
            a = a / a.norm()
    return a


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
    stream = torch.cuda.Stream()
    keepalive = _enqueue_busy_work(stream, _BUSY_ITERS_SHORT)

    event = backend.create_event(device)
    backend.record_event(event, stream)
    assert backend.query_event(event) is False

    stream.synchronize()
    assert backend.query_event(event) is True
    del keepalive


def test_same_process_export_import_orders_consumer_stream() -> None:
    """Imported event gates a consumer stream behind the producer's work
    (timeline-semaphore-only: CUDA event handles cannot be self-imported).
    """
    backend = TimelineSemaphoreEventIPCBackend()
    device = torch.device(DEVICE)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    payload = torch.zeros(1, device=DEVICE, dtype=torch.int64)
    keepalive = _enqueue_busy_work(producer, _BUSY_ITERS_SHORT)
    with torch.cuda.stream(producer):
        payload.fill_(42)
    event = backend.create_event(device)
    backend.record_event(event, producer)

    imported = backend.import_event(backend.export_event(event, device), device)
    assert backend.query_event(imported) is False

    with torch.cuda.stream(consumer):
        backend.wait_event(imported, consumer)
        observed = payload.clone()
    consumer.synchronize()

    assert observed.item() == 42
    assert backend.query_event(imported) is True
    assert producer.query() is True  # consumer only finished after producer
    del keepalive


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
    busy = torch.cuda.Stream()
    idle = torch.cuda.Stream()

    keepalive = _enqueue_busy_work(busy, _BUSY_ITERS_SHORT)
    busy_event = backend.create_event(device)
    backend.record_event(busy_event, busy)

    idle_event = backend.create_event(device)
    backend.record_event(idle_event, idle)
    backend.synchronize_event(idle_event, device)  # must not block on `busy`

    assert backend.query_event(busy_event) is False
    busy.synchronize()
    assert backend.query_event(busy_event) is True
    del keepalive


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

    keepalive = _enqueue_busy_work(stream, _BUSY_ITERS_SHORT)
    second = backend.create_event(device)
    backend.record_event(second, stream)
    imported = backend.import_event(backend.export_event(second, device), device)
    assert backend.query_event(imported) is False

    stream.synchronize()
    assert backend.query_event(imported) is True
    del keepalive


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
    """Child: import the received handle, synchronize on it, and report
    (query at import, blocked duration, query after).
    """
    torch.cuda.init()
    torch.cuda.set_device(0)
    backend = _make_backend(kind)
    device = torch.device(DEVICE)
    conn.send("ready")

    handle = conn.recv()
    event = backend.import_event(handle, device)
    query_at_import = backend.query_event(event)
    start = time.monotonic()
    backend.synchronize_event(event, device)
    blocked_for = time.monotonic() - start
    query_after = backend.query_event(event)
    conn.send((query_at_import, blocked_for, query_after))


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
        stream = torch.cuda.Stream()
        keepalive = _enqueue_busy_work(stream, _BUSY_ITERS_LONG)
        event = backend.create_event(device)
        backend.record_event(event, stream)
        busy_started = time.monotonic()
        parent_conn.send(backend.export_event(event, device))

        query_at_import, blocked_for, query_after = parent_conn.recv()
        stream.synchronize()
        busy_duration = time.monotonic() - busy_started

        assert query_at_import is False
        assert query_after is True
        # The child must have been genuinely blocked on the GPU work, and
        # released no later than the producer stream drained.
        assert blocked_for >= 0.05
        assert blocked_for <= busy_duration + 1.0
        del keepalive
    finally:
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0
