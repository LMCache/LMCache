# SPDX-License-Identifier: Apache-2.0
"""Cross-process ordering contract for ``EventIPCBackend``.

``import_event`` must return an event that genuinely represents the *producer's*
pending work.  The LMCache-driven transfer path relies on this: the producer
records an event on the stream that wrote the KV cache and ships only the handle
(``worker_transfer.py`` ``submit_store``/``submit_retrieve``), then the consumer
imports it and calls ``wait_event`` before reading those blocks
(``lmcache_driven_transfer.py``).  Nothing else orders the two processes.

A backend that returns an already-completed event -- for instance one that tries
to substitute a local ``current_stream().synchronize()`` for the real handle --
turns that ``wait_event`` into a no-op and lets the consumer read KV blocks the
producer has not finished writing.  That failure is silent and surfaces as wrong
cache contents rather than a crash, so it is worth pinning down.

The check is on data visibility after ``wait_event``, not on ``query_event`` of
the imported event: on CUDA an imported IPC event reports complete in the
importing process regardless of the producer's progress, because the recorded
state is per-process bookkeeping.  Cross-process ordering is carried by
``wait_event`` alone, so that is what this asserts.

The producer holds the GPU for roughly two seconds before writing, against a
sub-millisecond handle round trip.  A backend that drops the ordering therefore
reads unwritten memory with a wide margin; if the consumer is ever starved long
enough to miss it the test passes rather than failing, so it does not flake red.
"""

# Third Party
import pytest
import torch
import torch.multiprocessing as mp

# First Party
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

MAGIC = 7.0
NUMEL = 1 << 22
# ~2s of GPU busy-wait at 2 GHz, against a sub-millisecond handle round trip.
SLEEP_CYCLES = 4_000_000_000
HANDLE_TIMEOUT_S = 180.0


def _event_ipc_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        get_event_ipc_backend(torch.device("cuda:0")).check_event_support(
            torch.device("cuda:0")
        )
    except Exception:
        return False
    return True


requires_event_ipc = pytest.mark.skipif(
    not _event_ipc_supported(),
    reason="requires a CUDA/HIP device with interprocess event support",
)


def _producer(buf: torch.Tensor, handle_q: "mp.Queue") -> None:
    """Enqueue a slow write, record an event behind it, ship only the handle.

    Runs in a spawned child process and mirrors the worker side of the
    ``lmcache_driven`` path: the event is recorded on the stream that performs
    the write, and the handle is sent while that write is still in flight.

    Args:
        buf: Device buffer shared with the parent process, filled with
            ``MAGIC`` once the enqueued delay elapses.
        handle_q: Queue used to send the exported event handle to the parent.
    """
    device = buf.device
    torch.cuda.set_device(device)
    backend = get_event_ipc_backend(device)
    stream = torch.cuda.Stream(device=device)

    # Warm the fill kernel while the device is idle. In a freshly spawned
    # process the first ``fill_`` triggers a lazy module load that has to wait
    # for the device, so issuing it after the delay below blocks the host until
    # the delay drains -- measured at ~1.97s against a 2s delay on CUDA, versus
    # ~7ms on ROCm. That would push the export past the write and leave nothing
    # for the consumer to race, so the test would pass whatever the backend did.
    buf.fill_(0.0)
    torch.cuda.synchronize()

    event = backend.create_event(device)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SLEEP_CYCLES)
        buf.fill_(MAGIC)
        backend.record_event(event, stream)
    # Deliberately no synchronize: this mirrors the real producer, which hands
    # the handle to the message queue while the write is still in flight.
    handle_q.put(bytes(backend.export_event(event, device)))
    stream.synchronize()


@requires_event_ipc
def test_imported_event_orders_consumer_behind_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting on an imported event must make the producer's write visible.

    The producer enqueues a long delay, then the write, then records the event,
    and ships the handle without synchronizing. A backend that returns a real
    imported event orders the consumer behind that write; one that returns a
    locally completed event does not, and the consumer observes zeros.

    Args:
        monkeypatch: Used to clear ``CUDA_LAUNCH_BLOCKING`` for the child.
    """
    # CUDA_LAUNCH_BLOCKING makes every launch synchronous, which drains the
    # delay before the write is even issued and removes the window this test
    # depends on. The producer is spawned, so clearing it here is enough -- the
    # child reads the environment when it initialises its own context.
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    device = torch.device("cuda:0")
    backend = get_event_ipc_backend(device)

    buf = torch.zeros(NUMEL, device=device)
    buf.share_memory_()

    ctx = mp.get_context("spawn")
    handle_q: "mp.Queue" = ctx.Queue()
    producer = ctx.Process(target=_producer, args=(buf, handle_q))
    producer.start()
    try:
        handle = handle_q.get(timeout=HANDLE_TIMEOUT_S)
        imported = backend.import_event(handle, device)

        consumer_stream = torch.cuda.Stream(device=device)
        backend.wait_event(imported, consumer_stream)
        with torch.cuda.stream(consumer_stream):
            observed = buf.clone()
        consumer_stream.synchronize()

        unwritten = int((observed != MAGIC).sum().item())
        assert unwritten == 0, (
            f"consumer read {unwritten}/{NUMEL} elements before the producer "
            "finished writing them"
        )
    finally:
        producer.join(timeout=60)
        if producer.is_alive():
            producer.terminate()
            producer.join(timeout=10)
