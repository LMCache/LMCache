# SPDX-License-Identifier: Apache-2.0
"""ROCm workaround: wait for the engine's producer event on the host.

By default a STORE / RETRIEVE makes the server's transfer stream wait on the
IPC event the engine recorded (``hipStreamWaitEvent`` on an imported event)
and enqueues the copies right away. On MI3xx the command processor cannot
keep an unsatisfied cross-queue barrier resident: it disconnects the queue
and reconnects it on a timer, and those wakeups slow the engine's kernels on
the same hardware pipe while the engine runs the step that will signal the
event (about 30 % for prefill steps on MI355X).

With ``--rocm-host-event-wait`` (or ``LMCACHE_ROCM_HOST_EVENT_WAIT=1``) the
server enqueues a transfer only once its producer event has completed, so no
barrier is ever left pending. Enable it when the LMCache MP server runs next
to the engine on ROCm; it is ignored elsewhere. Per engine instance the
transfers keep their arrival order: one whose event has already completed,
with nothing queued ahead of it, runs inline in the handler; the others wait
in a FIFO that one poller thread drains with non-blocking ``query()`` calls,
and the handler answers with a :class:`DeferredResponse`. A transfer whose
event does not complete within ``timeout_s``, or that is still queued at
shutdown, is failed without enqueuing device work.

The transfer itself runs the unchanged store / retrieve code: inside it the
instance's event backend returns the already imported event and skips the
stream wait on it.

Removal: once a ROCm release shows no slowdown of the producer in a
two-process reproducer (the consumer's stream waits on the producer's pending
IPC event while the producer runs small kernels; their time must equal the
time without the waiting peer), delete this module, the option and the
``_host_event_waiter`` uses in ``lmcache_driven_transfer.py``.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import DeferredResponse

logger = init_logger(__name__)

ROCM_HOST_EVENT_WAIT_ENV = "LMCACHE_ROCM_HOST_EVENT_WAIT"
TIMEOUT_S = 60.0
_POLL_MIN_S = 50e-6
_POLL_MAX_S = 1e-3

# (handle, event) of the transfer this thread runs after its event completed
_completed = threading.local()


def _running_on_hip() -> bool:
    # Third Party
    import torch

    return torch.version.hip is not None


def create_host_event_waiter(enabled: bool) -> "HostEventWaiter | None":
    """Return the waiter if the option is enabled on ROCm, else None.

    Args:
        enabled: The ``--rocm-host-event-wait`` server option;
            ``LMCACHE_ROCM_HOST_EVENT_WAIT=1`` enables it as well.
    """
    env = os.environ.get(ROCM_HOST_EVENT_WAIT_ENV, "").strip().lower()
    if not (enabled or env in ("1", "true")):
        return None
    if not _running_on_hip():
        logger.warning(
            "--rocm-host-event-wait is only needed on ROCm; ignored on this platform"
        )
        return None
    logger.info("ROCm host event wait enabled for LMCache-driven STORE / RETRIEVE")
    return HostEventWaiter()


class _HostWaitEventBackend:
    """An instance's event backend while the host event wait is enabled.

    Delegates to the platform backend, except inside a transfer that the
    waiter runs after its producer event completed: there ``import_event`` of
    that handle returns the polled event and ``wait_event`` on it is a no-op.
    """

    def __init__(self, inner: Any, waiter: "HostEventWaiter") -> None:
        self._inner = inner
        self._waiter = waiter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def import_event(self, handle: bytes, device: object) -> object:
        done = getattr(_completed, "value", None)
        if done is not None and done[0] == handle:
            return done[1]
        return self._inner.import_event(handle, device)

    def wait_event(self, event: object, stream: object) -> None:
        done = getattr(_completed, "value", None)
        if done is not None and done[1] is event:
            return
        self._waiter.count("device_waits")
        self._inner.wait_event(event, stream)


@dataclass
class _Job:
    ready: Callable[[], bool]
    run: Callable[[], Any]
    fail: Callable[[], Any]
    deadline: float
    response: DeferredResponse = field(default_factory=DeferredResponse)


def _run_completed(done: tuple[Any, Any], run: Callable[[], Any]) -> Any:
    _completed.value = done
    try:
        return run()
    finally:
        _completed.value = None


class HostEventWaiter:
    """Runs transfers once their producer event has completed on the host.

    Args:
        timeout_s: Seconds a queued transfer may wait before it is failed.
        clock: Monotonic clock; injectable for tests.
    """

    def __init__(
        self, timeout_s: float = TIMEOUT_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._timeout_s = timeout_s
        self._clock = clock
        self._cond = threading.Condition()
        self._queues: dict[int, deque[_Job]] = {}
        self._busy: set[int] = set()
        self._stopped = False
        self._counters = dict.fromkeys(
            ("inline", "deferred", "timeout", "shutdown", "device_waits"), 0
        )
        self._thread = threading.Thread(
            target=self._poll, daemon=True, name="lmcache-host-event-wait"
        )
        self._thread.start()

    @staticmethod
    def in_run() -> bool:
        """Return whether this thread runs a transfer for the waiter."""
        return getattr(_completed, "value", None) is not None

    def wrap(self, backend: Any) -> _HostWaitEventBackend:
        """Return the event backend to register for an instance."""
        return _HostWaitEventBackend(backend, self)

    def count(self, name: str) -> None:
        with self._cond:
            self._counters[name] += 1

    def stats(self) -> dict[str, int]:
        """Return the counters and the number of queued transfers."""
        with self._cond:
            pending = sum(len(q) for q in self._queues.values())
            return {**self._counters, "pending": pending}

    def submit_after_event(
        self,
        instance_id: int,
        entry: Any,
        handle: bytes,
        run: Callable[[], Any],
        fail: Callable[[], Any],
    ) -> Any:
        """Run the handler call ``run`` once the event behind ``handle`` completed.

        Args:
            instance_id: The engine instance; transfers keep their order per
                instance and never run concurrently for it.
            entry: The instance's registry entry, or None if not registered.
            handle: IPC handle of the producer event.
            run: The unchanged handler call.
            fail: Returns the handler's failure result (timeout, shutdown).

        Returns:
            The result of ``run`` if it ran inline, else a
            :class:`DeferredResponse` for the result of ``run`` or ``fail``.
            Without a registered entry ``run`` runs right away (and rejects).
        """
        if entry is None:
            return _run_completed((None, None), run)
        backend = entry.event_backend
        event = backend.import_event(handle, entry.cache_context.device)
        job = _Job(
            lambda: backend.query_event(event),
            lambda: _run_completed((handle, event), run),
            fail,
            self._clock() + self._timeout_s,
        )
        with self._cond:
            if self._stopped:
                self._counters["shutdown"] += 1
                return fail()
            queue = self._queues.setdefault(instance_id, deque())
            probe = not queue and instance_id not in self._busy
            if probe:
                self._busy.add(instance_id)
        if probe:
            try:
                ready = job.ready()
            except Exception:
                ready = False  # the poller queries again and fails the job
            if ready:
                self.count("inline")
                try:
                    return job.run()
                finally:
                    self._release(instance_id)
        with self._cond:
            queue.append(job)
            self._busy.discard(instance_id)
            self._counters["deferred"] += 1
            self._cond.notify_all()
        return job.response

    def stop(self, timeout_s: float = 10.0) -> None:
        """Fail every queued transfer and stop the poller thread."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout_s)

    def _release(self, instance_id: int) -> None:
        with self._cond:
            self._busy.discard(instance_id)
            self._cond.notify_all()

    def _poll(self) -> None:
        delay = _POLL_MIN_S
        while True:
            with self._cond:
                while not self._stopped and not any(self._queues.values()):
                    self._cond.wait()
                if self._stopped and not any(self._queues.values()):
                    return
                heads = [
                    (iid, q[0])
                    for iid, q in self._queues.items()
                    if q and iid not in self._busy
                ]
                stopping = self._stopped
            progressed = False
            for iid, job in heads:
                outcome = self._outcome(job, stopping)
                if outcome is None:
                    continue
                with self._cond:
                    self._queues[iid].popleft()
                    self._busy.add(iid)
                    if outcome != "run":
                        self._counters[outcome] += 1
                try:
                    if outcome != "run":
                        logger.warning(
                            "Failing a transfer whose producer event did not "
                            "complete (%s)",
                            outcome,
                        )
                    result = job.run() if outcome == "run" else job.fail()
                    job.response.set_result(result)
                except BaseException as exc:
                    job.response.set_exception(exc)
                finally:
                    self._release(iid)
                progressed = True
            if progressed:
                delay = _POLL_MIN_S
            else:
                time.sleep(delay)
                delay = min(delay * 2, _POLL_MAX_S)

    def _outcome(self, job: _Job, stopping: bool) -> str | None:
        if stopping:
            return "shutdown"
        try:
            if job.ready():
                return "run"
        except Exception:
            logger.exception("Producer event query failed; failing the transfer")
            return "timeout"
        return "timeout" if self._clock() >= job.deadline else None
