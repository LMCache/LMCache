# SPDX-License-Identifier: Apache-2.0
"""ROCm workaround: wait for the engine's producer event on the host.

By default a STORE / RETRIEVE makes the server's transfer stream wait on the
IPC event the engine recorded (``hipStreamWaitEvent`` on an imported event)
and enqueues the copies right away. On ROCm, while that wait is pending, it
can slow the engine's kernels on the same hardware pipe.

With ``--rocm-host-event-wait`` (or ``LMCACHE_ROCM_HOST_EVENT_WAIT=1``) the
server enqueues a transfer only once its producer event has completed, so no
barrier is ever left pending. Enable it when the LMCache MP server runs next
to the engine on ROCm; it is ignored elsewhere. Per engine instance the
transfers keep their arrival order: one whose event has already completed,
with nothing queued ahead of it, runs inline in the handler; the others wait
in a FIFO whose events one poller thread checks with non-blocking ``query()``
calls, and the handler answers with a :class:`DeferredResponse`. The poller
only checks events: a completed transfer is handed back to the affinity worker
that received it and runs there, keeping the affinity pool's one thread per
instance, and only if the instance still has the registration that supplied
its event. A transfer whose event does not
complete within ``timeout_s``, or that has not started at shutdown, is failed
without enqueuing device work.

The transfer itself runs the unchanged store / retrieve code: inside it the
instance's event backend returns the already imported event and skips the
stream wait on it.

Removal: once a ROCm release shows no slowdown of the producer in a
two-process reproducer (the consumer's stream waits on the producer's pending
IPC event while the producer runs small kernels; their time must equal the
time without the waiting peer), delete this module, the option and the
``_host_event_waiter`` uses in ``lmcache_driven_transfer.py`` and
``current_worker_poster`` in ``affinity_pool.py``.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
import functools
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import (
    DeferredResponse,
    current_worker_poster,
)

logger = init_logger(__name__)

ROCM_HOST_EVENT_WAIT_ENV = "LMCACHE_ROCM_HOST_EVENT_WAIT"
TIMEOUT_S = 60.0
_POLL_MIN_S = 50e-6
_POLL_MAX_S = 1e-3

# (handle, event, entry) of the transfer this thread runs after its event
# completed; entry is the registration that supplied the event.
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


def current_run_entry() -> Any:
    """Return the registration the running transfer is bound to, else None."""
    done = getattr(_completed, "value", None)
    return done[2] if done is not None else None


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


@dataclass(eq=False)
class _Job:
    instance_id: int
    ready: Callable[[], bool]
    run: Callable[[], Any]
    fail: Callable[[], Any]
    deadline: float
    post: Callable[[Callable[[], None]], None] | None
    outcome: str = ""
    response: DeferredResponse = field(default_factory=DeferredResponse)


def _run_completed(done: tuple[Any, Any, Any], run: Callable[[], Any]) -> Any:
    _completed.value = done
    try:
        return run()
    finally:
        _completed.value = None


def _settle(response: DeferredResponse, call: Callable[[], Any]) -> None:
    try:
        response.set_result(call())
    except BaseException as exc:
        response.set_exception(exc)


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
        # Instances with a transfer probing, handed to a worker, or running.
        self._busy: set[int] = set()
        # Handed to a worker, not started yet.
        self._posted: set[_Job] = set()
        self._active = 0
        self._stopped = False
        self._counters = dict.fromkeys(
            ("inline", "deferred", "timeout", "shutdown", "stale", "device_waits"), 0
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
        """Return the counters and the number of transfers not started yet."""
        with self._cond:
            pending = sum(len(q) for q in self._queues.values()) + len(self._posted)
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

        A waiting transfer later runs on the affinity worker that called this
        (on the poller thread for any other caller).

        Args:
            instance_id: The engine instance; transfers keep their order per
                instance and never run concurrently for it.
            entry: The instance's registry entry, or None if not registered.
                The transfer runs only while it is still the current one.
            handle: IPC handle of the producer event.
            run: The unchanged handler call.
            fail: Returns the handler's failure result (timeout, shutdown).

        Returns:
            The result of ``run`` (or ``fail``) if it ran inline, else a
            :class:`DeferredResponse` for it. Without a registered entry
            ``run`` runs right away (and rejects).
        """
        if entry is None:
            return _run_completed((None, None, None), run)
        backend = entry.event_backend
        event = backend.import_event(handle, entry.cache_context.device)
        job = _Job(
            instance_id,
            lambda: backend.query_event(event),
            lambda: _run_completed((handle, event, entry), run),
            fail,
            self._clock() + self._timeout_s,
            current_worker_poster(),
        )
        with self._cond:
            stopped = self._stopped
            if stopped:
                self._counters["shutdown"] += 1
            elif self._queues.get(instance_id) or instance_id in self._busy:
                self._enqueue(job)
                return job.response
            else:
                self._busy.add(instance_id)
                self._active += 1
        if stopped:
            return fail()
        try:
            try:
                ready = job.ready()
            except Exception:
                ready = False
            with self._cond:
                stopped = self._stopped
                if stopped:
                    self._counters["shutdown"] += 1
                elif not ready:
                    self._enqueue(job)
                    return job.response
                else:
                    self._counters["inline"] += 1
            return fail() if stopped else job.run()
        finally:
            self._done_executing(instance_id)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Fail every transfer that has not started, wait for the running ones,
        and stop the poller thread."""
        with self._cond:
            self._stopped = True
            claimed = [job for q in self._queues.values() for job in q]
            claimed.extend(self._posted)
            self._queues.clear()
            self._posted.clear()
            self._counters["shutdown"] += len(claimed)
            self._cond.notify_all()
        for job in claimed:
            _settle(job.response, job.fail)
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self._active:
                logger.error(
                    "%d host event wait transfer(s) still running at shutdown",
                    self._active,
                )
        self._thread.join(max(deadline - time.monotonic(), 0.0))

    def _enqueue(self, job: _Job) -> None:
        """Queue ``job`` for the poller; the caller holds ``_cond``."""
        self._queues.setdefault(job.instance_id, deque()).append(job)
        self._counters["deferred"] += 1
        self._cond.notify_all()

    def _done_executing(self, instance_id: int) -> None:
        with self._cond:
            self._active -= 1
            self._busy.discard(instance_id)
            self._cond.notify_all()

    def _poll(self) -> None:
        delay = _POLL_MIN_S
        while True:
            with self._cond:
                while not self._stopped and not any(
                    q and iid not in self._busy for iid, q in self._queues.items()
                ):
                    self._cond.wait()
                if self._stopped:
                    return
                heads = [
                    (iid, q[0])
                    for iid, q in self._queues.items()
                    if q and iid not in self._busy
                ]
            progressed = False
            for iid, job in heads:
                outcome = self._outcome(job)
                if outcome is None:
                    continue
                with self._cond:
                    queue = self._queues.get(iid)
                    if not queue or queue[0] is not job:
                        continue
                    queue.popleft()
                    self._busy.add(iid)
                    self._posted.add(job)
                    job.outcome = outcome
                continuation = functools.partial(self._continue, job)
                if job.post is not None:
                    job.post(continuation)
                else:
                    continuation()
                progressed = True
            if progressed:
                delay = _POLL_MIN_S
            else:
                time.sleep(delay)
                delay = min(delay * 2, _POLL_MAX_S)

    def _continue(self, job: _Job) -> None:
        """Run or fail a handed-over transfer on its affinity worker."""
        with self._cond:
            if job not in self._posted:
                return  # failed by stop()
            self._posted.discard(job)
            self._active += 1
            if job.outcome != "run":
                self._counters[job.outcome] += 1
        try:
            if job.outcome != "run":
                logger.warning(
                    "Failing a transfer whose producer event did not complete (%s)",
                    job.outcome,
                )
            _settle(job.response, job.run if job.outcome == "run" else job.fail)
        finally:
            self._done_executing(job.instance_id)

    def _outcome(self, job: _Job) -> str | None:
        try:
            if job.ready():
                return "run"
        except Exception:
            logger.exception("Producer event query failed; failing the transfer")
            return "timeout"
        return "timeout" if self._clock() >= job.deadline else None
