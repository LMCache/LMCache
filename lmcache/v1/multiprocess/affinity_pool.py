# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` execute sequentially in FIFO
order. Bindings stay on their worker while work is pending, an idle key sharing
a worker may move to a reclaimed slot on its next submission.

This serializes GPU-bound request handlers (STORE / RETRIEVE) for each vLLM
instance, eliminating the need for per-instance locks on the shared temporary
GPU buffer.
"""

# Standard
from collections import Counter
from concurrent.futures import Future
import queue
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN = object()


class AffinityThreadPool:
    """Thread pool that routes tasks to workers by affinity key.

    Submission and key retirement are thread-safe. A retired key keeps its
    worker until all queued/running tasks for that key have drained.
    Reclaimed capacity also lets idle keys leave an overloaded worker.
    Callers must stop submitting before shutting down the pool.

    Args:
        max_workers: Number of worker threads.
        thread_name_prefix: Prefix for worker thread names.
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "affinity",
    ) -> None:
        self._num_workers = max_workers
        self._queues: list[queue.Queue] = [queue.Queue() for _ in range(max_workers)]
        self._threads: list[threading.Thread] = []
        # Maps an affinity_key -> the worker slot (thread index) bound to it.
        self._key_to_slot: dict[int, int] = {}
        self._pending: Counter[int] = Counter()
        self._retired: set[int] = set()
        # Avoid rescanning established bindings until capacity is reclaimed.
        self._rebalance = False
        self._lock = threading.Lock()
        self._overflow_warned = False
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i],),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Created AffinityThreadPool '%s' with %d worker slots: up to %d "
            "distinct affinity keys each bind to their own thread before slots "
            "are shared. Compare this against the number of clients expected to "
            "connect to confirm routing.",
            thread_name_prefix,
            max_workers,
            max_workers,
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs, affinity_key = item
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    self._finish_task(affinity_key)
                    future.set_exception(exc)
                else:
                    self._finish_task(affinity_key)
                    future.set_result(result)
            else:
                self._finish_task(affinity_key)

    def _finish_task(self, affinity_key: int) -> None:
        with self._lock:
            self._pending[affinity_key] -= 1
            if self._pending[affinity_key] == 0:
                del self._pending[affinity_key]
                if affinity_key in self._retired:
                    self._retired.remove(affinity_key)
                    del self._key_to_slot[affinity_key]
                    self._rebalance = True

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Choose a slot without moving running or queued work.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None and (not self._rebalance or self._pending[affinity_key]):
            return slot

        # Reuse freed slots before sharing a worker.
        bindings = Counter(self._key_to_slot.values())
        if slot is not None:
            free_slot = next(
                (i for i in range(self._num_workers) if not bindings[i]), None
            )
            self._rebalance = free_slot is not None and any(
                count > 1 for count in bindings.values()
            )
            if free_slot is None or bindings[slot] == 1:
                return slot
            slot = free_slot
        else:
            slot = min(range(self._num_workers), key=bindings.__getitem__)
        is_overflow = bindings[slot] > 0
        self._key_to_slot[affinity_key] = slot

        logger.info(
            "AffinityThreadPool: affinity_key=%d assigned to worker "
            "slot %d of %d (thread %s); %d distinct key(s) now bound",
            affinity_key,
            slot,
            self._num_workers,
            self._threads[slot].name,
            len(self._key_to_slot),
        )
        if is_overflow and not self._overflow_warned:
            self._overflow_warned = True
            logger.warning(
                "AffinityThreadPool: affinity_key=%d wrapped onto worker slot "
                "%d (only %d workers), so it shares a thread with an earlier "
                "key and their tasks are serialized. Increase the worker count "
                "to give each client its own thread.",
                affinity_key,
                slot,
                self._num_workers,
            )
        return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        with self._lock:
            slot = self._slot_for_key(affinity_key)
            self._pending[affinity_key] += 1
            self._queues[slot].put((future, fn, args, kwargs, affinity_key))
        return future

    def release_key(self, affinity_key: int) -> None:
        """Retire a binding without interrupting or moving outstanding work.

        Thread-safe and idempotent. Submissions for the same key keep using its
        old worker until its queue drains, including submissions after release.
        Once drained, the next submission may bind the key to a different worker.
        Idle keys on shared workers can reuse the freed slot on submission.
        """
        with self._lock:
            if self._pending[affinity_key]:
                self._retired.add(affinity_key)
            else:
                if self._key_to_slot.pop(affinity_key, None) is not None:
                    self._rebalance = True

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        Sends a shutdown sentinel to every worker.  If *wait* is true, blocks
        until all workers have exited.
        """
        for q in self._queues:
            q.put(_SHUTDOWN)
        if wait:
            for t in self._threads:
                t.join()
