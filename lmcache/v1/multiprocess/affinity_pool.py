# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread.  Within each worker, tasks execute sequentially in FIFO order.

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.

Worker assignment is *dynamic*: the first time a key is submitted it is bound
to the next free worker slot (round-robin over arrival order) and that binding
is remembered for the lifetime of the pool, so every later task for that key
reuses the same thread. There is no ``key % num_workers`` hashing, so the
numeric value of the key is irrelevant -- as long as the number of *distinct*
keys does not exceed ``max_workers``, each key gets its own dedicated worker
thread and distinct keys never collide onto one slot while another sits idle.
When there are more distinct keys than workers, the surplus keys wrap around
and share slots with earlier keys (their tasks are then serialized); the first
such overflow is logged once at WARNING level. Increase ``max_workers`` to
``>=`` the number of distinct clients to restore one worker per client.
"""

# Standard
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
        # Dynamic ``affinity_key -> worker slot`` assignment. The first time a
        # key is submitted it is bound to the next free slot (round-robin) and
        # cached here so all later tasks for that key reuse the same worker
        # thread. ``_next_slot`` is the monotonically increasing arrival
        # counter; ``slot = _next_slot % _num_workers``. Guarded by
        # ``_assign_lock`` because submit() may be called concurrently from
        # multiple threads; ``_overflow_warned`` makes the overflow warning
        # fire at most once.
        self._assign_lock = threading.Lock()
        self._key_to_slot: dict[int, int] = {}
        self._next_slot = 0
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

    @staticmethod
    def _worker(q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs = item
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except BaseException as exc:
                    future.set_exception(exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Return the worker slot bound to ``affinity_key``, assigning on first use.

        The first time a key is seen it is bound to the next free slot
        (``_next_slot % _num_workers``) and the binding is cached, so the
        mapping is independent of the key's numeric value: any set of up to
        ``_num_workers`` distinct keys lands on distinct worker threads. Once
        more than ``_num_workers`` distinct keys have been seen, later keys wrap
        around and share a slot with an earlier key; the first such overflow is
        logged once at WARNING level.

        Args:
            affinity_key: The routing key for the current submission.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        # Fast path: the binding already exists. Dict reads are atomic under the
        # GIL, so the common already-assigned case stays lock-free; the GPU work
        # this pool serializes dwarfs any contention here anyway.
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None:
            return slot

        # Slow path: first time we see this key -- assign a slot under the lock.
        with self._assign_lock:
            # Re-check: another thread may have assigned it while we waited.
            slot = self._key_to_slot.get(affinity_key)
            if slot is not None:
                return slot

            slot = self._next_slot % self._num_workers
            is_overflow = self._next_slot >= self._num_workers
            self._key_to_slot[affinity_key] = slot
            self._next_slot += 1

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
                    "AffinityThreadPool: more distinct affinity keys than "
                    "workers (%d) -- key %d wrapped onto worker slot %d, which "
                    "is already bound to an earlier key, so these clients share "
                    "one thread and are serialized. Increase the worker count "
                    "(e.g. --max-gpu-workers) to >= the number of distinct "
                    "clients for one worker per client.",
                    self._num_workers,
                    affinity_key,
                    slot,
                )
            return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        The first ``max_workers`` distinct keys each get their own worker
        thread regardless of their numeric values; further distinct keys wrap
        around and share a worker (logged once at WARNING level). Tasks sharing
        a key -- and tasks whose keys share a slot -- execute sequentially in
        FIFO order on that one thread.

        The first time each key is routed, its ``key -> worker slot (thread)``
        assignment is logged once at INFO level, so a run's log positively
        shows which worker thread each client was bound to.

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        slot = self._slot_for_key(affinity_key)
        self._queues[slot].put((future, fn, args, kwargs))
        return future

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
