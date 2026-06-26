# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread (determined by ``affinity_key % num_workers``).  Within each
worker, tasks execute sequentially in FIFO order.

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.

For one worker per instance, callers should pass a *dense* ``affinity_key``
(a rank index ``0..world_size-1``) and size the pool with
``max_workers >= world_size``: then ``affinity_key % num_workers`` is a 1:1
mapping. A hashed or sparse key gives no such guarantee -- distinct keys can
collide onto one worker while others sit idle (see ``_maybe_warn_collision``).
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
        # Collision tracking: maps a worker slot to the first affinity key
        # routed there, so we can warn once when two distinct keys (clients)
        # land on the same slot and get serialized onto one thread. Guarded by
        # ``_collision_lock`` since submit() may be called from multiple
        # threads.
        self._collision_lock = threading.Lock()
        self._slot_first_key: dict[int, int] = {}
        self._collision_warned = False
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i],),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.debug(
            "Created AffinityThreadPool with %d workers (prefix=%s)",
            max_workers,
            thread_name_prefix,
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
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker determined by *affinity_key*.

        For an even one-key-per-worker spread, pass a dense ``affinity_key``
        (e.g. a rank index ``0..world_size-1``) and size the pool with
        ``max_workers >= world_size``. Sparse or hashed keys may collide onto a
        shared worker even when idle workers exist; the first such collision is
        logged once at WARNING level.

        The first time each key is routed, its ``key -> worker slot (thread)``
        assignment is logged once at INFO level, so a run's log positively
        shows which worker thread each client (rank) was bound to.

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        slot = affinity_key % self._num_workers
        self._observe_routing(slot, affinity_key)
        self._queues[slot].put((future, fn, args, kwargs))
        return future

    def _observe_routing(self, slot: int, affinity_key: int) -> None:
        """Log each key's worker-slot assignment, and warn once on collision.

        The first key routed to a slot is logged at INFO as a positive
        confirmation of its ``key -> slot (thread)`` binding. If a *second*
        distinct key later lands on the same slot, those two clients share one
        worker thread (their tasks are serialized) while other workers may sit
        idle -- usually because the pool has fewer workers than distinct
        clients, or because keys are not dense. The first such collision is
        logged once at WARNING level to avoid log spam.

        Args:
            slot: The worker slot ``affinity_key`` mapped to.
            affinity_key: The routing key for the current submission.
        """
        with self._collision_lock:
            existing = self._slot_first_key.get(slot)
            if existing is None:
                self._slot_first_key[slot] = affinity_key
                logger.info(
                    "AffinityThreadPool: affinity_key=%d routed to worker "
                    "slot %d of %d (thread %s)",
                    affinity_key,
                    slot,
                    self._num_workers,
                    self._threads[slot].name,
                )
            elif existing != affinity_key and not self._collision_warned:
                self._collision_warned = True
                logger.warning(
                    "AffinityThreadPool: affinity keys %d and %d both map to "
                    "worker slot %d of %d -- these clients share one thread and "
                    "are serialized. Increase the worker count (e.g. "
                    "--max-gpu-workers) to >= the number of distinct clients "
                    "for one worker per client.",
                    existing,
                    affinity_key,
                    slot,
                    self._num_workers,
                )

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
