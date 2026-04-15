# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread (determined by ``affinity_key % num_workers``).  Within each
worker, tasks execute sequentially in FIFO order.

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.
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

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        slot = affinity_key % self._num_workers
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
