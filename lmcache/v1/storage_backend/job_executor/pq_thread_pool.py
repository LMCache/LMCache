# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import Any, Callable
import queue
import threading

# First Party
from lmcache.v1.storage_backend.job_executor.base_executor import BaseJobExecutor


class PQThreadPoolExecutor(BaseJobExecutor):
    def __init__(self, max_workers: int = 4):
        self.tasks: queue.PriorityQueue[
            tuple[int, Callable[..., Any], dict[str, Any], Future[Any]]
        ] = queue.PriorityQueue()
        self.shutdown_flag = threading.Event()
        self.threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(max_workers)
        ]
        for t in self.threads:
            t.start()

    def submit_job(
        self,
        fn: Callable,
        **kwargs,
    ) -> Future:
        # Assign highest priority by default
        priority = kwargs.pop("priority", 0)
        fut: Future[Any] = Future()
        self.tasks.put((priority, fn, kwargs, fut))
        return fut

    def _worker(self):
        while not self.shutdown_flag.is_set():
            priority, fn, kwargs, fut = self.tasks.get(block=True)
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                result = fn(**kwargs)
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
            finally:
                # decrement task count
                self.tasks.task_done()

    def shutdown(self, wait=True):
        self.shutdown_flag.set()
        if wait:
            for t in self.threads:
                t.join()
