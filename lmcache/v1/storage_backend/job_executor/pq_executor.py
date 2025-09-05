# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Awaitable, Callable
import asyncio
import itertools

# First Party
from lmcache.v1.storage_backend.job_executor.base_executor import BaseJobExecutor

_SENTINEL = object()


class AsyncPQExecutor(BaseJobExecutor):
    def __init__(self, max_workers: int = 4):
        max_size = 0  # infinite
        self._queue: asyncio.PriorityQueue[
            tuple[
                int,
                int,
                Callable[..., Awaitable[Any]],
                dict[str, Any],
                asyncio.Future[Any],
            ]
            | object
        ] = asyncio.PriorityQueue(maxsize=max_size)
        self._counter = itertools.count()
        self._workers = [
            asyncio.create_task(self._worker()) for _ in range(max_workers)
        ]
        self._closed = False

    async def submit_job(
        self,
        fn: Callable[..., Awaitable[Any]],
        **kwargs: Any,
    ) -> Any:
        # Assign highest priority by default
        priority = kwargs.pop("priority", 0)
        loop = asyncio.get_running_loop()
        done: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((priority, next(self._counter), fn, kwargs, done))
        return await done

    async def _worker(self):
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._q.task_done()
                break

            _, _, fn, kwargs, done = item
            try:
                result = await fn(**kwargs)
                done.set_result(result)
            except Exception as e:
                done.set_exception(e)
            finally:
                # decrement task count
                # join needs to wait until task count is zero
                self._queue.task_done()

    async def shutdown(self, wait=True):
        self._closed = True
        for _ in self._workers:
            await self._q.put(_SENTINEL)
        if wait:
            await self._q.join()
            await asyncio.gather(*self._workers, return_exceptions=True)
