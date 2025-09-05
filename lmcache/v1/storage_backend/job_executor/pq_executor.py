# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Awaitable, Callable
import asyncio
import itertools

# First Party
from lmcache.v1.storage_backend.job_executor.base_executor import BaseJobExecutor

_SENTINEL = object()


class AsyncPQExecutor(BaseJobExecutor):
    "Execute async functions with a priority queue andusing event loop"

    def __init__(self, loop: asyncio.AbstractEventLoop, max_workers: int = 4):
        max_size = 0  # infinite
        self.loop = loop
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
        self.max_workers = max_workers
        for _ in range(max_workers):
            asyncio.run_coroutine_threadsafe(self._worker(), self.loop)
        self._closed = False

    async def submit_job(
        self,
        fn: Callable[..., Awaitable[Any]],
        **kwargs: Any,
    ) -> Any:
        # Assign highest priority by default
        priority = kwargs.pop("priority", 0)
        done: asyncio.Future[Any] = self.loop.create_future()
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
        for _ in self.max_workers:
            await self._q.put(_SENTINEL)
        if wait:
            await self._q.join()
            await asyncio.gather(*self._workers, return_exceptions=True)


class AsyncPQThreadPoolExecutor(AsyncPQExecutor):
    "Execute sync functions with a priority queue and using threadpool"

    def __init__(self, loop: asyncio.AbstractEventLoop, max_workers: int = 4):
        max_size = 0  # infinite
        self.loop = loop
        self._queue: asyncio.PriorityQueue[
            tuple[
                int,
                int,
                Callable[..., Any],
                dict[str, Any],
                asyncio.Future[Any],
            ]
            | object
        ] = asyncio.PriorityQueue(maxsize=max_size)
        self._counter = itertools.count()
        for _ in range(max_workers):
            asyncio.run_coroutine_threadsafe(self._worker(), self.loop)
        self._closed = False

    async def _worker(self):
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._q.task_done()
                break

            _, _, fn, kwargs, done = item
            try:
                result = await asyncio.to_thread(fn, **kwargs)
                done.set_result(result)
            except Exception as e:
                done.set_exception(e)
            finally:
                # decrement task count
                # join needs to wait until task count is zero
                self._queue.task_done()
