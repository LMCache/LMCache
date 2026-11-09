# SPDX-License-Identifier: Apache-2.0
"""Bounded FS windows shared by MP streaming requests."""

# Standard
from collections.abc import Callable
import threading
import time

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2Adapter
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.distributed.storage_controllers.streaming_window_pool import (
    StreamingWindow,
    StreamingWindowPool,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError


class StreamingLoadTimeout(LMCacheTimeoutError):
    """A load timed out; its windows remain owned until the I/O completes."""


class FSStreamingReader:
    """Own a pinned pool and use the configured filesystem adapter to fill it.

    Timed-out loads retain their windows until the adapter reports completion;
    releasing those windows earlier would allow an in-flight disk read to write
    into memory already reused by a different request.
    """

    def __init__(
        self, adapter: FSL2Adapter, capacity_bytes: int, operation_timeout: float = 30.0
    ) -> None:
        if capacity_bytes <= 0 or capacity_bytes % 4096:
            raise ValueError("streaming pool size must be a positive multiple of 4096")
        if operation_timeout <= 0:
            raise ValueError("operation timeout must be positive")
        self.adapter = adapter
        self.capacity_bytes = capacity_bytes
        self.operation_timeout = operation_timeout
        self._allocator = L1MemoryManager(
            L1MemoryManagerConfig(
                size_in_bytes=capacity_bytes,
                align_bytes=4096,
                use_lazy=False,
                shm_name="",
            )
        )
        self.pool = StreamingWindowPool(self._allocator, capacity_bytes)
        self._condition = threading.Condition()
        self._closing = False
        self._closed = False
        self._inflight = 0
        self._abandoned: list[
            tuple[Callable[[], Bitmap | None], list[StreamingWindow]]
        ] = []
        self._reaper = threading.Thread(target=self._reap_abandoned, daemon=True)
        self._reaper.start()

    def can_fit(self, layouts: dict[int, MemoryLayoutDesc]) -> bool:
        """Return whether every object layout can fit in one window."""
        return all(
            0 < get_size_bytes(layout.shapes, layout.dtypes) <= self.capacity_bytes
            for layout in layouts.values()
        )

    def max_window_count(self, layout: MemoryLayoutDesc) -> int:
        """Return how many objects of this layout fit in the pool."""
        size = get_size_bytes(layout.shapes, layout.dtypes)
        aligned = ((size + 4095) // 4096) * 4096
        return self.capacity_bytes // aligned if aligned else 0

    def lookup(
        self, keys: list[ObjectKey], layouts: dict[int, MemoryLayoutDesc]
    ) -> Bitmap:
        """Look up objects without loading them into the hot cache."""
        self._begin()
        try:
            task = self.adapter.submit_lookup_and_lock_task(keys, layouts)
        except BaseException:
            self._finish()
            raise
        return self._wait(lambda: self.adapter.query_lookup_and_lock_result(task), [])

    def load_many(self, keys: list[ObjectKey], windows: list[StreamingWindow]) -> bool:
        """Load a batch, retaining its windows if I/O outlives the timeout."""
        if len(keys) != len(windows) or not keys:
            raise ValueError("keys and windows must have the same nonzero length")
        self._begin()
        pending: list[tuple[int, int]] = []
        result = Bitmap(len(keys))

        def poll() -> Bitmap | None:
            for index, task in list(pending):
                completed = self.adapter.query_load_result(task)
                if completed is None:
                    continue
                if completed.popcount() == 1:
                    result.set(index)
                pending.remove((index, task))
            return result if not pending else None

        try:
            for index, (key, window) in enumerate(zip(keys, windows, strict=True)):
                task = self.adapter.submit_load_task([key], [window.memory_obj])
                pending.append((index, task))
        except BaseException as exc:
            if pending:
                self._abandon(poll, windows)
                raise StreamingLoadTimeout("streaming batch submission failed") from exc
            self._finish()
            raise
        result = self._wait(poll, windows)
        return result.popcount() == len(keys)

    def close(self, timeout: float = 30.0) -> None:
        """Stop admission and close only after outstanding I/O has completed.

        Raises:
            RuntimeError: An operation or window is still active at the deadline.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self.pool.shutdown()
            self._condition.notify_all()
            deadline = time.monotonic() + timeout
            while self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("streaming I/O is still outstanding")
                self._condition.wait(remaining)
            self.pool.close()
            self._allocator.close()
            self._closed = True
            self._condition.notify_all()
        self._reaper.join()

    def _begin(self) -> None:
        with self._condition:
            if self._closing:
                raise RuntimeError("streaming reader is closed")
            self._inflight += 1

    def _finish(self) -> None:
        with self._condition:
            self._inflight -= 1
            self._condition.notify_all()

    def _wait(
        self, poll: Callable[[], Bitmap | None], windows: list[StreamingWindow]
    ) -> Bitmap:
        deadline = time.monotonic() + self.operation_timeout
        try:
            while True:
                result = poll()
                if result is not None:
                    self._finish()
                    return result
                with self._condition:
                    if self._closing or time.monotonic() >= deadline:
                        raise LMCacheTimeoutError("timed out waiting for streaming I/O")
                    self._condition.wait(
                        max(0, min(0.001, deadline - time.monotonic()))
                    )
        except BaseException as exc:
            self._abandon(poll, windows)
            if windows:
                raise StreamingLoadTimeout("streaming load has not completed") from exc
            raise

    def _abandon(
        self, poll: Callable[[], Bitmap | None], windows: list[StreamingWindow]
    ) -> None:
        with self._condition:
            self._abandoned.append((poll, list(windows)))
            self._condition.notify_all()

    def _reap_abandoned(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                if not self._abandoned:
                    self._condition.wait()
                    continue
                pending = list(self._abandoned)
            for item in pending:
                poll, windows = item
                try:
                    result = poll()
                except Exception:
                    continue
                if result is not None:
                    for window in windows:
                        self.pool.release(window)
                    with self._condition:
                        self._abandoned.remove(item)
                        self._inflight -= 1
                        self._condition.notify_all()
            with self._condition:
                if self._abandoned:
                    self._condition.wait(0.01)
