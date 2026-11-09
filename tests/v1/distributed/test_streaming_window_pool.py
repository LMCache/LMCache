# SPDX-License-Identifier: Apache-2.0
"""Public contracts of the bounded streaming window pool."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.distributed.storage_controllers.streaming_window_pool import (
    StreamingWindow,
    StreamingWindowPool,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator


def layout(size: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([size])], dtypes=[torch.uint8])


@pytest.fixture
def allocator(monkeypatch: pytest.MonkeyPatch) -> Iterator[L1MemoryManager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        memory_management, "_free_cpu_memory", lambda *args, **kwargs: None
    )
    manager = L1MemoryManager(
        L1MemoryManagerConfig(size_in_bytes=8192, align_bytes=4096, use_lazy=False)
    )
    yield manager
    manager.close()


class Fence:
    def __init__(self) -> None:
        self.event = Event()

    def is_complete(self) -> bool:
        return self.event.is_set()


def test_fence_controls_reuse_and_waiting_request_progress(
    allocator: L1MemoryManager,
) -> None:
    pool = StreamingWindowPool(allocator, 8192)
    first = pool.acquire(layout(4096))
    second = pool.acquire(layout(4096))
    assert pool.get_memory_usage() == (8192, 8192)
    fence = Fence()
    started = Event()

    def wait_for_window() -> StreamingWindow:
        started.set()
        return pool.acquire(layout(4096), timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait_for_window)
        assert started.wait(2)
        assert not pool.release(first, fence)
        assert not future.done()
        fence.event.set()
        assert pool.release(first, fence)
        third = future.result(timeout=5)
        assert pool.get_memory_usage() == (8192, 8192)
        assert pool.release(second)
        assert pool.release(third)
    assert pool.get_memory_usage() == (0, 8192)
    pool.close()


def test_timeout_and_oversized_window_do_not_leak(allocator: L1MemoryManager) -> None:
    pool = StreamingWindowPool(allocator, 8192)
    with pytest.raises(ValueError, match="fit"):
        pool.acquire(layout(8193))
    window = pool.acquire(layout(8192))
    with pytest.raises(LMCacheTimeoutError):
        pool.acquire(layout(4096), timeout=0)
    assert pool.release(window)
    assert pool.get_memory_usage() == (0, 8192)
    pool.close()


def test_close_rejects_live_window_and_releases_waiters(
    allocator: L1MemoryManager,
) -> None:
    pool = StreamingWindowPool(allocator, 8192)
    window = pool.acquire(layout(8192))
    with pytest.raises(RuntimeError, match="active windows"):
        pool.close()
    started = Event()

    def wait_for_window() -> StreamingWindow:
        started.set()
        return pool.acquire(layout(8192))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait_for_window)
        assert started.wait(2)
        pool.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=5)
        assert pool.release(window)
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(layout(4096))


def test_double_release_is_rejected(allocator: L1MemoryManager) -> None:
    pool = StreamingWindowPool(allocator, 8192)
    window = pool.acquire(layout(4096))
    assert pool.release(window)
    with pytest.raises(ValueError, match="not active"):
        pool.release(window)
    pool.close()
