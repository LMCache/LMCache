# SPDX-License-Identifier: Apache-2.0
"""Public contracts of the bounded streaming window pool."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Event

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.storage_controllers.streaming_window_pool import (
    StreamingWindow,
    StreamingWindowPool,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

pytestmark = pytest.mark.no_shared_allocator


def layout(size: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([size])], dtypes=[torch.uint8])


@pytest.fixture
def arena() -> torch.Tensor:
    return torch.empty(8192, dtype=torch.uint8)


def test_completed_copy_releases_window_to_waiter(
    arena: torch.Tensor,
) -> None:
    pool = StreamingWindowPool(arena)
    first = pool.acquire(layout(4096))
    second = pool.acquire(layout(4096))
    assert pool.get_memory_usage() == (8192, 8192)
    started = Event()

    def wait_for_window() -> StreamingWindow:
        started.set()
        return pool.acquire(layout(4096), timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait_for_window)
        assert started.wait(2)
        assert not future.done()
        pool.release_completed([first.id])
        third = future.result(timeout=5)
        assert pool.get_memory_usage() == (8192, 8192)
        pool.release(second)
        pool.release(third)
    assert pool.get_memory_usage() == (0, 8192)
    pool.close()


def test_timeout_and_oversized_window_do_not_leak(arena: torch.Tensor) -> None:
    pool = StreamingWindowPool(arena)
    with pytest.raises(ValueError, match="fit"):
        pool.acquire(layout(8193))
    window = pool.acquire(layout(8192))
    with pytest.raises(LMCacheTimeoutError):
        pool.acquire(layout(4096), timeout=0)
    pool.release(window)
    assert pool.get_memory_usage() == (0, 8192)
    pool.close()


def test_close_rejects_live_window_and_releases_waiters(
    arena: torch.Tensor,
) -> None:
    pool = StreamingWindowPool(arena)
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
        pool.release(window)
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(layout(4096))


def test_double_release_is_rejected(arena: torch.Tensor) -> None:
    pool = StreamingWindowPool(arena)
    window = pool.acquire(layout(4096))
    pool.release(window)
    with pytest.raises(ValueError, match="not active"):
        pool.release(window)
    pool.close()
