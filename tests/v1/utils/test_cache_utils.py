# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, repeat
from unittest.mock import patch
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.utils.cache_utils import TTLListCache


@pytest.mark.parametrize(
    ("refresh_completed_at", "reader_timeout", "expected"),
    [(120.0, None, [1]), (200.0, 5.0, [2])],
)
def test_waiting_reader_rechecks_expiration_after_lock(
    refresh_completed_at: float, reader_timeout: float | None, expected: list[int]
) -> None:
    """A waiting reader reuses a live snapshot or refreshes it with a full TTL."""
    cache = TTLListCache[int](timeout_seconds=30.0)
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    reader_sampled_time = threading.Event()
    now = 100.0

    def read_time() -> float:
        sampled_time = now
        if threading.current_thread().name.startswith("cache-reader"):
            reader_sampled_time.set()
        return sampled_time

    def slow_refresh() -> list[int]:
        snapshot = [1]
        refresh_started.set()
        assert release_refresh.wait(timeout=5.0), "refresh was not released"
        return snapshot

    with (
        patch("lmcache.v1.utils.cache_utils.time.time", side_effect=read_time),
        ThreadPoolExecutor(max_workers=1) as refresher,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-reader") as reader,
    ):
        first_result = refresher.submit(cache.get_cached, slow_refresh)
        try:
            assert refresh_started.wait(timeout=5.0)
            now = 110.0
            # The short override expires the first snapshot even if its refresh
            # finishes before the reader resumes after sampling the clock.
            second_result = reader.submit(
                cache.get_cached, lambda: [2], timeout_override=reader_timeout
            )
            assert reader_sampled_time.wait(timeout=5.0)

            now = refresh_completed_at
            release_refresh.set()
            assert first_result.result(timeout=5.0) == [1]
            assert second_result.result(timeout=5.0) == expected
            # A refresh's TTL starts when it can fetch, not before the lock wait.
            assert cache.get_cached(lambda: [3]) == expected
        finally:
            release_refresh.set()


def test_backward_clock_adjustment_does_not_revive_expired_cache() -> None:
    """A backward clock sample must not undo an observed expiration."""
    cache = TTLListCache[int](timeout_seconds=10.0)
    with patch("lmcache.v1.utils.cache_utils.time.time", return_value=80.0):
        assert cache.get_cached(lambda: [1]) == [1]

    with patch(
        "lmcache.v1.utils.cache_utils.time.time",
        side_effect=chain([100.0], repeat(50.0)),
    ):
        assert cache.get_cached(lambda: [2]) == [2]
