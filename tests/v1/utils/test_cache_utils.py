# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.utils.cache_utils import TTLListCache


@pytest.mark.parametrize(
    ("refresh_completed_at", "expected"),
    [(120.0, [1]), (200.0, [2])],
)
def test_waiting_reader_rechecks_expiration_after_lock(
    refresh_completed_at: float, expected: list[int]
) -> None:
    """Time spent waiting for a refresh must count toward the cache's TTL."""
    cache = TTLListCache[int](timeout_seconds=30.0)
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    reader_checked_expiration = threading.Event()
    now = 100.0
    is_expired = cache.is_expired

    def check_expiration(
        timeout_override: float | None = None, current_time: float | None = None
    ) -> bool:
        expired = is_expired(timeout_override, current_time)
        if threading.current_thread().name.startswith("cache-reader"):
            reader_checked_expiration.set()
        return expired

    def slow_refresh() -> list[int]:
        snapshot = [1]
        refresh_started.set()
        assert release_refresh.wait(timeout=5.0), "refresh was not released"
        return snapshot

    with (
        patch("lmcache.v1.utils.cache_utils.time.time", side_effect=lambda: now),
        patch.object(cache, "is_expired", side_effect=check_expiration),
        ThreadPoolExecutor(max_workers=1) as refresher,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-reader") as reader,
    ):
        first_result = refresher.submit(cache.get_cached, slow_refresh)
        try:
            assert refresh_started.wait(timeout=5.0)
            now = 110.0
            second_result = reader.submit(cache.get_cached, lambda: [2])
            assert reader_checked_expiration.wait(timeout=5.0)

            now = refresh_completed_at
            release_refresh.set()
            assert first_result.result(timeout=5.0) == [1]
            assert second_result.result(timeout=5.0) == expected
        finally:
            release_refresh.set()
