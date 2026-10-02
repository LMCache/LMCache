# SPDX-License-Identifier: Apache-2.0
"""Shared native-session ownership."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver


def test_concurrent_owners_open_once_and_close_on_last_release() -> None:
    driver = SharedDriver()
    owners = [object() for _ in range(16)]
    opened, closed = Mock(), Mock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda owner: driver.acquire(owner, opened), owners * 2))
        opened.assert_called_once()
        list(pool.map(lambda owner: driver.release(owner, closed), owners[:-1] * 2))
        closed.assert_not_called()
        list(pool.map(lambda _: driver.release(owners[-1], closed), range(8)))
        closed.assert_called_once()
    driver.acquire(owners[0], opened)
    assert opened.call_count == 2
    driver.release(owners[0], closed)
    assert closed.call_count == 2
