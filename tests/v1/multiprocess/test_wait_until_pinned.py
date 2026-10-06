# SPDX-License-Identifier: Apache-2.0
"""Worker-side wait for the server's deferred host pinning."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
)


class _Future:
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def result(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._value


def _client(*statuses):
    """Request client whose pin_status() replies with ``statuses`` in order."""
    client = MagicMock()
    client.pin_status.side_effect = [_Future(s) for s in statuses]
    return client


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(worker_transfer.time, "sleep", lambda _s: None)


def test_returns_immediately_when_pool_is_pinned(no_sleep):
    client = _client((100, 100))
    LMCacheDrivenTransferContext(1, client)._wait_until_pinned(mq_timeout=1.0)
    assert client.pin_status.call_count == 1


def test_returns_immediately_without_deferred_pinning(no_sleep):
    client = _client((0, 0))
    LMCacheDrivenTransferContext(1, client)._wait_until_pinned(mq_timeout=1.0)
    assert client.pin_status.call_count == 1


def test_polls_until_pinned(no_sleep):
    client = _client((10, 100), (50, 100), (100, 100))
    LMCacheDrivenTransferContext(1, client)._wait_until_pinned(mq_timeout=1.0)
    assert client.pin_status.call_count == 3


def test_skips_wait_when_server_lacks_pin_status(no_sleep):
    client = MagicMock()
    client.pin_status.return_value = _Future(error=LMCacheTimeoutError("no reply"))
    LMCacheDrivenTransferContext(1, client)._wait_until_pinned(mq_timeout=1.0)
    assert client.pin_status.call_count == 1


def test_stops_polling_once_closed(no_sleep):
    client = _client((10, 100), (20, 100))
    ctx = LMCacheDrivenTransferContext(1, client)
    ctx._mark_closed()
    ctx._wait_until_pinned(mq_timeout=1.0)
    assert client.pin_status.call_count == 1
