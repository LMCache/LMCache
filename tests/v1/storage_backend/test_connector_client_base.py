# SPDX-License-Identifier: Apache-2.0
"""
ConnectorClientBase must not lose a completion that the event-loop thread drains
between submit_*() and the registration of the Python future (*_sync methods).

FakeNative completes every request on submit and waits up to 0.3 s for the loop
thread to drain it before returning the id, which forces that interleaving. Before
the fix the completion was dropped and fut.result() blocked forever.
"""

# Standard
import asyncio
import concurrent.futures
import itertools
import os
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.native_clients.connector_client_base import (
    ConnectorClientBase,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "eventfd"), reason="needs os.eventfd (Linux)"
)


class FakeNative:
    def __init__(self):
        self._fd = os.eventfd(0, os.EFD_NONBLOCK)
        self._ids = itertools.count(1)
        self._done = []
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)

    def event_fd(self):
        return self._fd

    def _submit(self, n):
        fid = next(self._ids)
        with self._lock:
            self._done.append((fid, True, "", [False] * n))
        os.eventfd_write(self._fd, 1)
        with (
            self._drained
        ):  # give the loop thread a chance to drain before we return the id
            self._drained.wait_for(
                lambda: all(d[0] != fid for d in self._done), timeout=0.3
            )
        return fid

    def submit_batch_exists(self, keys):
        return self._submit(len(keys))

    def submit_batch_get(self, keys, bufs):
        return self._submit(len(keys))

    def submit_batch_set(self, keys, bufs):
        return self._submit(len(keys))

    def drain_completions(self):
        try:
            os.eventfd_read(self._fd)
        except BlockingIOError:
            pass
        with self._drained:
            items, self._done = self._done, []
            self._drained.notify_all()
        return items

    def close(self):
        os.close(self._fd)


def _client():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    ready = concurrent.futures.Future()
    loop.call_soon_threadsafe(
        lambda: ready.set_result(ConnectorClientBase(FakeNative(), loop))
    )
    return ready.result(5)


def _call(fn, *a):
    ex = concurrent.futures.ThreadPoolExecutor(1)
    return ex.submit(fn, *a).result(timeout=3)  # TimeoutError = completion lost


def test_sync_calls_survive_forced_race():
    c = _client()
    for i in range(5):
        assert _call(c.batch_exists_sync, [f"k{i}", "x"]) == [False, False]
        assert _call(c.batch_get_sync, [f"k{i}"], [memoryview(bytearray(1))]) is None
        assert _call(c.batch_set_sync, [f"k{i}"], [memoryview(bytearray(1))]) is None


def test_async_path_unchanged():
    c = _client()
    fut = asyncio.run_coroutine_threadsafe(c.batch_exists(["a", "b", "c"]), c.loop)
    assert fut.result(5) == [False, False, False]
