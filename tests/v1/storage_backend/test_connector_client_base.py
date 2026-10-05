# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ConnectorClientBase completion decoding.

Uses a pure-Python fake that mimics the pybind-wrapped C++ connector, so no
Redis/Aerospike server or native build is needed. The focus is the per-key
result mask: a batch GET where some keys miss completes with ``ok=True``, so
the mask is the only thing separating filled buffers from untouched ones.
"""

# Standard
from typing import Callable
import asyncio
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.native_clients.connector_client_base import (
    ConnectorClientBase,
)

# Local
from ..utils import close_asyncio_loop, init_asyncio_loop

MISS_SENTINEL = b"\xee\xee\xee\xee\xee\xee\xee\xee"


class FakeNativeConnector:
    """In-memory stand-in for the pybind C++ connector interface.

    Mirrors ``ConnectorBase``: a batch GET always reports one result byte per
    key (1 = buffer filled, 0 = key missing) and only fails the whole batch on
    a transport-level error.

    Args:
        get_mask: Rewrites the mask reported for batch GET; returning None
            simulates a connector that reports no per-key results at all.
        fail_get_with: When non-empty, batch GET completes with ``ok=False``
            and this error string.
    """

    def __init__(
        self,
        get_mask: Callable[[list[bool]], "list[bool] | None"] = lambda mask: mask,
        fail_get_with: str = "",
    ) -> None:
        self._notifier = create_event_notifier()
        self._store: dict[str, bytes] = {}
        self._next_id = 1
        self._completions: list[tuple[int, bool, str, "list[bool] | None"]] = []
        self._lock = threading.Lock()
        self._get_mask = get_mask
        self._fail_get_with = fail_get_with

    def event_fd(self) -> int:
        return self._notifier.fileno()

    def submit_batch_set(self, keys: list[str], bufs: list) -> int:
        fid = self._new_id()
        for key, buf in zip(keys, bufs, strict=True):
            self._store[key] = bytes(buf.cast("B"))
        self._push(fid, True, "", None)
        return fid

    def submit_batch_get(self, keys: list[str], bufs: list) -> int:
        fid = self._new_id()
        if self._fail_get_with:
            self._push(fid, False, self._fail_get_with, None)
            return fid
        mask: list[bool] = []
        for key, buf in zip(keys, bufs, strict=True):
            data = self._store.get(key)
            if data is None:
                mask.append(False)
                continue
            buf.cast("B")[: len(data)] = data
            mask.append(True)
        self._push(fid, True, "", self._get_mask(mask))
        return fid

    def submit_batch_exists(self, keys: list[str]) -> int:
        fid = self._new_id()
        self._push(fid, True, "", [key in self._store for key in keys])
        return fid

    def drain_completions(self) -> list[tuple[int, bool, str, "list[bool] | None"]]:
        try:
            self._notifier.consume()
        except BlockingIOError:
            pass
        with self._lock:
            completions = list(self._completions)
            self._completions.clear()
        return completions

    def close(self) -> None:
        self._notifier.close()

    def _new_id(self) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1
        return fid

    def _push(self, fid: int, ok: bool, error: str, mask: "list[bool] | None") -> None:
        with self._lock:
            self._completions.append((fid, ok, error, mask))
        try:
            self._notifier.notify()
        except OSError:
            pass


def _bufs(count: int) -> list[memoryview]:
    return [memoryview(bytearray(MISS_SENTINEL)) for _ in range(count)]


@pytest.mark.asyncio
async def test_batch_get_reports_per_key_hits_and_misses():
    client = ConnectorClientBase(FakeNativeConnector())
    try:
        payload = b"payload!"
        await client.batch_set(["hit"], [memoryview(bytearray(payload))])

        bufs = _bufs(3)
        hits = await client.batch_get(["miss_a", "hit", "miss_b"], bufs)

        assert hits == [False, True, False]
        # The hit in a partial batch is still the right data ...
        assert bytes(bufs[1]) == payload
        # ... and the misses left their staging buffers untouched.
        assert bytes(bufs[0]) == MISS_SENTINEL
        assert bytes(bufs[2]) == MISS_SENTINEL
    finally:
        client.close()


@pytest.mark.asyncio
async def test_get_returns_hit_flag():
    client = ConnectorClientBase(FakeNativeConnector())
    try:
        payload = b"payload!"
        await client.set("k", memoryview(bytearray(payload)))

        buf = memoryview(bytearray(MISS_SENTINEL))
        assert await client.get("k", buf) is True
        assert bytes(buf) == payload

        miss_buf = memoryview(bytearray(MISS_SENTINEL))
        assert await client.get("absent", miss_buf) is False
        assert bytes(miss_buf) == MISS_SENTINEL
    finally:
        client.close()


@pytest.mark.asyncio
async def test_batch_get_rejects_completion_without_per_key_results():
    client = ConnectorClientBase(FakeNativeConnector(get_mask=lambda mask: None))
    try:
        with pytest.raises(RuntimeError, match="per-key results"):
            await client.batch_get(["a", "b"], _bufs(2))
    finally:
        client.close()


@pytest.mark.asyncio
async def test_batch_get_propagates_batch_failure():
    client = ConnectorClientBase(FakeNativeConnector(fail_get_with="connection reset"))
    try:
        with pytest.raises(RuntimeError, match="connection reset"):
            await client.batch_get(["a"], _bufs(1))
    finally:
        client.close()


@pytest.mark.asyncio
async def test_malformed_completion_does_not_poison_other_futures():
    """A bad completion must fail its own future only, not in-flight siblings."""
    client = ConnectorClientBase(FakeNativeConnector(get_mask=lambda mask: mask[:1]))
    try:
        bad, good = await asyncio.gather(
            client.batch_get(["a", "b"], _bufs(2)),
            client.batch_exists(["a", "b"]),
            return_exceptions=True,
        )
        assert isinstance(bad, RuntimeError)
        assert "per-key results" in str(bad)
        assert good == [False, False]
    finally:
        client.close()


def test_batch_get_sync_reports_per_key_hits_and_misses():
    loop, thread = init_asyncio_loop()
    try:
        client = ConnectorClientBase(FakeNativeConnector(), loop)
        payload = b"payload!"
        client.batch_set_sync(["hit"], [memoryview(bytearray(payload))])

        bufs = _bufs(2)
        hits = client.batch_get_sync(["hit", "miss"], bufs)

        assert hits == [True, False]
        assert bytes(bufs[0]) == payload
        assert bytes(bufs[1]) == MISS_SENTINEL
        client.close()
    finally:
        close_asyncio_loop(loop, thread)
