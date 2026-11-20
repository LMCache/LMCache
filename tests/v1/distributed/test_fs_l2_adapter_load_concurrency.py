# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter concurrent loads (``load_concurrency``).

Loads used to be strictly sequential: one awaited file at a time. On a
network-backed volume that is one round trip per chunk file, slower than
recomputing the KV cache. These tests pin the concurrent path: every key
lands with intact data, misses stay per-key (buffers untouched, accessed
notification in key order), the read-ahead branch still works under the
gate, the adapter-wide bound is reached and not exceeded, and the config
knob is validated and documented.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
import asyncio
import time

# Third Party
import aiofiles
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _RecordingListener:
    """Records ``on_l2_keys_accessed`` (duck-typed L2AdapterListener)."""

    def __init__(self) -> None:
        self.accessed: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        pass

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.extend(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        pass


class _Buf:
    """Minimal MemoryObj stand-in: just the ``byte_array`` the FS
    adapter's store/load paths read and write."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


AdapterFixture = tuple[FSL2Adapter, _RecordingListener]


def _indexed_key(i: int) -> ObjectKey:
    """A distinct key per index (the chunk hash encodes ``i``)."""
    return ObjectKey(
        chunk_hash=i.to_bytes(4, "big"),
        model_name="llama",
        kv_rank=0,
        cache_salt="",
    )


def _payloads(n: int, base: int = 1024) -> list[bytes]:
    """Distinct content AND length per index, so a cross-wired buffer is
    caught by either."""
    return [bytes([i % 251]) * (base + i) for i in range(n)]


def make_adapter(tmp_path: Path, **kwargs: Any) -> AdapterFixture:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), **kwargs))
    listener = _RecordingListener()
    adp.register_listener(listener)  # type: ignore[arg-type]
    return adp, listener


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[AdapterFixture]:
    adp, listener = make_adapter(tmp_path, load_concurrency=4)
    try:
        yield adp, listener
    finally:
        adp.close()


def _bufs(payloads: list[bytes]) -> list[MemoryObj]:
    """Wrap raw payloads as MemoryObj-shaped buffers (see ``_Buf``)."""
    return cast("list[MemoryObj]", [_Buf(p) for p in payloads])


def _store_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], payloads: list[bytes]
) -> None:
    """Submit a store and poll until its result is available."""
    task_id = adp.submit_store_task(keys, _bufs(payloads))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            # L2StoreResult is an int: >= 0 success, -1 failure.
            assert int(completed[task_id]) >= 0
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


def _load_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], bufs: list[MemoryObj]
) -> list[bool]:
    """Submit a load and poll until its hit bitmap is available."""
    task_id = adp.submit_load_task(keys, bufs)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        bitmap = adp.query_load_result(task_id)
        if bitmap is not None:
            return [bitmap.test(i) for i in range(len(keys))]
        time.sleep(0.01)
    raise AssertionError("load task did not complete within 5s")


class TestConcurrentLoad:
    """``submit_load_task`` loads keys concurrently under ``load_concurrency``
    with unchanged per-key semantics."""

    def test_every_key_lands_with_intact_data(self, adapter: AdapterFixture) -> None:
        adp, _ = adapter
        n = 40
        keys = [_indexed_key(i) for i in range(n)]
        payloads = _payloads(n)
        _store_and_wait(adp, keys, payloads)

        bufs = _bufs([b"\x00" * len(p) for p in payloads])
        hits = _load_and_wait(adp, keys, bufs)

        assert hits == [True] * n
        for buf, payload in zip(bufs, payloads, strict=True):
            assert bytes(buf.byte_array) == payload

    def test_missing_keys_are_per_key_misses_and_untouched(
        self, adapter: AdapterFixture
    ) -> None:
        adp, listener = adapter
        keys = [_indexed_key(i) for i in range(6)]
        payloads = _payloads(6, base=64)
        _store_and_wait(adp, keys[::2], payloads[::2])

        bufs = _bufs([b"\x00" * len(p) for p in payloads])
        hits = _load_and_wait(adp, keys, bufs)

        assert hits == [True, False, True, False, True, False]
        for i, (buf, payload) in enumerate(zip(bufs, payloads, strict=True)):
            expected = payload if i % 2 == 0 else b"\x00" * len(payload)
            assert bytes(buf.byte_array) == expected
        assert listener.accessed == [keys[0], keys[2], keys[4]]

    def test_read_ahead_path_survives_concurrency(self, tmp_path: Path) -> None:
        adp, _ = make_adapter(tmp_path, load_concurrency=3, read_ahead_size=128)
        try:
            n = 8
            keys = [_indexed_key(i) for i in range(n)]
            payloads = _payloads(n, base=512)
            _store_and_wait(adp, keys, payloads)

            bufs = _bufs([b"\x00" * len(p) for p in payloads])
            hits = _load_and_wait(adp, keys, bufs)

            assert hits == [True] * n
            for buf, payload in zip(bufs, payloads, strict=True):
                assert bytes(buf.byte_array) == payload
        finally:
            adp.close()

    def test_bound_is_reached_and_not_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adapter-wide gate admits exactly ``load_concurrency`` files at
        once. Sequential code would peak at 1, so the equality also proves
        the loads really run concurrently."""
        bound = 3
        adp, _ = make_adapter(tmp_path, load_concurrency=bound)
        try:
            keys = [_indexed_key(i) for i in range(12)]
            payloads = _payloads(12, base=256)
            _store_and_wait(adp, keys, payloads)

            in_flight = {"now": 0, "peak": 0}
            real_open = aiofiles.open

            class _Counted:
                def __init__(self, cm: Any) -> None:
                    self._cm = cm

                async def __aenter__(self) -> Any:
                    in_flight["now"] += 1
                    in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
                    await asyncio.sleep(0.01)
                    return await self._cm.__aenter__()

                async def __aexit__(self, *args: Any) -> Any:
                    in_flight["now"] -= 1
                    return await self._cm.__aexit__(*args)

            monkeypatch.setattr(
                aiofiles, "open", lambda *a, **k: _Counted(real_open(*a, **k))
            )

            hits = _load_and_wait(
                adp, keys, _bufs([b"\x00" * len(p) for p in payloads])
            )

            assert hits == [True] * 12
            assert in_flight["peak"] == bound
        finally:
            adp.close()


class TestLoadConcurrencyConfig:
    """``load_concurrency`` is parsed, validated, defaulted and documented."""

    def test_parses_from_dict(self) -> None:
        cfg = FSL2AdapterConfig.from_dict(
            {"base_path": "/tmp/x", "load_concurrency": 8}
        )
        assert cfg.load_concurrency == 8

    def test_default_reaches_the_adapter(self, tmp_path: Path) -> None:
        adp = FSL2Adapter(FSL2AdapterConfig.from_dict({"base_path": str(tmp_path)}))
        try:
            assert adp.report_status()["load_concurrency"] == 16
        finally:
            adp.close()

    @pytest.mark.parametrize("bad", [0, -4, "16", 2.5, True])
    def test_rejects_non_positive_non_int(self, bad: object) -> None:
        with pytest.raises(ValueError, match="load_concurrency"):
            FSL2AdapterConfig.from_dict(
                {"base_path": "/tmp/x", "load_concurrency": bad}
            )

    def test_help_documents_the_field(self) -> None:
        assert "load_concurrency" in FSL2AdapterConfig.help()
