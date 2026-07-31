# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter load-batch concurrency.

``_execute_load`` used to await each key's I/O before starting the next, so a
batch of N keys cost N sequential round trips. The caller
(``prefetch_controller``) puts a whole request's keys for an adapter into one
``submit_load_task``, so nothing upstream masked it.

These tests assert the observable contract: keys in a batch overlap, and the
overlap is bounded by ``io_concurrency``. They do not measure wall-clock speed --
that is timing-dependent and belongs in a benchmark -- but instead observe how
many loads are in flight at once, which is the property the fix is about.
"""

# Standard
from pathlib import Path
from typing import cast
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _DEFAULT_IO_CONCURRENCY,
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _Buf:
    """Minimal MemoryObj stand-in: just the ``byte_array`` the FS adapter uses."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([i, i, i, i]),
        model_name="llama",
        kv_rank=0,
        cache_salt="salt",
    )


def _bufs(payloads: list[bytes]) -> list[MemoryObj]:
    return cast("list[MemoryObj]", [_Buf(p) for p in payloads])


def _adapter(tmp_path: Path, **kwargs) -> FSL2Adapter:
    """Build an adapter. Callers close it in their own ``finally``."""
    return FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), **kwargs))


def _store_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], payloads: list[bytes]
) -> None:
    task_id = adp.submit_store_task(keys, _bufs(payloads))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if task_id in adp.pop_completed_store_tasks():
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 10s")


def _load_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], objs: list[MemoryObj]
) -> list[bool]:
    task_id = adp.submit_load_task(keys, objs)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        bitmap = adp.query_load_result(task_id)
        if bitmap is not None:
            return [bitmap.test(i) for i in range(len(keys))]
        time.sleep(0.01)
    raise AssertionError("load task did not complete within 10s")


class _ConcurrencyProbe:
    """Counts how many wrapped calls are in flight simultaneously."""

    def __init__(self, inner, hold_s: float = 0.02) -> None:
        self._inner = inner
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            # Hold long enough that genuinely overlapping calls are observed as
            # overlapping; a serial implementation can never exceed a peak of 1.
            time.sleep(self._hold_s)
            return self._inner(*args, **kwargs)
        finally:
            with self._lock:
                self.in_flight -= 1


NUM_KEYS = 8
PAYLOAD = b"x" * 4096


def test_load_batch_runs_keys_concurrently(tmp_path: Path) -> None:
    """A batch's keys overlap; a serial loop would peak at 1 in flight.

    This is the regression test for the defect: before the fix, ``_execute_load``
    awaited inside ``for i, key in enumerate(keys)``, so no two keys were ever
    in flight together regardless of batch size.
    """
    adp = _adapter(tmp_path, use_odirect=True, io_concurrency=NUM_KEYS)
    try:
        keys = [_key(i) for i in range(NUM_KEYS)]
        _store_and_wait(adp, keys, [PAYLOAD] * NUM_KEYS)

        probe = _ConcurrencyProbe(adp._read_with_odirect)
        adp._read_with_odirect = probe  # type: ignore[method-assign]

        results = _load_and_wait(adp, keys, _bufs([b"\x00" * len(PAYLOAD)] * NUM_KEYS))

        assert all(results), "every key should load"
        assert probe.calls == NUM_KEYS
        assert probe.peak > 1, (
            f"loads did not overlap (peak in-flight = {probe.peak}); "
            "the batch is being processed serially"
        )
    finally:
        adp.close()


def test_io_concurrency_one_is_serial(tmp_path: Path) -> None:
    """``io_concurrency=1`` reproduces the old one-key-at-a-time behaviour.

    Pins the lower bound of the knob and documents what the defect was: this is
    exactly the pre-fix code path, and it is ~2x slower than the default on
    24 MiB values (67 ms vs 32 ms for a 32-key batch, measured on NVMe RAID-0).
    """
    adp = _adapter(tmp_path, use_odirect=True, io_concurrency=1)
    try:
        keys = [_key(i) for i in range(NUM_KEYS)]
        _store_and_wait(adp, keys, [PAYLOAD] * NUM_KEYS)

        probe = _ConcurrencyProbe(adp._read_with_odirect)
        adp._read_with_odirect = probe  # type: ignore[method-assign]

        results = _load_and_wait(adp, keys, _bufs([b"\x00" * len(PAYLOAD)] * NUM_KEYS))

        assert all(results)
        assert probe.calls == NUM_KEYS
        assert probe.peak == 1, (
            f"io_concurrency=1 should serialise, saw peak {probe.peak}"
        )
    finally:
        adp.close()


def test_load_concurrency_is_bounded_by_config(tmp_path: Path) -> None:
    """Overlap never exceeds ``io_concurrency``.

    The bound matters because batch size is chosen by the caller -- a long
    context can be hundreds of keys -- so an unbounded fan-out would put an
    arbitrary number of large transfers in the executor at once.
    """
    limit = 2
    adp = _adapter(tmp_path, use_odirect=True, io_concurrency=limit)
    try:
        keys = [_key(i) for i in range(NUM_KEYS)]
        _store_and_wait(adp, keys, [PAYLOAD] * NUM_KEYS)

        probe = _ConcurrencyProbe(adp._read_with_odirect)
        adp._read_with_odirect = probe  # type: ignore[method-assign]

        results = _load_and_wait(adp, keys, _bufs([b"\x00" * len(PAYLOAD)] * NUM_KEYS))

        assert all(results)
        assert probe.calls == NUM_KEYS
        assert probe.peak <= limit, (
            f"peak in-flight {probe.peak} exceeded io_concurrency={limit}"
        )
        assert probe.peak > 1, "with a limit of 2, some overlap is still expected"
    finally:
        adp.close()


def test_load_data_is_correct_when_concurrent(tmp_path: Path) -> None:
    """Concurrency must not cross-wire keys to buffers.

    Each key gets a distinct payload, so a bug that mismatched the enumerate
    index against the objects list would surface as wrong bytes rather than a
    failed load.
    """
    adp = _adapter(tmp_path, io_concurrency=NUM_KEYS)
    try:
        keys = [_key(i) for i in range(NUM_KEYS)]
        payloads = [bytes([i]) * 4096 for i in range(NUM_KEYS)]
        _store_and_wait(adp, keys, payloads)

        objs = _bufs([b"\x00" * 4096] * NUM_KEYS)
        results = _load_and_wait(adp, keys, objs)

        assert all(results)
        for i, obj in enumerate(objs):
            assert bytes(obj.byte_array) == payloads[i], f"key {i} got wrong bytes"
    finally:
        adp.close()


def test_partial_batch_reports_per_key_results(tmp_path: Path) -> None:
    """Missing keys come back as 0 in the bitmap without failing the batch.

    Worth pinning down alongside the concurrency change: with keys now loading
    out of order, a missing key must not affect its neighbours' results.
    """
    adp = _adapter(tmp_path, io_concurrency=NUM_KEYS)
    try:
        stored = [_key(i) for i in range(0, NUM_KEYS, 2)]
        _store_and_wait(adp, stored, [PAYLOAD] * len(stored))

        keys = [_key(i) for i in range(NUM_KEYS)]
        results = _load_and_wait(adp, keys, _bufs([b"\x00" * len(PAYLOAD)] * NUM_KEYS))

        for i, hit in enumerate(results):
            assert hit == (i % 2 == 0), f"key {i}: expected hit={i % 2 == 0}"
    finally:
        adp.close()


@pytest.mark.parametrize("bad", [0, -1, "8", 1.5])
def test_invalid_io_concurrency_is_rejected(bad) -> None:
    """A non-positive or non-integer bound is a config error, not a silent default."""
    with pytest.raises(ValueError, match="io_concurrency"):
        FSL2AdapterConfig.from_dict(
            {"base_path": "/tmp/does-not-need-to-exist", "io_concurrency": bad}
        )


def test_default_io_concurrency_is_applied() -> None:
    """Omitting the key yields the documented default rather than unbounded."""
    cfg = FSL2AdapterConfig.from_dict({"base_path": "/tmp/does-not-need-to-exist"})
    assert cfg.io_concurrency == _DEFAULT_IO_CONCURRENCY
