# SPDX-License-Identifier: Apache-2.0

"""Tests for ChunkBlockRegistry (thread-safe chunk_hash -> block_hashes map)."""

# Standard
import threading

# First Party
from lmcache.v1.mp_observability.dynamo.chunk_block_registry import (
    ChunkBlockRegistry,
)


def test_record_then_pop_then_pop_again() -> None:
    reg = ChunkBlockRegistry()
    reg.record(b"chunk-a", [1, 2, 3])

    assert reg.pop(b"chunk-a") == [1, 2, 3]
    # Already removed -> second pop returns None.
    assert reg.pop(b"chunk-a") is None


def test_pop_missing_key_returns_none() -> None:
    reg = ChunkBlockRegistry()
    assert reg.pop(b"never-recorded") is None


def test_record_overwrites() -> None:
    reg = ChunkBlockRegistry()
    reg.record(b"chunk-a", [1, 2, 3])
    reg.record(b"chunk-a", [9, 9])

    assert reg.pop(b"chunk-a") == [9, 9]


def test_len_tracks_records_and_pops() -> None:
    reg = ChunkBlockRegistry()
    assert len(reg) == 0

    reg.record(b"a", [1])
    reg.record(b"b", [2])
    assert len(reg) == 2

    # Overwrite does not change the count.
    reg.record(b"a", [11])
    assert len(reg) == 2

    reg.pop(b"a")
    assert len(reg) == 1
    # Popping a missing key leaves the count untouched.
    reg.pop(b"missing")
    assert len(reg) == 1


def test_concurrent_record_and_pop_smoke() -> None:
    reg = ChunkBlockRegistry()
    num_threads = 8
    per_thread = 500
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(per_thread):
                key = f"t{tid}-{i}".encode()
                reg.record(key, [tid, i])
                # Half the time pop it back out immediately.
                if i % 2 == 0:
                    reg.pop(key)
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Each thread leaves exactly the odd-indexed keys behind.
    expected_remaining = num_threads * (per_thread // 2)
    assert len(reg) == expected_remaining
