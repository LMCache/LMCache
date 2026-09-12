# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ filesystem connector."""

# Standard
from typing import Any
import ctypes
import os
import select
import time

# Third Party
import pytest


def _import_fs_client() -> type:
    try:
        # First Party
        from lmcache.lmcache_fs import LMCacheFSClient
    except ImportError as exc:
        pytest.skip(f"native FS extension is not available: {exc}")
    return LMCacheFSClient


def _buffer_addr(buf: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _aligned_memoryview(
    size: int,
    alignment: int,
) -> tuple[bytearray, memoryview]:
    raw = bytearray(size + alignment)
    raw_addr = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    offset = (-raw_addr) % alignment
    view = memoryview(raw)[offset : offset + size]
    assert _buffer_addr(view) % alignment == 0
    return raw, view


def _misaligned_memoryview(
    size: int,
    alignment: int,
) -> tuple[bytearray, memoryview]:
    raw = bytearray(size + alignment + 1)
    for offset in range(1, alignment + 1):
        view = memoryview(raw)[offset : offset + size]
        if _buffer_addr(view) % alignment != 0:
            return raw, view
    raise AssertionError("failed to create misaligned memoryview")


def _fill(view: memoryview) -> None:
    view[:] = bytes(i % 251 for i in range(len(view)))


def _wait_for_completion(
    client: Any,
    future_id: int,
    timeout: float = 5.0,
) -> tuple[int, bool, str, list[bool] | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for completion in client.drain_completions():
            if completion[0] == future_id:
                return completion

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        select.select([client.event_fd()], [], [], min(remaining, 0.1))

    raise TimeoutError(f"timed out waiting for future {future_id}")


def _submit_and_wait(
    client: Any,
    method_name: str,
    key: str,
    view: memoryview,
) -> tuple[int, bool, str, list[bool] | None]:
    future_id = getattr(client, method_name)([key], [view])
    return _wait_for_completion(client, future_id)


def test_odirect_read_does_not_split_for_read_ahead(tmp_path) -> None:
    """O_DIRECT reads should ignore read_ahead_size and use one aligned read."""
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is not available on this platform")

    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    size = block_size * 2
    key = "test_model@00000000@0123456789abcdef"
    _source_raw, source = _aligned_memoryview(size, block_size)
    _dest_raw, dest = _aligned_memoryview(size, block_size)
    _fill(source)

    client = LMCacheFSClient(str(tmp_path), 1, "", True, 1)
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        if not store[1]:
            pytest.skip(f"O_DIRECT is unsupported on this filesystem: {store[2]}")

        load = _submit_and_wait(client, "submit_batch_get", key, dest)
        assert load[1], load[2]
        assert bytes(dest) == bytes(source)
    finally:
        client.close()


def test_odirect_fails_for_misaligned_buffer(tmp_path) -> None:
    """Misaligned buffers should fail instead of hiding O_DIRECT misuse."""
    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    size = block_size * 2
    key = "test_model@00000000@fedcba9876543210"
    _source_raw, source = _misaligned_memoryview(size, block_size)
    _fill(source)

    client = LMCacheFSClient(str(tmp_path), 1, "", True, 1)
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        assert not store[1]
        assert "O_DIRECT buffer address is not aligned" in store[2]
    finally:
        client.close()


def _write_corpus(
    client: Any,
    keys: list[str],
    views: list[memoryview],
) -> None:
    """Store every key with the connector, failing the test on error."""
    future_id = client.submit_batch_set(keys, views)
    completion = _wait_for_completion(client, future_id)
    assert completion[1], completion[2]


def _aligned_pairs(
    count: int,
    size: int,
    block_size: int,
) -> tuple[list[memoryview], list[memoryview], list[bytearray]]:
    """Build `count` source/destination pairs, each source filled distinctly.

    The third return value owns the backing allocations and must stay
    referenced for as long as the views are used.
    """
    sources: list[memoryview] = []
    dests: list[memoryview] = []
    keep: list[bytearray] = []
    for index in range(count):
        source_raw, source = _aligned_memoryview(size, block_size)
        dest_raw, dest = _aligned_memoryview(size, block_size)
        keep.extend((source_raw, dest_raw))
        source[:] = bytes((i + index) % 251 for i in range(size))
        sources.append(source)
        dests.append(dest)
    return sources, dests, keep


@pytest.mark.parametrize(
    "read_io_depth,read_max_bytes_in_flight",
    [
        (0, 0),  # legacy path: one blocking read per object on the worker
        (1, 0),  # a pool of one, so every read still queues behind the last
        (4, 0),  # several reads of one batch in flight at once
        (8, 1),  # budget below one object: the group must still take one
        (8, 4096),  # a few objects per group
        (8, 1 << 30),  # larger than the whole batch: one group
    ],
)
def test_reads_return_identical_bytes(
    tmp_path,
    read_io_depth: int,
    read_max_bytes_in_flight: int,
) -> None:
    """Neither the pool nor the byte budget may change the bytes returned.

    They decide who issues a read and how many are outstanding, nothing
    else, so the destination buffer must be indistinguishable from the
    legacy path's.  A budget below one object size is the interesting
    case: the group must still take that object, or a batch containing it
    could never complete.
    """
    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    # An odd number of blocks, so an object is never a round power of two.
    size = block_size * 7
    keys = [f"test_model@00000000@{i:016x}" for i in range(6)]
    sources, dests, _keep = _aligned_pairs(len(keys), size, block_size)

    writer = LMCacheFSClient(str(tmp_path), 2, "", True, 0)
    try:
        _write_corpus(writer, keys, sources)
    finally:
        writer.close()

    reader = LMCacheFSClient(
        str(tmp_path), 2, "", True, 0, read_io_depth, read_max_bytes_in_flight
    )
    try:
        if read_max_bytes_in_flight:
            assert reader.read_budget_bytes() == read_max_bytes_in_flight
        future_id = reader.submit_batch_get(keys, dests)
        completion = _wait_for_completion(reader, future_id)
        assert completion[1], completion[2]
        per_key = completion[3]
        assert per_key is not None
        assert list(per_key) == [True] * len(keys)
        for index, dest in enumerate(dests):
            assert bytes(dest) == bytes(sources[index]), f"object {index} differs"
    finally:
        reader.close()


def test_pooled_read_tolerates_one_missing_object(tmp_path) -> None:
    """A missing object fails alone; its batch-mates still load."""
    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    size = block_size * 4
    present = "test_model@00000000@aaaaaaaaaaaaaaaa"
    absent = "test_model@00000000@bbbbbbbbbbbbbbbb"
    _source_raw, source = _aligned_memoryview(size, block_size)
    _first_raw, first = _aligned_memoryview(size, block_size)
    _second_raw, second = _aligned_memoryview(size, block_size)
    # A third buffer, so no two keys of the batch share a destination:
    # concurrent reads into one buffer would be a data race even when the
    # bytes happen to match.
    _third_raw, third = _aligned_memoryview(size, block_size)
    _fill(source)

    writer = LMCacheFSClient(str(tmp_path), 1, "", True, 0)
    try:
        _write_corpus(writer, [present], [source])
    finally:
        writer.close()

    reader = LMCacheFSClient(str(tmp_path), 2, "", True, 0, 4)
    try:
        future_id = reader.submit_batch_get(
            [present, absent, present], [first, second, third]
        )
        completion = _wait_for_completion(reader, future_id)
        per_key = completion[3]
        assert per_key is not None
        assert list(per_key) == [True, False, True]
        assert bytes(first) == bytes(source)
        assert bytes(third) == bytes(source)
    finally:
        reader.close()


def test_the_default_budget_is_used_when_none_is_configured(tmp_path) -> None:
    """Zero selects the documented default rather than "no budget".

    Reads in flight is what sets throughput, and inheriting the base
    class's behaviour leaves it equal to num_workers objects, which is
    far below what an array needs.  A caller that turns on read_io_depth
    and says nothing about bytes should get a working figure.
    """
    LMCacheFSClient = _import_fs_client()
    reader = LMCacheFSClient(str(tmp_path), 2, "", False, 0, 8)
    try:
        assert reader.read_budget_bytes() == 1536 << 20
    finally:
        reader.close()

    # The legacy path has no budget at all.
    legacy = LMCacheFSClient(str(tmp_path), 2)
    try:
        assert legacy.read_budget_bytes() == 0
    finally:
        legacy.close()

    # A budget configured without read_io_depth throttles nothing, so it
    # must not be reported as if it did.
    unused = LMCacheFSClient(str(tmp_path), 2, "", False, 0, 0, 1 << 20)
    try:
        assert unused.read_budget_bytes() == 0
    finally:
        unused.close()


def test_negative_read_io_depth_is_rejected(tmp_path) -> None:
    """A negative depth is a configuration error, not a silent fallback."""
    LMCacheFSClient = _import_fs_client()
    with pytest.raises(RuntimeError, match="depth must be >= 0"):
        LMCacheFSClient(str(tmp_path), 1, "", False, 0, -1)
