# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ filesystem connector."""

# Standard
from pathlib import Path
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


def _expected_filename(key: str) -> str:
    model, kv_rank, object_group_id, chunk_hash = key.split("@")
    safe_model = model.replace("/", "-SEP-")
    return f"{safe_model}@0x{kv_rank}@{object_group_id}@{chunk_hash}.data"


def _expected_path(base_path: Path, key: str, hash_subdir_levels: int) -> Path:
    chunk_hash = key.split("@")[3]
    path = base_path
    for level in range(hash_subdir_levels):
        path /= chunk_hash[level * 2 : level * 2 + 2]
    return path / _expected_filename(key)


@pytest.mark.parametrize("hash_subdir_levels", [0, 1, 2])
def test_hash_subdir_layout_round_trip_and_delete(
    tmp_path: Path,
    hash_subdir_levels: int,
) -> None:
    """All operations should use the configured chunk-hash path layout."""
    chunk_hash = "a1b2c3d4e5f60718"
    key = f"test_model@00000000@7@{chunk_hash}"
    LMCacheFSClient = _import_fs_client()
    client = LMCacheFSClient(
        str(tmp_path),
        1,
        hash_subdir_levels=hash_subdir_levels,
    )
    source_data = bytearray(b"hash-subdirectory-test-data")
    source = memoryview(source_data)
    destination_data = bytearray(len(source_data))
    destination = memoryview(destination_data)
    expected_path = _expected_path(tmp_path, key, hash_subdir_levels)

    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        assert store[1], store[2]
        assert expected_path.is_file()

        future_id = client.submit_batch_exists([key])
        exists = _wait_for_completion(client, future_id)
        assert exists[1], exists[2]
        assert exists[3] == [True]

        load = _submit_and_wait(client, "submit_batch_get", key, destination)
        assert load[1], load[2]
        assert load[3] == [True]
        assert destination_data == source_data

        future_id = client.submit_batch_delete([key])
        delete = _wait_for_completion(client, future_id)
        assert delete[1], delete[2]
        assert delete[3] == [True]
        assert not expected_path.exists()

        future_id = client.submit_batch_exists([key])
        exists = _wait_for_completion(client, future_id)
        assert exists[3] == [False]
    finally:
        client.close()


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
