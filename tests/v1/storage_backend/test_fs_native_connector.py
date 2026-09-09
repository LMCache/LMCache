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

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)


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


@pytest.mark.parametrize(
    ("wire_key", "filename"),
    [
        ("model@00000000@aabb", "model@0x00000000@aabb.data"),
        (
            "model@00000000@aabb@tenant",
            "model@0x00000000@aabb@tenant.data",
        ),
    ],
)
def test_legacy_key_shapes_keep_their_filenames(
    tmp_path, wire_key: str, filename: str
) -> None:
    """Three/four-field keys written by older clients remain readable."""
    LMCacheFSClient = _import_fs_client()
    payload = bytearray(b"legacy-payload")
    client = LMCacheFSClient(str(tmp_path), 1)
    try:
        completion = _submit_and_wait(
            client, "submit_batch_set", wire_key, memoryview(payload)
        )
        assert completion[1], completion[2]
        assert (tmp_path / filename).read_bytes() == payload
    finally:
        client.close()


@pytest.mark.parametrize("cache_salt", ["", "tenant"])
def test_current_object_group_keys_match_python_filename(
    tmp_path, cache_salt: str
) -> None:
    """Native FS and Python FS use one path for current ObjectKeys."""
    LMCacheFSClient = _import_fs_client()
    key = ObjectKey(
        chunk_hash=bytes.fromhex("aabbccdd"),
        model_name="org/model",
        kv_rank=0x01020304,
        object_group_id=0x2A,
        cache_salt=cache_salt,
    )
    payload = bytearray(b"object-group-payload")
    client = LMCacheFSClient(str(tmp_path), 1)
    try:
        completion = _submit_and_wait(
            client,
            "submit_batch_set",
            _object_key_to_string(key),
            memoryview(payload),
        )
        assert completion[1], completion[2]
        expected_path = tmp_path / _object_key_to_filename(key)
        assert expected_path.read_bytes() == payload
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
