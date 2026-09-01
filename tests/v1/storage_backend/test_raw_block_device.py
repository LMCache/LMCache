# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import ctypes
import os
import platform

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    is_skip_safe_io_error,
    make_raw_block_file,
)

lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
RawBlockDevice = lmcache_rust_raw_block_io.RawBlockDevice


def _buffer_address(buffer: bytearray | memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _make_aligned_buffer(size: int, alignment: int) -> tuple[bytearray, memoryview]:
    backing = bytearray(size + alignment - 1)
    offset = (-_buffer_address(backing)) % alignment
    return backing, memoryview(backing)[offset : offset + size]


def test_raw_block_device_posix_roundtrip_on_tmp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(
        str(path),
        writable=True,
        use_odirect=False,
        alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
        io_engine="posix",
        iouring_queue_depth=8,
    )

    try:
        assert dev.size_bytes() == RAW_BLOCK_CI_CAPACITY_BYTES

        payload1 = bytearray(b"raw-block-posix-ci-payload")
        payload2 = bytearray(bytes(range(64)))
        out1 = bytearray(len(payload1))
        out2 = bytearray(len(payload2))

        dev.pwrite_from_buffer(4096, payload1, len(payload1), len(payload1))
        dev.pwrite_from_buffer(8192, payload2, len(payload2), len(payload2))

        dev.pread_into(4096, out1, len(out1), len(out1))
        dev.pread_into(8192, out2, len(out2), len(out2))

        assert out1 == payload1
        assert out2 == payload2
    finally:
        dev.close()


def test_raw_block_device_read_past_capacity_raises(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(
        str(path),
        writable=True,
        use_odirect=False,
        alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
        io_engine="posix",
        iouring_queue_depth=8,
    )

    try:
        out = bytearray(1)
        with pytest.raises(RuntimeError, match="unexpected EOF"):
            dev.pread_into(RAW_BLOCK_CI_CAPACITY_BYTES, out, len(out), len(out))
    finally:
        dev.close()


@pytest.mark.skipif(platform.system() != "Linux", reason="io_uring is Linux only")
def test_raw_block_device_iouring_best_effort_roundtrip(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=False,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )

        payload = bytearray(b"raw-block-iouring-ci-payload")
        out = bytearray(len(payload))

        batch_id = dev.batched_write([4096], [payload], [len(payload)])
        assert dev.wait_iouring(batch_id) == ([True], [])
        batch_id = dev.batched_read([4096], [out], [len(out)])
        assert dev.wait_iouring(batch_id) == ([True], [])

        assert out == payload
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


@pytest.mark.skipif(platform.system() != "Linux", reason="io_uring is Linux only")
def test_raw_block_device_iouring_fixed_buffer_subranges(tmp_path: Path) -> None:
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=False,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )

        region_size = 2 * RAW_BLOCK_CI_BLOCK_ALIGN
        _backing, registered = _make_aligned_buffer(
            2 * region_size,
            RAW_BLOCK_CI_BLOCK_ALIGN,
        )
        base_ptr = _buffer_address(registered)

        with pytest.raises(ValueError, match="null pointer"):
            dev.register_fixed_buffers([0], [1])
        with pytest.raises(ValueError, match="zero size"):
            dev.register_fixed_buffers([base_ptr], [0])
        with pytest.raises(ValueError, match="1 GiB"):
            dev.register_fixed_buffers([base_ptr], [(1 << 30) + 1])
        with pytest.raises(ValueError, match="overlap"):
            dev.register_fixed_buffers(
                [base_ptr, base_ptr + RAW_BLOCK_CI_BLOCK_ALIGN],
                [region_size, region_size],
            )

        dev.register_fixed_buffers(
            [base_ptr, base_ptr + region_size],
            [region_size, region_size],
        )
        with pytest.raises(RuntimeError, match="already registered"):
            dev.register_fixed_buffers([base_ptr], [region_size])

        scalar_payload = b"fixed-buffer scalar interior subrange"
        scalar_view = registered[128 : 128 + len(scalar_payload)]
        scalar_view[:] = scalar_payload
        dev.write_uring(4096, scalar_view, len(scalar_view), len(scalar_view))
        scalar_view[:] = bytes(len(scalar_view))
        dev.read_uring(4096, scalar_view, len(scalar_view), len(scalar_view))
        assert bytes(scalar_view) == scalar_payload

        interior_payload = b"fixed-buffer batched interior subrange"
        interior_start = region_size + 128
        interior_view = registered[
            interior_start : interior_start + len(interior_payload)
        ]
        interior_view[:] = interior_payload

        crossing_payload = bytes(range(64))
        crossing_start = region_size - len(crossing_payload) // 2
        crossing_view = registered[
            crossing_start : crossing_start + len(crossing_payload)
        ]
        crossing_view[:] = crossing_payload

        batch_id = dev.batched_write(
            [8192, 12288],
            [interior_view, crossing_view],
            [len(interior_view), len(crossing_view)],
        )
        assert dev.wait_iouring(batch_id) == ([True, True], [])

        interior_view[:] = bytes(len(interior_view))
        crossing_view[:] = bytes(len(crossing_view))
        batch_id = dev.batched_read(
            [8192, 12288],
            [interior_view, crossing_view],
            [len(interior_view), len(crossing_view)],
        )
        assert dev.wait_iouring(batch_id) == ([True, True], [])
        assert bytes(interior_view) == interior_payload
        assert bytes(crossing_view) == crossing_payload
    except Exception as e:
        message = str(e).lower()
        memlock_unavailable = (
            "register_buffers failed" in message and "cannot allocate memory" in message
        )
        if is_skip_safe_io_error(e) or memlock_unavailable:
            pytest.skip(f"io_uring fixed buffers are unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_ODIRECT_SMOKE") != "1",
    reason="O_DIRECT smoke is opt-in and not part of default PR CI",
)
def test_raw_block_device_odirect_optional_smoke(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="posix",
            iouring_queue_depth=8,
        )

        payload = bytearray([17]) * RAW_BLOCK_CI_BLOCK_ALIGN
        out = bytearray(len(payload))
        dev.pwrite_from_buffer(4096, payload, len(payload), len(payload))
        dev.pread_into(4096, out, len(out), len(out))
        assert out == payload
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"O_DIRECT is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()
