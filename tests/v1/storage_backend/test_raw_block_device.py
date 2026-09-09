# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
import ctypes
import mmap
import os
import platform
from pathlib import Path

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


def test_raw_block_device_posix_stats_and_validation(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(str(path), writable=True, io_engine="posix")
    try:
        assert all(value == 0 for value in dev.io_stats_snapshot().values())
        with pytest.raises(ValueError):
            dev.pwrite_from_buffer(0, b"data", 5, 5)
        assert dev.io_stats_snapshot()["write_attempts"] == 0
        dev.pwrite_from_buffer(4096, b"data", 4, 4096)
        out = bytearray(4)
        dev.pread_into(4096, out, 4, 4096)
        assert out == b"data"
        snapshot = dev.io_stats_snapshot()
        assert snapshot["write_attempts"] == 1
        assert snapshot["read_attempts"] == 1
        assert snapshot["write_submitted_bytes"] == 4096
        assert snapshot["read_submitted_bytes"] == 4096
        assert snapshot["bounce_attempts"] == 2
        assert snapshot["bounce_submitted_bytes"] == 8192
        assert snapshot["completed_attempts"] == 2
        assert snapshot["failed_attempts"] == 0
        assert snapshot["outstanding_requests"] == 0
        assert snapshot["queued_requests"] == 0
        assert snapshot["peak_queued_requests"] == 0
        assert snapshot["peak_outstanding_requests"] >= 1
        assert snapshot["fixed_buffer_attempts"] == 0
        with pytest.raises(RuntimeError, match="unexpected EOF"):
            dev.pread_into(RAW_BLOCK_CI_CAPACITY_BYTES, out, 4, 4)
        snapshot = dev.io_stats_snapshot()
        assert snapshot["read_attempts"] == 2
        assert snapshot["failed_attempts"] == 1
        assert snapshot["outstanding_requests"] == 0
    finally:
        dev.close()
    assert dev.io_stats_snapshot() == snapshot


def test_raw_block_device_posix_stats_concurrent(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = RawBlockDevice(str(path), writable=True, io_engine="posix")
    try:

        def write(index: int) -> None:
            dev.pwrite_from_buffer(4096 * index, bytes(4096), 4096, 4096)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(64)))
        snapshot = dev.io_stats_snapshot()
        assert snapshot["write_attempts"] == 64
        assert snapshot["write_submitted_bytes"] == 64 * 4096
        assert snapshot["completed_attempts"] == 64
        assert snapshot["failed_attempts"] == 0
        assert snapshot["outstanding_requests"] == 0
    finally:
        dev.close()


@pytest.mark.parametrize("fixed", [False, True])
def test_raw_block_device_iouring_queue_and_buffer_stats(tmp_path, fixed):
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            io_engine="io_uring",
            iouring_queue_depth=2,
        )
    except Exception as error:
        if is_skip_safe_io_error(error):
            pytest.skip(f"io_uring unavailable: {error}")
        raise
    buffer = mmap.mmap(-1, 4096)
    try:
        if fixed:
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            dev.register_fixed_buffers([pointer], [4096])
        buffers = [buffer] * 32
        lengths = [4096] * 32
        offsets = [4096 * index for index in range(32)]
        batch = dev.batched_write(offsets, buffers, lengths)
        assert dev.wait_iouring(batch) == ([True] * 32, [])
        snapshot = dev.io_stats_snapshot()
        assert snapshot["write_attempts"] == 32
        assert snapshot["write_submitted_bytes"] == 32 * 4096
        assert snapshot["completed_attempts"] == 32
        assert snapshot["failed_attempts"] == 0
        assert snapshot["outstanding_requests"] == 0
        assert snapshot["queued_requests"] == 0
        assert snapshot["peak_queued_requests"] >= 1
        assert snapshot["fixed_buffer_attempts"] == (32 if fixed else 0)
        assert snapshot["fixed_buffer_submitted_bytes"] == (32 * 4096 if fixed else 0)
        assert snapshot["bounce_attempts"] == 0
    finally:
        dev.close()
        buffer.close()


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_device_short_read_counts_attempts(
    tmp_path: Path, io_engine: str
) -> None:
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(str(path), writable=True, io_engine=io_engine)
    except Exception as error:
        if is_skip_safe_io_error(error):
            pytest.skip(f"io_uring unavailable: {error}")
        raise
    try:
        if io_engine == "posix":
            with pytest.raises(RuntimeError, match="unexpected EOF"):
                dev.pread_into(RAW_BLOCK_CI_CAPACITY_BYTES - 4, bytearray(8), 8, 8)
        else:
            batch = dev.batched_read(
                [RAW_BLOCK_CI_CAPACITY_BYTES - 4],
                [bytearray(8)],
                [8],
            )
            success, errors = dev.wait_iouring(batch)
            assert success == [False]
            assert len(errors) == 1
        snapshot = dev.io_stats_snapshot()
        assert snapshot["read_attempts"] == 2
        assert snapshot["read_submitted_bytes"] == 12
        assert snapshot["completed_attempts"] == 1
        assert snapshot["failed_attempts"] == 1
        assert snapshot["outstanding_requests"] == 0
        assert snapshot["queued_requests"] == 0
    finally:
        dev.close()


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_device_validation_does_not_count_attempts(
    tmp_path: Path, io_engine: str
) -> None:
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(str(path), writable=True, io_engine=io_engine)
    except Exception as error:
        if is_skip_safe_io_error(error):
            pytest.skip(f"I/O engine unavailable: {error}")
        raise
    try:
        before = dev.io_stats_snapshot()
        if io_engine == "posix":
            with pytest.raises(ValueError):
                dev.pwrite_from_buffer(0, b"data", 5, 5)
        else:
            with pytest.raises(ValueError):
                dev.batched_write([0, 4096], [bytes(4096)], [4096, 4096])
        assert dev.io_stats_snapshot() == before
        assert "in_flight_operations" not in before
        assert "read_operations" not in before
        assert "read_bytes" not in before
    finally:
        dev.close()


def test_raw_block_device_iouring_bounce_and_failed_write(tmp_path):
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(str(path), writable=False, io_engine="io_uring")
    except Exception as error:
        if is_skip_safe_io_error(error):
            pytest.skip(f"io_uring unavailable: {error}")
        raise
    try:
        with pytest.raises(OSError):
            dev.write_uring(4096, b"data", 4, 4096)
        snapshot = dev.io_stats_snapshot()
        assert snapshot["write_attempts"] == 1
        assert snapshot["failed_attempts"] == 1
        assert snapshot["completed_attempts"] == 0
        assert snapshot["bounce_attempts"] == 1
        assert snapshot["bounce_submitted_bytes"] == 4096
        assert snapshot["outstanding_requests"] == 0
    finally:
        dev.close()


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
        snapshot = dev.io_stats_snapshot()
        assert snapshot["read_attempts"] == 1
        assert snapshot["write_attempts"] == 1
        assert snapshot["read_submitted_bytes"] == len(payload)
        assert snapshot["write_submitted_bytes"] == len(payload)
        assert snapshot["completed_attempts"] == 2
        assert snapshot["failed_attempts"] == 0
        assert snapshot["outstanding_requests"] == 0
        assert snapshot["queued_requests"] == 0
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
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
