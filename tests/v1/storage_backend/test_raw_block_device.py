# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import mmap
import os
import platform
import weakref

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


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("size", [17, 4096, 4097, 10000])
def test_iouring_tail_roundtrip_preserves_source_and_guards(
    tmp_path: Path, batched: bool, size: int
) -> None:
    """Exercise padded scalar and vectored I/O through the native public API."""
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=4096,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )
    except Exception as exc:
        if is_skip_safe_io_error(exc):
            pytest.skip(f"io_uring unavailable: {exc}")
        raise
    total = (size + 4095) // 4096 * 4096
    source = mmap.mmap(-1, total + 4096)
    target = mmap.mmap(-1, total + 4096)
    source[:] = bytes([0xA5]) * len(source)
    target[:] = bytes([0xCC]) * len(target)
    src = memoryview(source)[:size]
    dst = memoryview(target)[:size]
    try:
        if batched:
            batch = dev.batched_write([0], [src], [total], None, [size])
            assert dev.wait_iouring(batch) == ([True], [])
            batch = dev.batched_read([0], [dst], [total])
            assert dev.wait_iouring(batch) == ([True], [])
        else:
            dev.write_uring(0, src, size, total)
            dev.read_uring(0, dst, size, total)
        assert source[:] == bytes([0xA5]) * len(source)
        assert target[:size] == bytes([0xA5]) * size
        assert target[size:] == bytes([0xCC]) * (len(target) - size)
        with path.open("rb") as stored:
            assert stored.read(total) == bytes(src) + bytes(total - size)
    finally:
        dev.close()
        src.release()
        dst.release()
        source.close()
        target.close()


def test_iouring_short_vectored_read_does_not_copy_tail(tmp_path: Path) -> None:
    """EOF inside the tail fails rather than retrying with scalar addressing."""
    path = make_raw_block_file(tmp_path)
    with path.open("r+b") as stored:
        stored.truncate(4096)
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=4096,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )
    except Exception as exc:
        if is_skip_safe_io_error(exc):
            pytest.skip(f"io_uring unavailable: {exc}")
        raise
    target = mmap.mmap(-1, 8192)
    target[:] = bytes([0xCC]) * len(target)
    dst = memoryview(target)[:5000]
    try:
        batch = dev.batched_read([0], [dst], [8192])
        results, errors = dev.wait_iouring(batch)
        assert results == [False]
        assert errors
        assert target[4096:] == bytes([0xCC]) * 4096
    finally:
        dev.close()
        dst.release()
        target.close()


def test_iouring_batch_retains_payload_view_until_wait(tmp_path: Path) -> None:
    """The batch must own the direct prefix exporter, not only its pointer."""
    path = make_raw_block_file(tmp_path)
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=4096,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )
    except Exception as exc:
        if is_skip_safe_io_error(exc):
            pytest.skip(f"io_uring unavailable: {exc}")
        raise
    source = mmap.mmap(-1, 8192)
    source[:] = bytes([9]) * 8192
    view = memoryview(source)[:5000]
    reference = weakref.ref(view)
    try:
        batch = dev.batched_write([0], [view], [8192], None, [5000])
        del view
        assert reference() is not None
        assert dev.wait_iouring(batch) == ([True], [])
        assert reference() is None
        with path.open("rb") as stored:
            assert stored.read(8192) == bytes([9]) * 5000 + bytes(3192)
    finally:
        dev.close()
        source.close()


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


def test_raw_block_device_iouring_batched_write_padded_roundtrip(tmp_path):
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

        payload = bytearray(b"padded-payload")
        total = RAW_BLOCK_CI_BLOCK_ALIGN
        out = bytearray(total)

        # payload_len < total_len: the source holds only len(payload) bytes but
        # the transfer is padded up to total. batched_write must copy only the
        # payload and zero-fill the tail.
        batch_id = dev.batched_write(
            [4096], [payload], [total], payload_lens=[len(payload)]
        )
        dev.wait_iouring(batch_id)
        batch_id = dev.batched_read([4096], [out], [total])
        dev.wait_iouring(batch_id)

        assert out[: len(payload)] == payload
        assert out[len(payload) :] == bytearray(total - len(payload))
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


def test_raw_block_device_iouring_batched_write_zeroes_existing_tail(tmp_path):
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

        payload = b"padded-payload"
        total = RAW_BLOCK_CI_BLOCK_ALIGN
        buf = bytearray([0xAB]) * total
        buf[: len(payload)] = payload
        out = bytearray(total)

        batch_id = dev.batched_write(
            [4096], [buf], [total], payload_lens=[len(payload)]
        )
        dev.wait_iouring(batch_id)
        batch_id = dev.batched_read([4096], [out], [total])
        dev.wait_iouring(batch_id)

        assert out[: len(payload)] == payload
        assert out[len(payload) :] == bytearray(total - len(payload))
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


def test_raw_block_device_iouring_batched_write_mixed_padding_batch(tmp_path):
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

        total = RAW_BLOCK_CI_BLOCK_ALIGN
        non_padded = bytearray(b"non-padded")
        zero_tail_payload = b"zero-tail-payload"
        zero_tail = bytearray(total)
        zero_tail[: len(zero_tail_payload)] = zero_tail_payload
        stale_tail_payload = b"stale-tail-payload"
        stale_tail = bytearray([0xEF]) * total
        stale_tail[: len(stale_tail_payload)] = stale_tail_payload
        short_payload = bytearray(b"short-source-payload")

        offsets = [4096, 8192, 12288, 16384]
        buffers = [non_padded, zero_tail, stale_tail, short_payload]
        total_lens = [len(non_padded), total, total, total]
        payload_lens = [
            len(non_padded),
            len(zero_tail_payload),
            len(stale_tail_payload),
            len(short_payload),
        ]

        batch_id = dev.batched_write(
            offsets, buffers, total_lens, payload_lens=payload_lens
        )
        dev.wait_iouring(batch_id)

        outs = [
            bytearray(len(non_padded)),
            bytearray(total),
            bytearray(total),
            bytearray(total),
        ]
        batch_id = dev.batched_read(offsets, outs, total_lens)
        dev.wait_iouring(batch_id)

        assert outs[0] == non_padded
        assert outs[1][: len(zero_tail_payload)] == zero_tail_payload
        assert outs[1][len(zero_tail_payload) :] == bytearray(
            total - len(zero_tail_payload)
        )
        assert outs[2][: len(stale_tail_payload)] == stale_tail_payload
        assert outs[2][len(stale_tail_payload) :] == bytearray(
            total - len(stale_tail_payload)
        )
        assert outs[3][: len(short_payload)] == short_payload
        assert outs[3][len(short_payload) :] == bytearray(total - len(short_payload))
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


def test_raw_block_device_iouring_write_uring_zeroes_existing_tail(tmp_path):
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

        payload = b"serial-padded-payload"
        total = RAW_BLOCK_CI_BLOCK_ALIGN
        buf = bytearray([0xCD]) * total
        buf[: len(payload)] = payload
        out = bytearray(total)

        dev.write_uring(4096, buf, len(payload), total)
        batch_id = dev.batched_read([4096], [out], [total])
        dev.wait_iouring(batch_id)

        assert out[: len(payload)] == payload
        assert out[len(payload) :] == bytearray(total - len(payload))
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"io_uring is unavailable on this runner: {e}")
        raise
    finally:
        if dev is not None:
            dev.close()


def test_raw_block_device_iouring_batched_write_validates_payload_lengths(tmp_path):
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

        with pytest.raises(
            ValueError, match="payload_lens must have same length as offsets"
        ):
            dev.batched_write([4096], [bytearray(b"payload")], [4096], payload_lens=[])

        with pytest.raises(ValueError, match="total_len must be >= payload_len"):
            dev.batched_write([4096], [bytearray(b"payload")], [4], payload_lens=[5])

        with pytest.raises(ValueError, match="input buffer too small"):
            dev.batched_write([4096], [bytearray(b"x")], [4096], payload_lens=[2])
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
def test_raw_block_device_odirect_batched_write_padded_roundtrip(tmp_path):
    path = make_raw_block_file(tmp_path)
    dev = None
    try:
        dev = RawBlockDevice(
            str(path),
            writable=True,
            use_odirect=True,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="io_uring",
            iouring_queue_depth=8,
        )

        payload = bytearray(b"padded-odirect-payload")
        total = RAW_BLOCK_CI_BLOCK_ALIGN

        batch_id = dev.batched_write(
            [4096], [payload], [total], payload_lens=[len(payload)]
        )
        dev.wait_iouring(batch_id)
        dev.close()
        dev = None

        # Read the bytes physically written with a non-O_DIRECT device so the
        # padding region can be inspected without aligned-buffer requirements.
        verify = RawBlockDevice(
            str(path),
            writable=False,
            use_odirect=False,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
            io_engine="posix",
            iouring_queue_depth=8,
        )
        try:
            out = bytearray(total)
            verify.pread_into(4096, out, total, total)
            assert out[: len(payload)] == payload
            assert out[len(payload) :] == bytearray(total - len(payload))
        finally:
            verify.close()
    except Exception as e:
        if is_skip_safe_io_error(e):
            pytest.skip(f"O_DIRECT is unavailable on this runner: {e}")
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
