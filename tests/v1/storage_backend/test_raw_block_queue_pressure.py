# SPDX-License-Identifier: Apache-2.0
"""Large batches must complete without exhausting the raw-block worker ring."""

# Standard
from pathlib import Path

# Third Party
import pytest

raw_block = pytest.importorskip("lmcache_rust_raw_block_io")


@pytest.mark.parametrize("depth", [1, 2, 4, 64])
@pytest.mark.parametrize("direct", [False, True])
def test_read_batches_larger_than_ring(
    tmp_path: Path, depth: int, direct: bool
) -> None:
    """Repeated oversized batches must return every per-I/O result and payload."""
    path = tmp_path / "queue-pressure.bin"
    page_bytes = 4096
    payload = b"".join(bytes([i]) * page_bytes for i in range(32))
    path.write_bytes(payload)
    device = raw_block.RawBlockDevice(
        str(path),
        writable=False,
        use_odirect=direct,
        io_engine="io_uring",
        iouring_queue_depth=depth,
    )
    try:
        for repeat in range(3):
            pages = [(i * 7 + repeat) % 32 for i in range(depth * 16)]
            buffers = [bytearray(page_bytes) for _ in pages]
            batch = device.batched_read(
                [page * page_bytes for page in pages],
                buffers,
                [page_bytes] * len(pages),
            )
            success, errors = device.wait_iouring(batch)
            assert success == [True] * len(pages)
            assert errors == []
            for page, buffer in zip(pages, buffers, strict=True):
                assert buffer == bytes([page]) * page_bytes
    finally:
        device.close()
