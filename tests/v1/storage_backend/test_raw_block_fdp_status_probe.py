# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated FDP status probe for the Rust raw-block device binding."""

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    is_skip_safe_device_setup_error,
    is_skip_safe_fdp_status_error,
    require_fdp_char_device_path,
)


def test_uring_cmd_fetch_fdp_status_status_only_probe() -> None:
    """Probe FDP status on an explicitly configured NVMe character device.

    This hardware-gated test is status-only. It does not write to the device,
    initialize the raw-block adapter layout, or verify KV placement policy.
    Set ``LMCACHE_TEST_FDP_CHAR_DEVICE`` to an FDP-capable NVMe namespace
    character device such as ``/dev/ng0n1``.
    """
    device_path = require_fdp_char_device_path()

    lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
    raw_device = None
    try:
        raw_device = lmcache_rust_raw_block_io.RawBlockDevice(
            device_path,
            writable=False,
            use_odirect=False,
            alignment=4096,
            io_engine="io_uring",
            use_uring_cmd=True,
            iouring_queue_depth=256,
        )
    except Exception as e:
        if is_skip_safe_device_setup_error(e):
            pytest.skip(f"FDP status probe setup is unavailable on {device_path}: {e}")
        raise

    try:
        status = raw_device.fetch_fdp_status()
    except Exception as e:
        if is_skip_safe_fdp_status_error(e):
            pytest.skip(f"FDP status probe is unavailable on {device_path}: {e}")
        raise
    finally:
        if raw_device is not None:
            raw_device.close()

    assert isinstance(status, list)
    assert status, f"Expected FDP status entries from {device_path}, got none"
    for placement_id, ruh_id in status:
        assert isinstance(placement_id, int)
        assert isinstance(ruh_id, int)
        assert 0 <= placement_id <= 0xFFFF
        assert 0 <= ruh_id <= 0xFFFF
