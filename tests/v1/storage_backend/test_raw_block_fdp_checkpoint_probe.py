# SPDX-License-Identifier: Apache-2.0
"""Destructive hardware-gated FDP checkpoint write probe."""

# Standard
import errno
import os

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    is_skip_safe_device_setup_error,
    is_skip_safe_fdp_status_error,
    make_memory_obj,
    make_object_key,
    require_fdp_char_device_path,
)


def _require_destructive_checkpoint_device() -> str:
    """Return a verified NVMe char device path for destructive checkpoint writes."""
    if os.environ.get("LMCACHE_TEST_RAW_BLOCK_DESTRUCTIVE") != "1":
        pytest.skip(
            "Set LMCACHE_TEST_RAW_BLOCK_DESTRUCTIVE=1 to write checkpoints "
            "to the raw-block test device."
        )

    return require_fdp_char_device_path()


def _select_checkpoint_placement_id(device_path: str) -> int:
    """Select a non-zero FDP placement identifier from the device or environment."""
    configured = os.environ.get("LMCACHE_TEST_META_CHECKPOINT_PLACEMENT_ID")
    if configured is not None and configured != "":
        return int(configured)

    lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
    raw_device = None
    try:
        raw_device = lmcache_rust_raw_block_io.RawBlockDevice(
            device_path,
            writable=False,
            use_odirect=False,
            alignment=RAW_BLOCK_CI_BLOCK_ALIGN,
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

    for placement_id, _ruh_id in status:
        if int(placement_id) > 0:
            return int(placement_id)
    pytest.skip(f"No non-zero FDP placement identifiers reported by {device_path}.")
    raise AssertionError("pytest.skip should not return")


def _make_checkpoint_probe_core(
    device_path: str,
    *,
    load_checkpoint_on_init: bool,
    meta_checkpoint_placement_id: int | None,
) -> RawBlockCore:
    """Create a small destructive RawBlockCore over the configured NVMe device."""
    return RawBlockCore(
        RawBlockCoreConfig(
            device_path=device_path,
            capacity_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES + RAW_BLOCK_CI_SLOT_BYTES * 16,
            block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
            header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            use_odirect=False,
            enable_zero_copy=False,
            meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            load_checkpoint_on_init=load_checkpoint_on_init,
            meta_verify_on_load=True,
            io_engine="io_uring",
            iouring_queue_depth=8,
            use_uring_cmd=True,
            max_data_transfer_size=RAW_BLOCK_CI_BLOCK_ALIGN,
            meta_checkpoint_placement_id=meta_checkpoint_placement_id,
        ),
        key_namespace="object",
    )


def test_uring_cmd_checkpoint_write_with_metadata_placement() -> None:
    """Write and reload a metadata checkpoint with an FDP placement identifier.

    This test writes raw-block metadata and one data slot to the configured NVMe
    namespace character device. Run it only on a device reserved for destructive
    raw-block testing. Unit tests cover the fake-device placement call boundary;
    this probe validates the real device write/reload path.
    """
    device_path = _require_destructive_checkpoint_device()
    placement_id = _select_checkpoint_placement_id(device_path)
    key = encode_object_key(make_object_key(9001))

    core = None
    recovered = None
    try:
        core = _make_checkpoint_probe_core(
            device_path,
            load_checkpoint_on_init=False,
            meta_checkpoint_placement_id=placement_id,
        )
        result = core.put_many([key], [make_memory_obj(b"checkpoint-probe")])
        assert result.results == [True]
        core.checkpoint_now()
        core.close()
        core = None

        recovered = _make_checkpoint_probe_core(
            device_path,
            load_checkpoint_on_init=True,
            meta_checkpoint_placement_id=None,
        )
        assert recovered.contains_key(key.encoded)
    except OSError as e:
        if e.errno in {errno.EACCES, errno.ENOSYS, errno.ENOTTY, errno.EPERM}:
            pytest.skip(f"FDP checkpoint probe unavailable on {device_path}: {e}")
        raise
    finally:
        if core is not None:
            core.close()
        if recovered is not None:
            recovered.close()
