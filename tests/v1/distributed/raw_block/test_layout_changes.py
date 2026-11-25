# SPDX-License-Identifier: Apache-2.0
"""Startup after a metadata layout change or a metadata read error."""

# Standard
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import importlib.util
import random

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import (
    RawBlockCore,
    encode_object_key,
)
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import PosixBlockIO
from tests.v1.distributed.raw_block.test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)

ALIGN = RAW_BLOCK_CI_BLOCK_ALIGN
HEADER = RAW_BLOCK_CI_HEADER_BYTES
M1 = RAW_BLOCK_CI_META_TOTAL_BYTES


def _spec(i: int):
    return encode_object_key(make_object_key(i))


def _payload(i: int, size: int) -> bytes:
    return random.Random(i).randbytes(size)


def _config(tmp_path: Path, data_bytes: int = 64 << 20):
    path = make_raw_block_file(tmp_path, M1 + data_bytes)
    return replace(
        make_raw_block_core_config(path, M1 + data_bytes),
        meta_enable_periodic=False,
        meta_verify_on_load=True,
    )


def _put(core: RawBlockCore, i: int, payload: bytes) -> bool:
    return core.put_many([_spec(i)], [make_memory_obj(payload)]).results == [True]


def _load(core: RawBlockCore, i: int, size: int) -> bytes | None:
    obj = make_empty_memory_obj(size)
    if core.load_many_into([_spec(i).encoded], [obj]) != [True]:
        return None
    return memory_obj_bytes(obj)


def test_changing_meta_total_bytes_and_back_never_serves_overwritten_bytes(tmp_path):
    """Run 1 (meta_total_bytes=M1) lists K in both copies. Run 2 with a
    larger meta_total_bytes rejects that checkpoint and stores over K's
    payload but not its header. Run 3 with M1 again must not load run 1's
    second copy (at M1/2, outside run 2's layout) over those bytes."""
    config = _config(tmp_path)
    m2 = 4 * M1
    k_payload = _payload(1, 8 * ALIGN)
    run1 = RawBlockCore(config)
    try:
        # Filler so K's header sits one block before run 2's data base.
        filler_alloc = m2 - ALIGN - (M1 + ALIGN)
        assert _put(run1, 0, _payload(0, filler_alloc - HEADER))
        assert _put(run1, 1, k_payload)
        assert run1.entry_offset(_spec(1).encoded) == m2 - ALIGN
        run1.checkpoint_now()  # seq 1 -> container 0
        assert _put(run1, 2, _payload(2, 100))
        run1.checkpoint_now()  # seq 2 -> container at M1/2, lists K
    finally:
        run1.close()

    run2 = RawBlockCore(replace(config, meta_total_bytes=m2))
    try:
        assert run2.indexed_key_count() == 0  # rejected: meta_total_bytes
        # Lands at m2 + ALIGN: inside K's payload, K's header untouched.
        assert _put(run2, 9, _payload(9, 4 * ALIGN))
    finally:
        run2.close()  # one checkpoint, into run 2's container at M2/2

    run3 = RawBlockCore(config)
    try:
        got = _load(run3, 1, len(k_payload))
        served_wrong = got is not None and got != k_payload
        assert not served_wrong, "K served bytes another run wrote"
    finally:
        run3.close()


def test_a_transient_metadata_read_error_does_not_destroy_the_checkpoint(tmp_path):
    """A metadata container that cannot be read is not a missing checkpoint:
    startup stops instead of voiding it, and a later start loads it."""
    config = _config(tmp_path)
    run1 = RawBlockCore(config)
    try:
        assert _put(run1, 1, _payload(1, 3 * ALIGN))
    finally:
        run1.close()
    meta_offsets = set(run1.metadata_container_offsets())
    real_read = PosixBlockIO.read_buffers

    def eio_on_metadata(self, offsets, *args, **kwargs):
        if any(off in meta_offsets for off in offsets):
            raise OSError(5, "EIO")
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", eio_on_metadata):
        with pytest.raises(RuntimeError, match="metadata region"):
            RawBlockCore(config)
    again = RawBlockCore(config)
    try:
        assert again.snapshot_indexed_keys() == [_spec(1).encoded]
    finally:
        again.close()


def test_a_metadata_read_that_reports_failure_stops_startup(tmp_path):
    """A failed read reported as a result, not raised, is not a missing
    checkpoint either."""
    config = _config(tmp_path)
    run1 = RawBlockCore(config)
    try:
        assert _put(run1, 1, _payload(1, 3 * ALIGN))
    finally:
        run1.close()
    meta_offsets = set(run1.metadata_container_offsets())
    real_read = PosixBlockIO.read_buffers

    def failed_on_metadata(self, offsets, *args, **kwargs):
        if any(off in meta_offsets for off in offsets):
            return [False] * len(offsets)
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", failed_on_metadata):
        with pytest.raises(RuntimeError, match="metadata region"):
            RawBlockCore(config)
    again = RawBlockCore(config)
    try:
        assert again.snapshot_indexed_keys() == [_spec(1).encoded]
    finally:
        again.close()


def test_a_failed_store_leaves_nothing_to_checkpoint(tmp_path):
    config = _config(tmp_path)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, _payload(1, ALIGN))
        core.checkpoint_now()
        seq = core.report_status()["metadata_seq"]
        with patch.object(PosixBlockIO, "write_buffers", side_effect=OSError("x")):
            assert not _put(core, 2, _payload(2, ALIGN))
        core.checkpoint_now()
        assert core.report_status()["metadata_seq"] == seq
    finally:
        core.close()
