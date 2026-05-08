# SPDX-License-Identifier: Apache-2.0
"""Tests for the incremental (base + delta log) checkpoint format.

These exercise ``RawBlockCore`` directly through a fake raw-block device,
covering the v2 layout, v1 legacy layout, cross-version load, delta append,
compaction triggers, and torn-write recovery.
"""

# Future
from __future__ import annotations

# Standard
import sys
import types
import zlib

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.raw_block import RawBlockCore, RawBlockCoreConfig
from lmcache.v1.storage_backend.raw_block.core import (
    _DELTA_RECORD_HEADER_STRUCT,
    _DELTA_RECORD_MAGIC,
    _DELTA_RECORD_VERSION,
    _META_HEADER_V1_STRUCT,
    _META_HEADER_V2_STRUCT,
    _META_VERSION_V1,
    _META_VERSION_V2,
    _Entry,
)


class _FakeRawBlockDevice:
    """A purely in-memory device backing for unit tests."""

    def __init__(self, buf: bytearray):
        self._data = buf

    def size_bytes(self) -> int:
        return len(self._data)

    def pread_into(self, offset, out, payload_len, total_len=None):
        del total_len
        out[:payload_len] = self._data[offset : offset + payload_len]

    def pwrite_from_buffer(self, offset, data, payload_len=None, total_len=None):
        del total_len
        length = len(data) if payload_len is None else payload_len
        self._data[offset : offset + length] = bytes(memoryview(data)[:length])

    def close(self) -> None:
        return None


@pytest.fixture
def shared_device(monkeypatch):
    """Install a shared in-memory device that survives across cores."""

    holder: dict[str, bytearray] = {"buf": bytearray(2 * 1024 * 1024)}

    def factory(path: str, **kwargs):
        del path, kwargs
        return _FakeRawBlockDevice(holder["buf"])

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=factory),
    )
    return holder


def _config(**overrides) -> RawBlockCoreConfig:
    base = dict(
        device_path="/tmp/raw-block-incremental-test",
        capacity_bytes=2 * 1024 * 1024,
        block_align=4096,
        header_bytes=4096,
        slot_bytes=8192,
        use_odirect=False,
        enable_zero_copy=True,
        meta_total_bytes=256 * 1024,
        meta_magic=b"LMCIDX01",
        meta_version=1,
        meta_checkpoint_interval_sec=600,
        meta_idle_quiet_ms=0,
        meta_enable_periodic=False,
        load_checkpoint_on_init=True,
        meta_verify_on_load=False,
        io_engine="posix",
        iouring_queue_depth=256,
        meta_incremental_enabled=True,
        meta_full_checkpoint_max_deltas=10_000,
        meta_delta_high_watermark_pct=100,
    )
    base.update(overrides)
    return RawBlockCoreConfig(**base)


def _put_locked(core: RawBlockCore, key: str, slot_idx: int, size: int = 512) -> None:
    """Inject a synthetic indexed entry through the public dirty-log helpers."""
    meta = DiskCacheMetadata(
        path=f"{core.device_path}@{slot_idx}",
        size=size,
        shape=torch.Size((2, 16, 8, 128)),
        dtype=torch.bfloat16,
        cached_positions=None,
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    entry = _Entry(
        offset=core._data_base_offset + slot_idx * core.slot_bytes,
        size=size,
        meta=meta,
    )
    with core._lock:
        core._index[key] = entry
        core._next_slot = max(core._next_slot, slot_idx + 1)
        core._record_put_op_locked(key, entry)
        core._meta_dirty_total += 1


def _delete_locked(core: RawBlockCore, key: str) -> None:
    with core._lock:
        removed = core._index.pop(key, None)
        if removed is None:
            return
        slot = core._offset_to_slot(int(removed.offset))
        core._append_free_slot_locked(slot)
        core._record_delete_op_locked(key, slot)
        core._meta_dirty_total += 1


def test_v2_layout_carves_base_and_delta_regions(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        layout = core._meta_layout
        assert core._use_incremental is True
        assert layout.delta_region_bytes > 0
        assert (
            layout.base_copy_bytes * 2 + layout.delta_region_bytes
            <= core.meta_total_bytes
        )
    finally:
        core.close()


def test_small_meta_region_falls_back_to_v1(shared_device):
    # 16 KB total: too small for 2 base mirrors + delta region.
    core = RawBlockCore(_config(meta_total_bytes=16 * 1024), key_namespace="object")
    try:
        assert core._use_incremental is False
        assert core._meta_layout.delta_region_bytes == 0
    finally:
        core.close()


def test_first_checkpoint_is_full_then_subsequent_are_delta(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        assert core._checkpoint_once(force=True) is True
        assert core._meta_seq == 1
        assert core._delta_seq == 0
        assert core._active_base_is_v2 is True

        # Second mutation should append a delta record (meta_seq fixed).
        _put_locked(core, "k1", slot_idx=1)
        assert core._checkpoint_once(force=True) is True
        assert core._meta_seq == 1
        assert core._delta_seq == 1
        assert core._delta_tail_off > 0

        # On-disk delta record header is well-formed.
        delta_off = core._meta_layout.delta_region_off
        hdr_bytes = bytes(
            shared_device["buf"][
                delta_off : delta_off + _DELTA_RECORD_HEADER_STRUCT.size
            ]
        )
        (
            magic,
            ver,
            base_seq,
            delta_seq,
            payload_len,
            payload_crc,
            total_blocks,
            op_count,
            _reserved,
        ) = _DELTA_RECORD_HEADER_STRUCT.unpack(hdr_bytes)
        assert magic == _DELTA_RECORD_MAGIC
        assert ver == _DELTA_RECORD_VERSION
        assert base_seq == 1
        assert delta_seq == 1
        assert op_count >= 1
        assert total_blocks >= 1

        payload_start = delta_off + _DELTA_RECORD_HEADER_STRUCT.size
        payload = bytes(
            shared_device["buf"][payload_start : payload_start + payload_len]
        )
        assert (zlib.crc32(payload) & 0xFFFFFFFF) == payload_crc
    finally:
        core.close()


def test_round_trip_replays_deltas_on_remount(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)  # full
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)  # delta 1
    _put_locked(core, "k2", slot_idx=2)
    core._checkpoint_once(force=True)  # delta 2
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert {"k0", "k1", "k2"}.issubset(revived._index.keys())
        assert revived._meta_seq == 1
        assert revived._delta_seq == 2
        assert revived._next_slot >= 3
    finally:
        revived.close()


def test_torn_last_delta_truncates_replay(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)
    _put_locked(core, "k2", slot_idx=2)
    core._checkpoint_once(force=True)
    last_record_size = core._delta_record_total_bytes(1)  # one block per record
    last_record_off = (
        core._meta_layout.delta_region_off + core._delta_tail_off - last_record_size
    )
    # Flip a single payload byte to break the CRC of the last delta record.
    target = last_record_off + _DELTA_RECORD_HEADER_STRUCT.size + 4
    shared_device["buf"][target] ^= 0xFF
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "k0" in revived._index
        assert "k1" in revived._index
        assert "k2" not in revived._index
        assert revived._delta_seq == 1
    finally:
        revived.close()


def test_stale_base_seq_deltas_are_ignored(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)  # base seq=1
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)  # delta(base_seq=1, dseq=1)

    # Manually rewrite the *other* base mirror with seq=2 to simulate a fresh
    # full snapshot whose deltas have not been written yet. The leftover
    # deltas in the region have base_seq=1 and must be ignored on replay.
    other_base_off = core._meta_layout.base_copy_offsets[1]
    payload = (
        b'{"version":1,"device_path":"/tmp/raw-block-incremental-test",'
        b'"capacity_bytes":2097152,"block_align":4096,"header_bytes":4096,'
        b'"slot_bytes":8192,"meta_total_bytes":262144,"meta_magic":"LMCIDX01",'
        b'"meta_version":1,"data_base_offset":262144,"next_slot":1,'
        b'"free_slots":[],"entries":{"k0":{"offset":262144,"size":512,'
        b'"shape":[2,16,8,128],"dtype":"bfloat16","fmt":"KV_T2D",'
        b'"cached_positions":null}}}'
    )
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = bytearray(core.block_align)
    header[: _META_HEADER_V2_STRUCT.size] = _META_HEADER_V2_STRUCT.pack(
        b"LMCIDX01", _META_VERSION_V2, 2, len(payload), crc, 0, 0
    )
    shared_device["buf"][other_base_off : other_base_off + core.block_align] = bytes(
        header
    )
    aligned_payload = bytearray(
        ((len(payload) + core.block_align - 1) // core.block_align) * core.block_align
    )
    aligned_payload[: len(payload)] = payload
    shared_device["buf"][
        other_base_off + core.block_align : other_base_off
        + core.block_align
        + len(aligned_payload)
    ] = bytes(aligned_payload)
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert revived._meta_seq == 2
        assert "k0" in revived._index
        assert "k1" not in revived._index
        assert revived._delta_seq == 0
    finally:
        revived.close()


def test_non_monotonic_delta_seq_stops_replay(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)
    _put_locked(core, "k2", slot_idx=2)
    core._checkpoint_once(force=True)
    delta_off = core._meta_layout.delta_region_off
    hdr_bytes = bytearray(
        shared_device["buf"][delta_off : delta_off + _DELTA_RECORD_HEADER_STRUCT.size]
    )
    fields = list(_DELTA_RECORD_HEADER_STRUCT.unpack(bytes(hdr_bytes)))
    fields[3] = 99  # tamper with delta_seq
    shared_device["buf"][delta_off : delta_off + _DELTA_RECORD_HEADER_STRUCT.size] = (
        _DELTA_RECORD_HEADER_STRUCT.pack(*fields)
    )
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "k0" in revived._index
        assert "k1" not in revived._index
        assert "k2" not in revived._index
        assert revived._delta_seq == 0
    finally:
        revived.close()


def test_max_deltas_threshold_triggers_compaction(shared_device):
    core = RawBlockCore(
        _config(meta_full_checkpoint_max_deltas=4), key_namespace="object"
    )
    try:
        _put_locked(core, "seed", slot_idx=0)
        core._checkpoint_once(force=True)
        seed_seq = core._meta_seq
        for i in range(8):
            _put_locked(core, f"k{i}", slot_idx=i + 1)
            core._checkpoint_once(force=True)
        assert core._meta_seq > seed_seq, (
            "compaction should have advanced base seq past max_deltas"
        )
    finally:
        core.close()


def test_v1_to_v2_migration_writes_full_first(shared_device):
    v1 = RawBlockCore(_config(meta_incremental_enabled=False), key_namespace="object")
    _put_locked(v1, "k0", slot_idx=0)
    v1._checkpoint_once(force=True)
    v1_seq = v1._meta_seq
    v1.close()

    v2 = RawBlockCore(_config(), key_namespace="object")
    try:
        # Loading a v1 base must mark active_base_is_v2 as False.
        assert v2._active_base_is_v2 is False

        # Force a write while still in migration: must be a full snapshot.
        _put_locked(v2, "k1", slot_idx=1)
        v2._checkpoint_once(force=True)
        assert v2._meta_seq == v1_seq + 1
        assert v2._active_base_is_v2 is True
        assert v2._delta_seq == 0
    finally:
        v2.close()


def test_report_status_exposes_incremental_state(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        status = core.report_status()
        assert status["incremental_enabled"] is True
        assert "delta_seq" in status
        assert "delta_tail_off" in status
        assert "delta_region_bytes" in status
        assert "active_base_seq" in status
    finally:
        core.close()


def test_disabling_incremental_keeps_v1_layout(shared_device):
    core = RawBlockCore(_config(meta_incremental_enabled=False), key_namespace="object")
    try:
        assert core._use_incremental is False
        assert core._meta_layout.delta_region_bytes == 0
        # First write goes to base mirror via the v1 header struct.
        _put_locked(core, "k0", slot_idx=0)
        assert core._checkpoint_once(force=True) is True
        hdr = bytes(shared_device["buf"][: _META_HEADER_V1_STRUCT.size])
        magic, version, *_ = _META_HEADER_V1_STRUCT.unpack(hdr)
        assert magic == b"LMCIDX01"
        assert version == _META_VERSION_V1
    finally:
        core.close()


def test_delete_op_replays_through_delta(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)  # full with both keys
    _delete_locked(core, "k0")
    core._checkpoint_once(force=True)  # delta drops k0
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "k0" not in revived._index
        assert "k1" in revived._index
    finally:
        revived.close()


def test_close_with_pending_dirty_log_flushes_via_delta(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)  # full base
    _put_locked(core, "k1", slot_idx=1)
    core.close()  # final flush

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "k0" in revived._index
        assert "k1" in revived._index
        assert revived._delta_seq >= 1, (
            "close must persist pending mutations as a delta record"
        )
    finally:
        revived.close()


def test_v1_disk_with_latest_state_in_mirror_1_loads_under_v2(shared_device):
    """Regression for B1.

    A v1 disk where the most recent checkpoint sits in mirror_1 (50/50
    layout, offset = ``meta_total_bytes / 2``) must remain loadable when
    mounted by the v2 code path. v2's own base offsets do not cover the
    legacy mirror_1 location, so the loader must fall back to the v1
    offsets.
    """
    v1 = RawBlockCore(_config(meta_incremental_enabled=False), key_namespace="object")
    _put_locked(v1, "k0", slot_idx=0)
    v1._checkpoint_once(force=True)  # seq=1 -> v1 mirror_0
    _put_locked(v1, "k1", slot_idx=1)
    v1._checkpoint_once(force=True)  # seq=2 -> v1 mirror_1
    assert v1._meta_seq == 2
    v1.close()

    v2 = RawBlockCore(_config(), key_namespace="object")
    try:
        assert v2._meta_seq == 2
        assert "k0" in v2._index
        assert "k1" in v2._index, "v1 mirror_1's latest entry must be recovered"
    finally:
        v2.close()


def test_concurrent_checkpoint_calls_do_not_lose_delta_records(shared_device):
    """Regression for B2.

    Two concurrent ``_checkpoint_once`` invocations must each persist their
    delta record without overwriting the other. Verified by inspecting the
    delta region (two distinct records on disk) and by reloading the keys
    on remount.
    """
    # Standard
    import threading
    import time

    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "seed", slot_idx=0)
    core._checkpoint_once(force=True)

    # Slow the device so the writer-lock contention window is observable.
    real_pwrite = core._rawdev().pwrite_from_buffer

    def slow_pwrite(*args, **kwargs):
        time.sleep(0.005)
        return real_pwrite(*args, **kwargs)

    core._rawdev().pwrite_from_buffer = slow_pwrite  # type: ignore[method-assign]

    def worker(idx: int) -> None:
        _put_locked(core, f"key{idx}", slot_idx=idx + 1)
        core._checkpoint_once(force=True)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert core._delta_seq == 2, (
        "both checkpoint_once calls must produce a unique delta record"
    )
    delta_off = core._meta_layout.delta_region_off
    region_bytes = core._meta_layout.delta_region_bytes
    visible = []
    walk = 0
    while walk + _DELTA_RECORD_HEADER_STRUCT.size <= region_bytes:
        hdr = bytes(
            shared_device["buf"][
                delta_off + walk : delta_off + walk + _DELTA_RECORD_HEADER_STRUCT.size
            ]
        )
        magic, _, _, dseq, _, _, blocks, _, _ = _DELTA_RECORD_HEADER_STRUCT.unpack(hdr)
        if magic != _DELTA_RECORD_MAGIC:
            break
        visible.append(int(dseq))
        walk += int(blocks) * core.block_align
    assert visible == [1, 2], f"expected two records on disk; got {visible}"
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert {"seed", "key0", "key1"}.issubset(revived._index.keys())
    finally:
        revived.close()


def test_safe_freelist_push_refuses_slot_owned_by_indexed_entry(shared_device):
    """Regression: the delta-replay free-slot helper guards owned slots.

    The fast ``_append_free_slot_locked`` is used on hot paths that have
    just removed the owning entry, so it skips the guard. The safe helper
    used by delta replay must refuse to free a slot still claimed by an
    indexed entry, otherwise a malformed delta could cause double-issue
    on the next allocation.
    """
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        _put_locked(core, "live_key", slot_idx=4)
        with core._lock:
            free_before = len(core._free_slots)
            core._safe_append_free_slot_locked(4)
            assert len(core._free_slots) == free_before, (
                "safe append must refuse a slot that an indexed entry owns"
            )
            core._safe_append_free_slot_locked(7)
            assert 7 in core._free_slots
    finally:
        core.close()


def test_external_apply_loaded_state_does_not_resurrect_stale_deltas(
    shared_device,
):
    """Regression for B-1.

    Mounting a v2 disk, calling ``apply_loaded_state`` with a synthetic state,
    then writing a fresh checkpoint must not re-introduce keys whose delta
    records are still on disk. The new full base resets the active seq and
    invalidates the head of the delta region; without that, stale records
    whose ``base_seq`` happens to match the new full's seq number would
    replay on top of the externally supplied state.
    """
    initial = RawBlockCore(_config(), key_namespace="object")
    _put_locked(initial, "real0", slot_idx=0)
    initial._checkpoint_once(force=True)  # full base
    _put_locked(initial, "real1", slot_idx=1)
    initial._checkpoint_once(force=True)  # delta1
    _put_locked(initial, "real2", slot_idx=2)
    initial._checkpoint_once(force=True)  # delta2
    initial.close()

    overrider = RawBlockCore(_config(), key_namespace="object")
    assert {"real0", "real1", "real2"}.issubset(overrider._index.keys())
    overrider.apply_loaded_state(
        {
            "version": 1,
            "device_path": overrider.device_path,
            "capacity_bytes": overrider.capacity_bytes,
            "block_align": overrider.block_align,
            "header_bytes": overrider.header_bytes,
            "slot_bytes": overrider.slot_bytes,
            "meta_total_bytes": overrider.meta_total_bytes,
            "meta_magic": overrider.meta_magic_text,
            "meta_version": overrider.meta_version,
            "data_base_offset": overrider._data_base_offset,
            "next_slot": 0,
            "free_slots": [],
            "entries": {},
        }
    )
    assert overrider._meta_seq == 0
    assert overrider._active_base_seq == 0
    assert overrider._active_base_is_v2 is False

    _put_locked(overrider, "newkey", slot_idx=5)
    overrider._checkpoint_once(force=True)
    overrider.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert sorted(revived._index.keys()) == ["newkey"], (
            f"stale deltas resurrected: {sorted(revived._index.keys())}"
        )
        assert revived._delta_seq == 0
    finally:
        revived.close()


def test_validate_loaded_entries_runs_after_delta_replay(shared_device):
    """Regression for S-1.

    When ``meta_verify_on_load=True``, validation must run after deltas have
    been applied so a key whose slot-header is corrupt cannot survive the
    final state, even if a delta later re-PUTs it.
    """
    core = RawBlockCore(_config(meta_verify_on_load=True), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)  # full base
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)  # delta1 PUT k1
    core.close()

    # Stub _read_slot_header so every entry looks invalid: the validator
    # should drop both k0 (from base) and k1 (re-introduced by the delta)
    # because validation runs *after* replay.
    revived = RawBlockCore(_config(meta_verify_on_load=False), key_namespace="object")
    revived.close()

    revived2 = RawBlockCore(_config(meta_verify_on_load=True), key_namespace="object")
    try:
        revived2._read_slot_header = lambda _off: None  # type: ignore[method-assign]
        revived2._validate_loaded_entries()
        assert revived2._index == {}, (
            "validator must drop entries (including delta-replayed) when "
            "slot headers do not match"
        )
    finally:
        revived2.close()


def test_fast_append_free_slot_skips_owned_index_check(shared_device):
    """Regression for S-2.

    The fast ``_append_free_slot_locked`` is used on hot paths after the
    caller has just removed the owning entry; it must not pay an O(N)
    index scan. The safe variant ``_safe_append_free_slot_locked`` does
    pay that cost and is used by delta replay.
    """
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        _put_locked(core, "live", slot_idx=3)
        with core._lock:
            # Fast path appends regardless of ownership; this matches the
            # contract that hot-path callers have already vacated the slot.
            core._append_free_slot_locked(3)
            assert 3 in core._free_slots

            core._free_slots = []
            # Safe path refuses the owned slot.
            core._safe_append_free_slot_locked(3)
            assert 3 not in core._free_slots
    finally:
        core.close()


def test_replay_logs_warning_on_unknown_op_kind(shared_device):
    """Regression for N4: unknown ops should be logged but not abort replay."""
    # Standard
    import logging

    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)
    # The lmcache loggers do not propagate to root, so caplog cannot see
    # them. Attach a one-shot handler directly to the module logger.
    captured: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _ListHandler(logging.WARNING)
    target_logger = logging.getLogger("lmcache.v1.storage_backend.raw_block.core")
    target_logger.addHandler(handler)
    try:
        ok = core._apply_delta_ops([{"op": "no_such_op", "k": "x"}])
    finally:
        target_logger.removeHandler(handler)
    core.close()
    assert ok is True, "unknown ops must be tolerated during replay"
    assert any("unknown delta op" in record.getMessage() for record in captured)


# --------------------------------------------------------------------------
# Recovery scenarios
# --------------------------------------------------------------------------


def test_recovery_replays_multi_op_delta_record(shared_device):
    """One delta record may carry several ops; replay must apply them all."""
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "seed", slot_idx=0)
    core._checkpoint_once(force=True)  # full base
    # Three mutations under a single checkpoint -> one delta with op_count=3.
    _put_locked(core, "k1", slot_idx=1)
    _put_locked(core, "k2", slot_idx=2)
    _delete_locked(core, "k1")
    core._checkpoint_once(force=True)
    assert core._delta_seq == 1, "all three mutations must share one record"

    # Inspect on-disk record header to confirm op_count == 3.
    delta_off = core._meta_layout.delta_region_off
    hdr = bytes(
        shared_device["buf"][delta_off : delta_off + _DELTA_RECORD_HEADER_STRUCT.size]
    )
    *_, op_count, _reserved = _DELTA_RECORD_HEADER_STRUCT.unpack(hdr)
    assert op_count == 3
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "seed" in revived._index
        assert "k1" not in revived._index, "delete op must take effect"
        assert "k2" in revived._index
    finally:
        revived.close()


def test_recovery_handles_put_delete_put_same_key(shared_device):
    """Replay must honor in-record op order; last write wins."""
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "key", slot_idx=0, size=128)
    core._checkpoint_once(force=True)
    # Same checkpoint: delete the key, then re-put at a different slot+size.
    _delete_locked(core, "key")
    _put_locked(core, "key", slot_idx=4, size=512)
    core._checkpoint_once(force=True)
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        entry = revived._index.get("key")
        assert entry is not None, "key must survive the delete+put cycle"
        expected_offset = revived._data_base_offset + 4 * revived.slot_bytes
        assert int(entry.offset) == expected_offset, (
            "final state must reflect the last PUT, not the first"
        )
        assert int(entry.size) == 512
    finally:
        revived.close()


def test_recovery_replays_long_delta_chain(shared_device):
    """A long delta chain replays correctly even when compaction interleaves.

    The exact chain length before compaction depends on payload size and
    the delta-region geometry, so this test only asserts on the recovered
    key set. Deltas spanning at least one full record after a compaction
    are present (see ``test_recovery_after_multiple_compaction_cycles`` for
    the compaction-cycle assertion).
    """
    chain_len = 30
    # Larger meta region so the long chain doesn't trip the watermark too
    # quickly; we still want at least a few deltas before compaction.
    core = RawBlockCore(
        _config(
            meta_total_bytes=1024 * 1024,
            meta_full_checkpoint_max_deltas=10_000,
        ),
        key_namespace="object",
    )
    _put_locked(core, "seed", slot_idx=0)
    core._checkpoint_once(force=True)
    for i in range(chain_len):
        _put_locked(core, f"chain_{i}", slot_idx=i + 1)
        core._checkpoint_once(force=True)
    core.close()

    revived = RawBlockCore(
        _config(meta_total_bytes=1024 * 1024), key_namespace="object"
    )
    try:
        recovered = set(revived._index.keys())
        expected = {"seed"} | {f"chain_{i}" for i in range(chain_len)}
        assert recovered == expected, f"missing after replay: {expected - recovered}"
    finally:
        revived.close()


def test_recovery_after_multiple_compaction_cycles(shared_device):
    """State survives several compactions interleaved with delta runs."""
    core = RawBlockCore(
        # Force a compaction every few deltas so cycles repeat.
        _config(meta_full_checkpoint_max_deltas=3),
        key_namespace="object",
    )
    expected_keys: set[str] = set()
    for i in range(20):
        key = f"k{i}"
        _put_locked(core, key, slot_idx=i)
        core._checkpoint_once(force=True)
        expected_keys.add(key)
    assert core._meta_seq >= 4, "test should produce at least 4 base writes"
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert set(revived._index.keys()) == expected_keys
    finally:
        revived.close()


def test_recovery_survives_close_mount_loop(shared_device):
    """Three close/mount cycles preserve every committed key."""
    cycles = 3
    keys_per_cycle = 4
    expected: set[str] = set()
    next_slot = 0
    for cycle in range(cycles):
        core = RawBlockCore(_config(), key_namespace="object")
        try:
            assert set(core._index.keys()) == expected
            for k in range(keys_per_cycle):
                key = f"c{cycle}_k{k}"
                _put_locked(core, key, slot_idx=next_slot)
                next_slot += 1
                expected.add(key)
            # Periodically force a delta vs full mix.
            core._checkpoint_once(force=True)
        finally:
            core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert set(revived._index.keys()) == expected
        assert revived._next_slot >= next_slot
    finally:
        revived.close()


def test_recovery_stops_on_delta_with_invalid_json(shared_device):
    """A delta whose payload passes CRC but parses to invalid JSON stops replay."""
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)  # full
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)  # delta1
    _put_locked(core, "k2", slot_idx=2)
    core._checkpoint_once(force=True)  # delta2
    layout = core._meta_layout
    # Corrupt delta1 payload to non-JSON bytes, then refresh CRC so the
    # CRC gate cannot mask the JSON failure we want to test.
    delta1_off = layout.delta_region_off
    hdr_buf = bytes(
        shared_device["buf"][delta1_off : delta1_off + _DELTA_RECORD_HEADER_STRUCT.size]
    )
    fields = list(_DELTA_RECORD_HEADER_STRUCT.unpack(hdr_buf))
    payload_len = int(fields[4])
    payload_off = delta1_off + _DELTA_RECORD_HEADER_STRUCT.size
    new_payload = b"<<<not json>>>" + b"\x00" * (payload_len - len("<<<not json>>>"))
    shared_device["buf"][payload_off : payload_off + payload_len] = new_payload
    fields[5] = zlib.crc32(new_payload) & 0xFFFFFFFF  # patch CRC to match
    shared_device["buf"][delta1_off : delta1_off + _DELTA_RECORD_HEADER_STRUCT.size] = (
        _DELTA_RECORD_HEADER_STRUCT.pack(*fields)
    )
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "k0" in revived._index
        assert "k1" not in revived._index, "JSON-invalid delta1 must stop replay"
        assert "k2" not in revived._index, "later deltas must not be applied"
        assert revived._delta_seq == 0
    finally:
        revived.close()


def test_recovery_reuses_freed_slot_after_delta_delete(shared_device):
    """A slot freed via a delta delete is available to the post-recovery allocator."""
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "victim", slot_idx=0)
    _put_locked(core, "spare", slot_idx=1)
    core._checkpoint_once(force=True)  # full base
    _delete_locked(core, "victim")
    core._checkpoint_once(force=True)  # delta drops slot 0
    core.close()

    revived = RawBlockCore(_config(), key_namespace="object")
    try:
        assert "victim" not in revived._index
        assert 0 in revived._free_slots, "freed slot must be on the free list"
        with revived._lock:
            offset = revived._allocate_slot_locked()
        expected = revived._data_base_offset + 0 * revived.slot_bytes
        assert offset == expected, (
            "allocator should reuse slot 0 instead of advancing _next_slot"
        )
    finally:
        revived.close()


def test_load_checkpoint_on_init_false_skips_delta_replay(shared_device):
    """``load_checkpoint_on_init=False`` must not replay deltas either."""
    core = RawBlockCore(_config(), key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)
    _put_locked(core, "k1", slot_idx=1)
    core._checkpoint_once(force=True)
    core.close()

    fresh = RawBlockCore(_config(load_checkpoint_on_init=False), key_namespace="object")
    try:
        assert fresh._index == {}
        assert fresh._meta_seq == 0
        assert fresh._delta_seq == 0
        assert fresh._delta_tail_off == 0
        assert fresh._active_base_is_v2 is False
    finally:
        fresh.close()
