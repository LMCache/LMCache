# SPDX-License-Identifier: Apache-2.0
"""Tests for the incremental (base + delta tail) checkpoint format.

These exercise ``RawBlockCore`` directly through a fake raw-block device,
covering the per-mirror tail-append layout, base/delta binding, replay,
torn-write recovery, and compaction triggers.
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
    _META_HEADER_STRUCT,
    _Entry,
)


class _FakeRawBlockDevice:
    """Purely in-memory device backing for unit tests."""

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
        meta_full_checkpoint_max_deltas=10_000,
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


def _active_tail_start(core: RawBlockCore) -> int:
    return core._delta_tail_start_off(
        core._active_mirror_idx, core._active_base_payload_total
    )


def test_layout_is_two_equal_mirrors(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        layout = core._meta_layout
        assert len(layout.base_copy_offsets) == 2
        assert layout.base_copy_offsets[0] == 0
        assert layout.base_copy_offsets[1] == layout.base_copy_bytes
        # The two mirrors share meta_total_bytes evenly, no separate region.
        assert layout.base_copy_bytes * 2 <= core.meta_total_bytes
    finally:
        core.close()


def test_first_checkpoint_is_full_then_subsequent_are_delta(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        assert core._checkpoint_once(force=True) is True
        assert core._meta_seq == 1
        assert core._delta_seq == 0
        assert core._active_base_seq == 1
        assert core._active_base_crc != 0
        first_mirror = core._active_mirror_idx

        _put_locked(core, "k1", slot_idx=1)
        assert core._checkpoint_once(force=True) is True
        assert core._meta_seq == 1
        assert core._delta_seq == 1
        assert core._delta_tail_off > 0
        # No mirror flip on a delta append.
        assert core._active_mirror_idx == first_mirror

        tail_off = _active_tail_start(core)
        hdr_bytes = bytes(
            shared_device["buf"][
                tail_off : tail_off + _DELTA_RECORD_HEADER_STRUCT.size
            ]
        )
        (
            magic,
            ver,
            base_seq,
            base_crc,
            delta_seq,
            prev_crc,
            payload_len,
            payload_crc,
            op_count,
            flags,
            total_blocks,
            reserved,
        ) = _DELTA_RECORD_HEADER_STRUCT.unpack(hdr_bytes)
        assert magic == _DELTA_RECORD_MAGIC
        assert ver == _DELTA_RECORD_VERSION
        assert base_seq == core._active_base_seq
        assert base_crc == core._active_base_crc
        assert delta_seq == 1
        assert prev_crc == core._active_base_crc  # first record anchors at base_crc
        assert flags == 0
        assert total_blocks * core.block_align >= _DELTA_RECORD_HEADER_STRUCT.size
        assert reserved == 0
        # Payload CRC matches the bytes that follow the header.
        payload_off = tail_off + _DELTA_RECORD_HEADER_STRUCT.size
        payload = bytes(
            shared_device["buf"][payload_off : payload_off + payload_len]
        )
        assert (zlib.crc32(payload) & 0xFFFFFFFF) == payload_crc
        assert op_count == 1
    finally:
        core.close()


def test_compaction_flips_mirror_and_resets_tail(shared_device):
    core = RawBlockCore(
        _config(meta_full_checkpoint_max_deltas=1), key_namespace="object"
    )
    try:
        _put_locked(core, "k0", slot_idx=0)
        assert core._checkpoint_once(force=True) is True
        first_mirror = core._active_mirror_idx
        first_seq = core._active_base_seq
        first_crc = core._active_base_crc

        # Append exactly one delta then trigger compaction by exceeding the
        # max_deltas threshold on the next checkpoint.
        _put_locked(core, "k1", slot_idx=1)
        assert core._checkpoint_once(force=True) is True
        assert core._delta_seq == 1

        _put_locked(core, "k2", slot_idx=2)
        assert core._checkpoint_once(force=True) is True
        assert core._meta_seq == 2
        assert core._delta_seq == 0
        assert core._delta_tail_off == 0
        assert core._active_mirror_idx != first_mirror
        assert core._active_base_seq == first_seq + 1
        assert core._active_base_crc != first_crc
    finally:
        core.close()


def test_replay_across_reopen(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        for i in range(5):
            _put_locked(core, f"k{i}", slot_idx=i)
            assert core._checkpoint_once(force=True) is True
        assert core._delta_seq == 4  # one full base + four deltas
    finally:
        core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        assert set(fresh._index.keys()) == {f"k{i}" for i in range(5)}
        assert fresh._delta_seq == 4
        assert fresh._meta_seq == 1
        assert fresh._active_base_seq == 1
    finally:
        fresh.close()


def test_torn_record_stops_replay_at_last_good(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        _put_locked(core, "k1", slot_idx=1)
        core._checkpoint_once(force=True)  # delta_seq=1
        _put_locked(core, "k2", slot_idx=2)
        core._checkpoint_once(force=True)  # delta_seq=2

        # Corrupt the last record's payload so replay stops after delta 1.
        # The last record fills the final block_align block before tail_off;
        # flipping a payload byte breaks payload_crc.
        tail_base = _active_tail_start(core)
        last_rec_off = tail_base + core._delta_tail_off - core.block_align
        payload_byte = last_rec_off + _DELTA_RECORD_HEADER_STRUCT.size + 4
        shared_device["buf"][payload_byte] ^= 0xFF
    finally:
        core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        # Last delta is rejected on CRC; first delta still applied.
        assert "k0" in fresh._index
        assert "k1" in fresh._index
        assert "k2" not in fresh._index
        assert fresh._delta_seq == 1
    finally:
        fresh.close()


def test_stale_tail_after_external_override_is_filtered(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        _put_locked(core, "k1", slot_idx=1)
        core._checkpoint_once(force=True)  # delta exists in mirror 0's tail
    finally:
        core.close()

    # Externally override to a different state that lands a fresh full base.
    overrider = RawBlockCore(cfg, key_namespace="object")
    try:
        overrider.apply_loaded_state(
            {
                "version": 1,
                "device_path": cfg.device_path,
                "slot_bytes": cfg.slot_bytes,
                "meta_total_bytes": cfg.meta_total_bytes,
                "meta_magic": "LMCIDX01",
                "meta_version": 1,
                "next_slot": 0,
                "free_slots": [],
                "entries": {},
            }
        )
        # The override resets the active base. Now write a fresh full base.
        _put_locked(overrider, "kx", slot_idx=0)
        assert overrider._checkpoint_once(force=True) is True
    finally:
        overrider.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        # The old delta records still on disk under the previous base_seq /
        # base_crc must not resurrect "k1".
        assert "k1" not in fresh._index
        assert "kx" in fresh._index
    finally:
        fresh.close()


def test_chain_break_stops_replay(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        _put_locked(core, "k1", slot_idx=1)
        core._checkpoint_once(force=True)
        _put_locked(core, "k2", slot_idx=2)
        core._checkpoint_once(force=True)  # three deltas total

        # Corrupt the prev_record_crc field of the SECOND record so the
        # chain breaks between record 1 and record 2. Layout offsets in
        # the header struct: magic(4) + ver(4) + base_seq(8) + base_crc(4)
        # + delta_seq(8) → prev_record_crc starts at byte 28.
        tail_base = _active_tail_start(core)
        # Walk past first record. Read its total_blocks.
        first_hdr = bytes(
            shared_device["buf"][
                tail_base : tail_base + _DELTA_RECORD_HEADER_STRUCT.size
            ]
        )
        first_total_blocks = _DELTA_RECORD_HEADER_STRUCT.unpack(first_hdr)[10]
        second_off = tail_base + first_total_blocks * core.block_align
        # Flip a bit in prev_record_crc (offset 28..32 within the header).
        for i in range(28, 32):
            shared_device["buf"][second_off + i] ^= 0xFF
    finally:
        core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        assert "k0" in fresh._index
        assert "k1" in fresh._index
        assert "k2" not in fresh._index
        assert fresh._delta_seq == 1
    finally:
        fresh.close()


def test_compaction_on_max_deltas(shared_device):
    cfg = _config(meta_full_checkpoint_max_deltas=3)
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        for i in range(6):
            _put_locked(core, f"k{i}", slot_idx=i)
            core._checkpoint_once(force=True)
        # After 6 mutations: full + 3 deltas (compaction at delta 4) + ...
        # We at least observe that compaction has happened (meta_seq > 1).
        assert core._meta_seq >= 2
    finally:
        core.close()


def test_compaction_on_full_tail(shared_device):
    # A small mirror so the tail fills quickly.
    cfg = _config(
        meta_total_bytes=32 * 1024,  # 16KiB per mirror
        meta_full_checkpoint_max_deltas=1_000_000,
    )
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        compacted = False
        for i in range(200):
            _put_locked(core, f"k{i}", slot_idx=i)
            core._checkpoint_once(force=True)
            if core._meta_seq >= 2:
                compacted = True
                break
        assert compacted, "expected at least one compaction once tail filled"
    finally:
        core.close()


def test_close_flushes_pending_delta(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    _put_locked(core, "k0", slot_idx=0)
    core._checkpoint_once(force=True)
    _put_locked(core, "k1", slot_idx=1)
    # No explicit checkpoint -- close() must flush.
    core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        assert "k0" in fresh._index
        assert "k1" in fresh._index
    finally:
        fresh.close()


def test_higher_seq_mirror_wins_on_load(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        # Force two compactions so both mirrors hold valid (different) bases.
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        # Set a tight max_deltas to force compaction on next checkpoint.
        core.meta_full_checkpoint_max_deltas = 0
        _put_locked(core, "k1", slot_idx=1)
        core._checkpoint_once(force=True)  # compaction → mirror flip
        assert core._meta_seq == 2
        winning_mirror = core._active_mirror_idx
    finally:
        core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        assert fresh._meta_seq == 2
        assert fresh._active_mirror_idx == winning_mirror
        assert "k0" in fresh._index
        assert "k1" in fresh._index
    finally:
        fresh.close()


def test_base_header_decodes_under_single_format(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        mirror_off = core._meta_layout.base_copy_offsets[core._active_mirror_idx]
        hdr = bytes(
            shared_device["buf"][mirror_off : mirror_off + _META_HEADER_STRUCT.size]
        )
        magic, version, seq, payload_len, crc = _META_HEADER_STRUCT.unpack(hdr)
        assert magic == cfg.meta_magic
        assert version == cfg.meta_version
        assert seq == 1
        assert payload_len > 0
        assert crc == core._active_base_crc
    finally:
        core.close()


def test_delete_replays_through_delta(shared_device):
    cfg = _config()
    core = RawBlockCore(cfg, key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        _put_locked(core, "k1", slot_idx=1)
        core._checkpoint_once(force=True)
        _delete_locked(core, "k0")
        core._checkpoint_once(force=True)
    finally:
        core.close()

    fresh = RawBlockCore(cfg, key_namespace="object")
    try:
        assert "k0" not in fresh._index
        assert "k1" in fresh._index
    finally:
        fresh.close()


def test_status_exposes_tail_metrics(shared_device):
    core = RawBlockCore(_config(), key_namespace="object")
    try:
        _put_locked(core, "k0", slot_idx=0)
        core._checkpoint_once(force=True)
        status = core.report_status()
        assert "delta_seq" in status
        assert "delta_tail_off" in status
        assert "delta_tail_capacity" in status
        assert "active_base_seq" in status
        assert "active_mirror_idx" in status
        assert status["active_base_seq"] == 1
        assert status["delta_tail_capacity"] > 0
    finally:
        core.close()
