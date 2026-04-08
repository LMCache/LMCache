# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.gating import (
    NullStorageGate,
    SsdStorageGate,
    WriteVetoReason,
    build_storage_gate_from_extra,
)

# Local
from ..utils import dumb_cache_engine_key


def test_null_gate_admits_all():
    gate = NullStorageGate()
    key = dumb_cache_engine_key(1)
    assert gate.on_lookup(key)
    assert gate.on_read(key)
    assert gate.on_write(key, 1)
    assert gate.on_delete(key)
    assert gate.explain_write_veto(key, 1) is None


def test_ssd_gate_length_veto():
    gate = SsdStorageGate(min_size_bytes=100, min_read_count_before_write=0)
    key = dumb_cache_engine_key(1)
    assert gate.explain_write_veto(key, 50) == WriteVetoReason.LENGTH
    assert gate.explain_write_veto(key, 100) is None
    assert gate.explain_write_veto(key, 200) is None


def test_ssd_gate_frequency_veto():
    gate = SsdStorageGate(min_size_bytes=0, min_read_count_before_write=3)
    key = dumb_cache_engine_key(1)
    assert gate.explain_write_veto(key, 1000) == WriteVetoReason.FREQUENCY
    gate.record_read(key)
    assert gate.explain_write_veto(key, 1000) == WriteVetoReason.FREQUENCY
    gate.record_read(key)
    assert gate.explain_write_veto(key, 1000) == WriteVetoReason.FREQUENCY
    gate.record_read(key)
    assert gate.explain_write_veto(key, 1000) is None


def test_ssd_gate_record_write_resets_read_counter():
    gate = SsdStorageGate(min_size_bytes=0, min_read_count_before_write=2)
    key = dumb_cache_engine_key(1)
    gate.record_read(key)
    gate.record_read(key)
    assert gate.explain_write_veto(key, 500) is None
    gate.record_write(key, new_admission=True)
    assert gate.explain_write_veto(key, 500) == WriteVetoReason.FREQUENCY


def test_ssd_gate_record_write_refresh_does_not_bump_write_counter():
    gate = SsdStorageGate(min_size_bytes=0, min_read_count_before_write=0)
    key = dumb_cache_engine_key(1)
    gate.record_write(key, new_admission=True)
    gate.record_write(key, new_admission=False)
    h = key.chunk_hash
    assert gate._write_counts.get(h) == 1


def test_ssd_gate_lookup_trim():
    gate = SsdStorageGate(
        min_size_bytes=0,
        min_read_count_before_write=0,
        max_tracked_chunk_hashes=2,
    )
    gate.record_lookup(dumb_cache_engine_key(0))
    gate.record_lookup(dumb_cache_engine_key(1))
    assert len(gate._lookup_counts) == 2
    gate.record_lookup(dumb_cache_engine_key(2))
    assert len(gate._lookup_counts) == 1


def test_build_gate_from_extra_empty():
    assert isinstance(build_storage_gate_from_extra({}), NullStorageGate)


def test_build_gate_from_extra_ssd():
    g = build_storage_gate_from_extra({"ssd_gate_min_size_bytes": 64})
    assert isinstance(g, SsdStorageGate)
