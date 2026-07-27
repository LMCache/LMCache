# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    encode_object_key,
    resolve_raw_block_device_id,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def test_raw_block_core_store_load_and_exists(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 2048,
            bytes([3]) * 3072,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.exists_many([spec.encoded for spec in specs]) == [
            True,
            True,
            True,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_duplicate_put_keeps_original_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(11))
        original = b"original"
        duplicate = b"mutated!"

        first_result = core.put_many([spec], [make_memory_obj(original)])
        duplicate_result = core.put_many([spec], [make_memory_obj(duplicate)])

        assert first_result.results == [True]
        assert first_result.stored_keys == [spec.encoded]
        assert duplicate_result.results == [True]
        assert duplicate_result.stored_keys == []

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_delete_and_missing_load(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(21))
        missing = encode_object_key(make_object_key(22))

        put_result = core.put_many([existing], [make_memory_obj(b"delete-me")])
        assert put_result.results == [True]
        assert core.contains_key(existing.encoded) is True

        assert core.delete_many([existing.encoded, missing.encoded]) == [True, False]
        assert core.exists_many([existing.encoded, missing.encoded]) == [False, False]

        loaded = make_empty_memory_obj(len(b"delete-me"))
        assert core.load_many_into([existing.encoded], [loaded]) == [False]
    finally:
        core.close()


def test_raw_block_core_requires_device_id_for_unsupported_checkpoint_path(tmp_path):
    path = tmp_path / "missing_raw_block.bin"
    config = make_raw_block_core_config(path)

    with pytest.raises(ValueError, match="requires a stable device_id"):
        RawBlockCore(config, key_namespace="object")


def test_raw_block_core_recovers_checkpoint_from_temp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(31))
    payload = b"recoverable-raw-block-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is True
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()


def test_raw_block_device_id_resolves_nvme_namespace_paths(monkeypatch):
    observed_paths: list[str] = []

    def fake_read_sysfs_text(path: str) -> str:
        observed_paths.append(path)
        return "nvme.11111111-2222-3333-4444-555555555555"

    monkeypatch.setattr(
        "lmcache.v1.storage_backend.raw_block.core._read_sysfs_text",
        fake_read_sysfs_text,
    )

    assert (
        resolve_raw_block_device_id("/dev/nvme0n1")
        == "nvme:nvme.11111111-2222-3333-4444-555555555555"
    )
    assert observed_paths[-1] == "/sys/block/nvme0n1/wwid"

    assert (
        resolve_raw_block_device_id("/dev/ng0n1")
        == "nvme:nvme.11111111-2222-3333-4444-555555555555"
    )
    assert observed_paths[-1] == "/sys/block/nvme0n1/wwid"

    assert (
        resolve_raw_block_device_id("/dev/nvme0n1p1")
        == "nvme:nvme.11111111-2222-3333-4444-555555555555:p1"
    )
    assert observed_paths[-1] == "/sys/block/nvme0n1/wwid"

    assert (
        resolve_raw_block_device_id("/dev/nvme0n1p2")
        == "nvme:nvme.11111111-2222-3333-4444-555555555555:p2"
    )
    assert observed_paths[-1] == "/sys/block/nvme0n1/wwid"


def test_raw_block_device_id_resolves_regular_file_identity(tmp_path):
    path = make_raw_block_file(tmp_path)
    st = path.stat()

    assert (
        resolve_raw_block_device_id(str(path))
        == f"file:{st.st_dev}:{st.st_ino}:{st.st_size}"
    )


def test_raw_block_core_rebuilds_missing_free_slots_from_checkpoint(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    existing = encode_object_key(make_object_key(41))
    recovered = encode_object_key(make_object_key(42))
    existing_payload = b"already-committed"
    recovered_payload = b"recovered-hole"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([existing], [make_memory_obj(existing_payload)])
        assert put_result.results == [True]

        committed_offset = core.entry_offset(existing.encoded)
        assert committed_offset == core.data_base_offset()

        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_id": core.device_id,
                "capacity_bytes": core.capacity_bytes,
                "block_align": core.block_align,
                "header_bytes": core.header_bytes,
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "data_base_offset": core.data_base_offset(),
                # Simulates a checkpoint taken after slot 1 was reserved but
                # before its key was committed into the metadata index.
                "next_slot": 2,
                # Older checkpoints include free_slots; keep this empty to
                # verify recovery ignores stale/missing free-list data and
                # reconstructs reusable slots from entries plus next_slot.
                "free_slots": [],
                "entries": {
                    existing.encoded: {
                        "offset": committed_offset,
                        "size": len(existing_payload),
                        "shape": [len(existing_payload)],
                        "dtype": "uint8",
                        "fmt": "BINARY",
                        "cached_positions": None,
                    }
                },
            }
        )

        assert applied is True
        status = core.report_status()
        assert status["next_slot"] == 2
        assert status["free_slot_count"] == 1

        put_recovered = core.put_many([recovered], [make_memory_obj(recovered_payload)])

        assert put_recovered.results == [True]
        assert core.entry_offset(recovered.encoded) == (
            core.data_base_offset() + core.slot_bytes
        )
        assert core.report_status()["next_slot"] == 2
    finally:
        core.close()


def test_raw_block_core_accepts_checkpoint_with_matching_device_id(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        core.device_id = "nvme:nvme.11111111-2222-3333-4444-555555555555"
        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_path": "/dev/ng0n1",
                "device_id": "nvme:nvme.11111111-2222-3333-4444-555555555555",
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "next_slot": 0,
                "free_slots": [],
                "entries": {},
            }
        )

        assert applied is True
    finally:
        core.close()


def test_raw_block_core_rejects_checkpoint_missing_device_id(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_path": "/dev/ng0n1",
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "next_slot": 0,
                "free_slots": [],
                "entries": {},
            }
        )

        assert applied is False
    finally:
        core.close()


def test_raw_block_core_allows_legacy_checkpoint_missing_device_id(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = replace(
        make_raw_block_core_config(path),
        allow_legacy_checkpoint_without_device_id=True,
    )
    core = RawBlockCore(config, key_namespace="object")

    try:
        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_path": "/dev/ng0n1",
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "next_slot": 0,
                "free_slots": [],
                "entries": {},
            }
        )

        assert applied is True
    finally:
        core.close()


def test_raw_block_core_rejects_checkpoint_with_different_device_id(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        core.device_id = "nvme:nvme.11111111-2222-3333-4444-555555555555"
        applied = core.apply_loaded_state(
            {
                "version": 1,
                "device_path": str(path),
                "device_id": "nvme:nvme.aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "slot_bytes": core.slot_bytes,
                "meta_total_bytes": core.meta_total_bytes,
                "meta_magic": core.meta_magic_text,
                "meta_version": core.meta_version,
                "next_slot": 0,
                "free_slots": [],
                "entries": {},
            }
        )

        assert applied is False
    finally:
        core.close()
