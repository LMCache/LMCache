# SPDX-License-Identifier: Apache-2.0
"""Tests for key-directory snapshot capture, encoding, and restore."""

# Standard
from collections.abc import Mapping
import io

# Third Party
import msgspec
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_gate import InstanceStreamStats
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.persistence.snapshot_codec import (
    SnapshotFormatError,
    read_snapshot,
    write_snapshot,
)


def _chash(hash_byte: int) -> bytes:
    return bytes([hash_byte]) * 4


def _key(hash_byte: int, kv_rank: int = 0, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=_chash(hash_byte),
        model_name="m",
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


def _batch(
    instance_id: str = "node-a",
    incarnation: int = 1,
    seq: int = 1,
    event_type: CacheEventType = CacheEventType.STORE,
    tier: Tier = Tier.L1,
    backend: str = "dram",
    keys: list[ObjectKey] | None = None,
    size_bytes: int = 1024,
    shared: bool = False,
    ts: float = 0.0,
    token_ids: list[int] | None = None,
    token_offset: int = 0,
) -> CacheEventBatch:
    entries = [
        CacheEventEntry(
            key=k.to_encoded_object_key(),
            size_bytes=size_bytes,
            token_ids=token_ids or [],
            token_offset=token_offset,
        )
        for k in (keys or [_key(0xAA)])
    ]
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        entries=entries,
        shared=shared,
        ts=ts,
    )


def _populated() -> KeyDirectory:
    """A directory exercising every field the format carries."""
    directory = KeyDirectory()
    directory.consume(
        _batch(seq=1, keys=[_key(1), _key(2)], token_ids=[7, 8], token_offset=512)
    )
    directory.consume(
        _batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)], size_bytes=4096)
    )
    directory.consume(
        _batch(
            seq=3,
            tier=Tier.L2,
            backend="s3",
            shared=True,
            keys=[_key(3)],
            size_bytes=64,
            ts=99.5,
        )
    )
    # A second reporter, a salted key, a non-zero rank, and a seq gap.
    directory.consume(
        _batch(instance_id="node-b", seq=1, keys=[_key(4, kv_rank=3, cache_salt="u1")])
    )
    directory.consume(_batch(instance_id="node-b", seq=7, keys=[_key(5)]))
    return directory


def _cursors(**streams: tuple[int, int, bool]) -> dict[str, InstanceStreamStats]:
    """Gate cursors as ``instance_id=(incarnation, last_seq, gap_detected)``."""
    return {
        instance_id: InstanceStreamStats(
            incarnation=incarnation, last_seq=last_seq, gap_detected=gap
        )
        for instance_id, (incarnation, last_seq, gap) in streams.items()
    }


def _encode(
    directory: KeyDirectory,
    cursors: dict[str, InstanceStreamStats] | None = None,
    sections: dict[str, Mapping[str, object]] | None = None,
) -> io.BytesIO:
    """Encode a directory, and optionally cursors and component sections."""
    stream = io.BytesIO()
    write_snapshot(directory.snapshot(), cursors or {}, sections or {}, stream)
    stream.seek(0)
    return stream


def _round_trip(directory: KeyDirectory) -> KeyDirectory:
    """Encode ``directory``'s snapshot and restore it into a fresh one."""
    snapshot, _, _ = read_snapshot(_encode(directory))
    restored = KeyDirectory()
    restored.restore(snapshot)
    return restored


def _all_keys(directory: KeyDirectory) -> dict[ObjectKey, list]:
    total, page = directory.list_keys(limit=1000)
    assert total == len(page)
    return page


# -- Round trip --------------------------------------------------------------


def test_round_trip_preserves_placements():
    directory = _populated()

    restored = _round_trip(directory)

    assert _all_keys(restored) == _all_keys(directory)
    for key in (_key(1), _key(2), _key(3), _key(4, kv_rank=3, cache_salt="u1")):
        assert restored.lookup([key]) == directory.lookup([key])


def test_round_trip_preserves_counts_and_l1_reverse_index():
    directory = _populated()

    restored = _round_trip(directory)

    before = directory.stats()
    after = restored.stats()
    assert after.num_keys == before.num_keys
    assert after.num_placements == before.num_placements
    assert after.l1_keys_by_instance == before.l1_keys_by_instance


def test_round_trip_preserves_gate_cursors():
    """Restored placements are only fenceable if the cursors come back
    with them, so the codec carries both."""
    cursors = _cursors(**{"node-a": (1, 3, False), "node-b": (7, 12, True)})

    _, restored, _ = read_snapshot(_encode(_populated(), cursors))

    assert restored == cursors
    # The gap the gate saw is carried across, not laundered.
    assert restored["node-b"].gap_detected is True


def test_round_trip_preserves_token_bindings():
    directory = _populated()

    restored = _round_trip(directory)

    assert restored.get_token_ids([_chash(1), _chash(2)]) == [(7, 8), (7, 8)]
    # A chunk that never carried content restores as a content-free binding.
    assert restored.get_token_ids([_chash(3)]) == [()]


def test_round_trip_of_empty_directory():
    restored = _round_trip(KeyDirectory())

    assert restored.stats().num_keys == 0
    assert restored.stats().l1_keys_by_instance == {}


def test_restored_directory_accepts_further_events():
    directory = _populated()
    restored = _round_trip(directory)

    restored.consume(_batch(seq=4, keys=[_key(6)]))

    assert len(restored.lookup([_key(6)])[0]) == 1


def test_restored_l1_key_sets_drive_fencing():
    """Fencing needs the per-instance L1 key set, so the snapshot must
    carry it rather than re-deriving it from placements."""
    restored = _round_trip(_populated())

    restored.fence_instance("node-a")

    # L1 placements go; the L2 ones the same instance reported survive.
    assert restored.lookup([_key(2)]) == [[]]
    [l2_only] = restored.lookup([_key(1)])
    assert [p.tier for p in l2_only] == [Tier.L2]


# -- Blend index -------------------------------------------------------------


def test_blend_matching_works_after_restore():
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[7, 8], token_offset=512))

    snapshot, _, _ = read_snapshot(_encode(directory))
    restored = KeyDirectory()
    restored.enable_blend_lookup(chunk_size=2, probe_stride=1)
    restored.restore(snapshot)

    (match,) = restored.blend_match(np.asarray([7, 8], dtype=np.uint64))
    assert match.old_st == 512
    assert restored.blend_stats().num_chunks == 1


def test_blend_index_stays_empty_when_lookup_is_disabled():
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)
    directory.consume(_batch(seq=1, keys=[_key(1)], token_ids=[7, 8], token_offset=512))

    # Restoring without enabling blend lookup keeps the content, not the index.
    restored = _round_trip(directory)

    assert restored.get_token_ids([_chash(1)]) == [(7, 8)]
    assert restored.blend_match(np.asarray([7, 8], dtype=np.uint64)) == []


# -- Restore guards ----------------------------------------------------------


def test_restore_rejects_a_non_empty_directory():
    directory = _populated()
    snapshot = directory.snapshot()
    occupied = KeyDirectory()
    occupied.consume(_batch(seq=1, keys=[_key(9)]))

    with pytest.raises(ValueError, match="empty directory"):
        occupied.restore(snapshot)


# -- Malformed streams -------------------------------------------------------


def test_read_rejects_a_foreign_stream():
    with pytest.raises(SnapshotFormatError, match="not a directory snapshot"):
        read_snapshot(io.BytesIO(b"\x00" * 64))


def test_read_rejects_an_unsupported_version():
    body = bytearray(_encode(_populated()).getvalue())
    body[8] = 99  # the version word follows the 8-byte magic
    with pytest.raises(SnapshotFormatError, match="unsupported snapshot version 99"):
        read_snapshot(io.BytesIO(bytes(body)))


def test_read_rejects_a_truncated_stream():
    body = _encode(_populated()).getvalue()

    with pytest.raises(SnapshotFormatError, match="truncated snapshot"):
        read_snapshot(io.BytesIO(body[: len(body) // 2]))


def test_read_rejects_an_out_of_range_key_index():
    """The writer cannot emit one — it derives indices from the same key
    list — so this guards a corrupted file."""
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1)]))
    body = _encode(directory).getvalue()
    # Prefix is magic(8) + version(4) + structure length(4).
    length = int.from_bytes(body[12:16], "little")
    structure = msgspec.msgpack.decode(body[16 : 16 + length])
    structure["l1_keys_by_instance"]["node-a"] = [5]
    payload = msgspec.msgpack.encode(structure)
    corrupted = (
        body[:12] + len(payload).to_bytes(4, "little") + payload + body[16 + length :]
    )

    with pytest.raises(SnapshotFormatError, match="references key index 5"):
        read_snapshot(io.BytesIO(corrupted))
