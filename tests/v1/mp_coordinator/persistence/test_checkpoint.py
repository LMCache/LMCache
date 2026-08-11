# SPDX-License-Identifier: Apache-2.0
"""Tests for coordinator checkpoint load/write: the file lifecycle, the
views rebuilt from it, pins, and the failure paths (missing, corrupt,
unwritable)."""

# Standard
from pathlib import Path
import json

# Third Party
from fastapi.testclient import TestClient
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate, IngestResult
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.persistence import checkpoint
from lmcache.v1.mp_coordinator.persistence.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from lmcache.v1.mp_coordinator.persistence.snapshot_codec import read_snapshot
from lmcache.v1.mp_coordinator.persistence.store import LocalArtifactStore


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _batch(
    seq: int = 1,
    keys: list[ObjectKey] | None = None,
    tier: Tier = Tier.L1,
    backend: str = "dram",
    token_ids: list[int] | None = None,
    token_offset: int = 0,
    ts: float = 0.0,
    incarnation: int = 1,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=incarnation,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=tier,
        backend=backend,
        ts=ts,
        entries=[
            CacheEventEntry(
                key=k.to_encoded_object_key(),
                size_bytes=1024,
                token_ids=token_ids or [],
                token_offset=token_offset,
            )
            for k in (keys or [_key(1)])
        ],
    )


def _gate() -> EventGate:
    """A gate whose cursors ride along with the directory snapshot."""
    return EventGate(CacheEventBroadcaster())


def _populated() -> KeyDirectory:
    directory = KeyDirectory()
    directory.consume(_batch(seq=1, keys=[_key(1), _key(2)], token_ids=[7, 8]))
    directory.consume(_batch(seq=2, tier=Tier.L2, backend="fs", keys=[_key(1)]))
    return directory


# -- Happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_then_load_round_trips(tmp_path: Path):
    path = tmp_path / "directory.snapshot"
    directory = _populated()

    await save_checkpoint(LocalArtifactStore(path), directory, _gate())
    restored = KeyDirectory()
    load_checkpoint(
        LocalArtifactStore(path), restored, _gate(), CacheEventBroadcaster()
    )

    assert restored.stats().num_keys == directory.stats().num_keys
    assert restored.stats().num_placements == directory.stats().num_placements
    assert restored.lookup([_key(1)]) == directory.lookup([_key(1)])
    assert restored.get_token_ids([_key(1).chunk_hash]) == [(7, 8)]


@pytest.mark.asyncio
async def test_write_creates_missing_parent_directories(tmp_path: Path):
    path = tmp_path / "state" / "nested" / "directory.snapshot"

    await save_checkpoint(LocalArtifactStore(path), _populated(), _gate())

    assert path.is_file()


@pytest.mark.asyncio
async def test_write_leaves_no_temporary_file(tmp_path: Path):
    path = tmp_path / "directory.snapshot"

    await save_checkpoint(LocalArtifactStore(path), _populated(), _gate())

    assert [p.name for p in tmp_path.iterdir()] == ["directory.snapshot"]


@pytest.mark.asyncio
async def test_write_replaces_an_existing_checkpoint(tmp_path: Path):
    path = tmp_path / "directory.snapshot"
    await save_checkpoint(LocalArtifactStore(path), _populated(), _gate())
    grown = _populated()
    grown.consume(_batch(seq=3, keys=[_key(3)]))

    await save_checkpoint(LocalArtifactStore(path), grown, _gate())

    restored = KeyDirectory()
    load_checkpoint(
        LocalArtifactStore(path), restored, _gate(), CacheEventBroadcaster()
    )
    assert restored.stats().num_keys == 3


@pytest.mark.asyncio
async def test_blend_lookup_survives_a_checkpoint(tmp_path: Path):
    path = tmp_path / "directory.snapshot"
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=2, probe_stride=1)
    directory.consume(_batch(seq=1, token_ids=[7, 8], token_offset=512))
    await save_checkpoint(LocalArtifactStore(path), directory, _gate())

    restored = KeyDirectory()
    restored.enable_blend_lookup(chunk_size=2, probe_stride=1)
    load_checkpoint(
        LocalArtifactStore(path), restored, _gate(), CacheEventBroadcaster()
    )

    (match,) = restored.blend_match(np.asarray([7, 8], dtype=np.uint64))
    assert match.old_st == 512


# -- Failure paths -----------------------------------------------------------


def test_load_of_a_missing_file_starts_cold(tmp_path: Path):
    directory = KeyDirectory()

    load_checkpoint(
        LocalArtifactStore(tmp_path / "absent.snapshot"),
        directory,
        _gate(),
        CacheEventBroadcaster(),
    )

    assert directory.stats().num_keys == 0


def test_load_of_a_corrupt_file_starts_cold(tmp_path: Path, logged):
    path = tmp_path / "directory.snapshot"
    path.write_bytes(b"not a snapshot at all")
    directory = KeyDirectory()

    messages = logged(checkpoint.logger)
    load_checkpoint(
        LocalArtifactStore(path), directory, _gate(), CacheEventBroadcaster()
    )

    assert directory.stats().num_keys == 0
    assert any("Ignoring coordinator checkpoint" in m for m in messages)


@pytest.mark.asyncio
async def test_load_of_a_truncated_file_starts_cold(tmp_path: Path):
    path = tmp_path / "directory.snapshot"
    full = tmp_path / "full.snapshot"
    # Build a valid checkpoint, then keep only a prefix of it.
    await save_checkpoint(LocalArtifactStore(full), _populated(), _gate())
    path.write_bytes(full.read_bytes()[:20])

    restored = KeyDirectory()
    load_checkpoint(
        LocalArtifactStore(path), restored, _gate(), CacheEventBroadcaster()
    )

    assert restored.stats().num_keys == 0


@pytest.mark.asyncio
async def test_load_into_a_populated_directory_is_rejected(tmp_path: Path):
    path = tmp_path / "directory.snapshot"
    await save_checkpoint(LocalArtifactStore(path), _populated(), _gate())
    occupied = KeyDirectory()
    occupied.consume(_batch(seq=1, keys=[_key(9)]))

    with pytest.raises(ValueError, match="empty directory"):
        load_checkpoint(
            LocalArtifactStore(path), occupied, _gate(), CacheEventBroadcaster()
        )


@pytest.mark.asyncio
async def test_write_to_an_unwritable_path_is_logged_not_raised(tmp_path: Path, logged):
    # A file where the parent directory must go: mkdir cannot succeed.
    blocker = tmp_path / "blocked"
    blocker.write_bytes(b"")
    path = blocker / "directory.snapshot"

    messages = logged(checkpoint.logger)
    await save_checkpoint(LocalArtifactStore(path), _populated(), _gate())

    assert any("Failed to write coordinator checkpoint" in m for m in messages)


# -- Derived views -----------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_rebuilds_the_l2_usage_ledger(tmp_path: Path):
    """A restored directory with an empty usage view would leave quota
    enforcement blind to everything stored before the restart."""
    path = tmp_path / "directory.snapshot"
    directory = KeyDirectory()
    salted = ObjectKey(
        chunk_hash=b"\x01" * 4, model_name="m", kv_rank=0, cache_salt="tenant-a"
    )
    directory.consume(_batch(seq=1, tier=Tier.L2, backend="fs", keys=[salted]))
    await save_checkpoint(LocalArtifactStore(path), directory, _gate())

    eviction = FleetEvictionController()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(eviction)
    load_checkpoint(LocalArtifactStore(path), KeyDirectory(), _gate(), broadcaster)

    usage = eviction.usage
    assert usage.get("tenant-a") == 1024
    assert usage.get_total() == 1024
    assert usage.get_key_size(salted) == 1024


@pytest.mark.asyncio
async def test_restore_reproduces_the_policy_order_exactly(tmp_path: Path):
    """The policy rides in its own section, so its order survives verbatim —
    including an order the placements' timestamps could not imply."""
    path = tmp_path / "directory.snapshot"
    directory = KeyDirectory()
    live = FleetEvictionController()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(directory)
    broadcaster.register_consumer(live)
    for index, hash_byte in enumerate((1, 2, 3)):
        # Every batch shares one ts, as a real flush does, so nothing in the
        # directory records which key was touched first.
        broadcaster.broadcast(
            _batch(
                seq=index + 1,
                tier=Tier.L2,
                backend="fs",
                keys=[_key(hash_byte)],
                ts=7.0,
            )
        )
    # Touch the first-stored key: the live LRU now orders 2, 3, 1.
    broadcaster.broadcast(
        CacheEventBatch(
            instance_id="node-a",
            incarnation=1,
            seq=4,
            event_type=CacheEventType.ACCESS,
            tier=Tier.L2,
            backend="",
            ts=7.0,
            entries=[CacheEventEntry(key=_key(1).to_encoded_object_key())],
        )
    )
    await save_checkpoint(LocalArtifactStore(path), directory, _gate(), [live.policy])

    restored = FleetEvictionController()
    restored.quota.set_quota("", 0)  # A zero quota evicts the salt, in LRU order.
    restored_broadcaster = CacheEventBroadcaster()
    restored_broadcaster.register_consumer(restored)
    load_checkpoint(
        LocalArtifactStore(path),
        KeyDirectory(),
        _gate(),
        restored_broadcaster,
        [restored.policy],
    )

    assert restored.compute_eviction_plan() == {"": [_key(2), _key(3), _key(1)]}


# -- Coordinator wiring ------------------------------------------------------


def _coordinator(snapshot_path: str, metadata_path: str = "") -> TestClient:
    """A coordinator whose only background work is persistence."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        snapshot_path=snapshot_path,
        metadata_path=metadata_path,
        snapshot_interval=0.0,
    )
    return TestClient(create_app(config))


def _store_one_key(client: TestClient) -> None:
    resp = client.post(
        "/events",
        json={
            "batches": [
                {
                    "instance_id": "node-a",
                    "incarnation": 1,
                    "seq": 1,
                    "event_type": "store",
                    "tier": "l2",
                    "backend": "fs",
                    "entries": [
                        {
                            "key": {
                                "chunk_hash_hex": "aa",
                                "model_name": "m",
                                "kv_rank": 0,
                            },
                            "size_bytes": 1024,
                        }
                    ],
                }
            ]
        },
    )
    assert resp.status_code == 200


def test_a_restarted_coordinator_recovers_its_directory(tmp_path: Path):
    """The whole point: L2 placements outlive the coordinator process."""
    path = tmp_path / "directory.snapshot"

    with _coordinator(str(path)) as client:
        _store_one_key(client)
        assert client.get("/directory/stats").json()["num_keys"] == 1

    # Shutdown checkpointed; a fresh process picks it back up.
    assert path.is_file()
    with _coordinator(str(path)) as restarted:
        stats = restarted.get("/directory/stats").json()
        assert stats["num_keys"] == 1
        assert stats["num_placements"] == 1


def test_an_unconfigured_coordinator_writes_nothing(tmp_path: Path):
    with _coordinator("") as client:
        _store_one_key(client)

    assert list(tmp_path.iterdir()) == []


def test_both_artifacts_survive_a_restart(tmp_path: Path):
    """The two halves are stored separately but must come back together:
    the directory says what is cached, the metadata state says what may
    not be evicted."""
    snapshot = tmp_path / "directory.snapshot"
    metadata = tmp_path / "metadata.json"

    with _coordinator(str(snapshot), str(metadata)) as client:
        _store_one_key(client)
        assert (
            client.put("/quota/config", json={"default_limit_gb": 2}).status_code == 200
        )

    assert snapshot.is_file() and metadata.is_file()
    with _coordinator(str(snapshot), str(metadata)) as restarted:
        assert restarted.get("/directory/stats").json()["num_keys"] == 1
        assert restarted.get("/quota/config").json()["default_limit_gb"] == 2


def test_each_component_is_routed_to_the_artifact_its_type_names(tmp_path: Path):
    """The app wires durability by ``persistence_type`` alone, so this pins the
    routing rather than the flags: derived state must reach the checkpoint
    and operator intent the document, with neither carrying the other."""
    snapshot = tmp_path / "directory.snapshot"
    metadata = tmp_path / "metadata.json"

    with _coordinator(str(snapshot), str(metadata)) as client:
        _store_one_key(client)
        assert (
            client.put("/quota/config", json={"default_limit_gb": 2}).status_code == 200
        )

    with snapshot.open("rb") as f:
        _, _, sections = read_snapshot(f)
    document = json.loads(metadata.read_text())["components"]

    assert set(sections) == {"lru_order"}
    assert set(document) == {"pins", "quotas"}


@pytest.mark.asyncio
async def test_restored_cursors_let_a_restarted_emitter_be_fenced(tmp_path: Path):
    """The reason cursors ride in the snapshot: fencing compares against a
    prior incarnation, so without them a restored L1 slice is unfenceable."""
    path = tmp_path / "directory.snapshot"
    directory = KeyDirectory()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(directory)
    gate = EventGate(broadcaster)
    gate.ingest(_batch(seq=1, keys=[_key(1)]))  # L1 under incarnation 1
    await save_checkpoint(LocalArtifactStore(path), directory, gate)

    # A fresh coordinator restores, then the emitter comes back restarted.
    restored = KeyDirectory()
    restored_broadcaster = CacheEventBroadcaster()
    restored_broadcaster.register_consumer(restored)
    restored_gate = EventGate(restored_broadcaster)
    load_checkpoint(
        LocalArtifactStore(path), restored, restored_gate, restored_broadcaster
    )
    assert restored.lookup([_key(1)])[0], "the L1 placement should be restored"

    restored_gate.ingest(_batch(incarnation=2, seq=1, keys=[_key(9)]))

    # Incarnation 2 fences incarnation 1's L1 facts.
    assert restored.lookup([_key(1)]) == [[]]


@pytest.mark.asyncio
async def test_a_batch_admitted_mid_capture_survives_the_restart(tmp_path: Path):
    """No admitted batch may end up both absent from the checkpoint and
    rejected on redelivery.

    Capture is not atomic across the gate and the directory, so a batch
    can slip between the two reads. Whichever way it lands, the emitter's
    retry has to be able to put it back -- which only holds while the
    stored cursors are no further ahead than the stored directory.
    """
    store = LocalArtifactStore(tmp_path / "directory.snapshot")
    directory = KeyDirectory()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(directory)
    gate = EventGate(broadcaster)
    gate.ingest(_batch(seq=1, keys=[_key(1)]))

    # A batch lands after the directory has been read: the worst case for
    # the cursors, since they are written from the same capture.
    real_snapshot = directory.snapshot

    def snapshot_then_admit():
        captured = real_snapshot()
        gate.ingest(_batch(seq=2, keys=[_key(2)]))
        return captured

    directory.snapshot = snapshot_then_admit  # type: ignore[method-assign]
    await save_checkpoint(store, directory, gate)
    directory.snapshot = real_snapshot  # type: ignore[method-assign]

    # Restart, then let the emitter retry the batch the checkpoint missed.
    fresh_directory = KeyDirectory()
    fresh_broadcaster = CacheEventBroadcaster()
    fresh_broadcaster.register_consumer(fresh_directory)
    fresh_gate = EventGate(fresh_broadcaster)
    load_checkpoint(store, fresh_directory, fresh_gate, fresh_broadcaster)
    assert fresh_directory.lookup([_key(2)]) == [[]], (
        "the checkpoint should have missed this batch"
    )

    result = fresh_gate.ingest(_batch(seq=2, keys=[_key(2)]))

    assert result is IngestResult.ADMITTED, (
        "the retry was skipped as already-processed, so this placement is "
        "lost for good -- the stored cursors ran ahead of the stored directory"
    )
    assert fresh_directory.lookup([_key(2)])[0], (
        "the retry was admitted but its placement never reached the directory"
    )
