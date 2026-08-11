# SPDX-License-Identifier: Apache-2.0
"""Tests for durable metadata state: pins and quotas round-tripping,
the enforcement they restore, and the failure paths."""

# Standard
from dataclasses import asdict
from pathlib import Path
import json

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, PersistenceType, Tier
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers import ControllerRegistry
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.persistence import metadata_persister
from lmcache.v1.mp_coordinator.persistence.metadata_persister import MetadataPersister
from lmcache.v1.mp_coordinator.persistence.store import (
    LocalArtifactStore,
    NullArtifactStore,
)


def _key(hash_byte: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([hash_byte]) * 4,
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _l2_store(eviction: FleetEvictionController, key: ObjectKey) -> None:
    """Feed one L2 store event through the controller, as the gate does."""
    batch = CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=1024)],
    )
    eviction.consume(batch)


def _managers() -> tuple[QuotaManager, FleetEvictionController]:
    """The controller and the quota registry it owns — the two durable
    components the persister carries."""
    controller = FleetEvictionController()
    return controller.quota, controller


def _captured_pins(*pins: tuple[ObjectKey, int]) -> dict[str, object]:
    """The durable form ``FleetEvictionController.capture`` produces."""
    return {
        "entries": [
            {"key": asdict(key.to_encoded_object_key()), "count": count}
            for key, count in pins
        ]
    }


def _persister(store, quota: QuotaManager, eviction: FleetEvictionController):
    """A persister wired the way ``create_app`` wires it."""
    persister = MetadataPersister(store)
    persister.register(eviction)
    persister.register(quota)
    return persister


# -- Round trip --------------------------------------------------------------


@pytest.mark.asyncio
async def test_pins_and_quotas_round_trip(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "metadata.json")
    quota, eviction = _managers()
    quota.set_quota("tenant-a", 4096)
    quota.set_default_limit_bytes(1024)
    eviction.pin([_key(1), _key(1), _key(2)])

    await _persister(store, quota, eviction).save()
    restored_quota, restored_eviction = _managers()
    _persister(store, restored_quota, restored_eviction).load()

    assert restored_eviction.capture() == _captured_pins((_key(1), 2), (_key(2), 1))
    assert restored_quota.get_limit_bytes("tenant-a") == 4096
    assert restored_quota.get_default_limit_bytes() == 1024


@pytest.mark.asyncio
async def test_salted_keys_round_trip(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "metadata.json")
    quota, eviction = _managers()
    salted = _key(1, cache_salt="tenant-a")
    eviction.pin([salted])

    await _persister(store, quota, eviction).save()
    _, restored_eviction = _managers()
    _persister(store, QuotaManager(), restored_eviction).load()

    assert restored_eviction.capture() == _captured_pins((salted, 1))


@pytest.mark.asyncio
async def test_an_empty_state_round_trips(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "metadata.json")
    quota, eviction = _managers()

    await _persister(store, quota, eviction).save()
    restored_quota, restored_eviction = _managers()
    _persister(store, restored_quota, restored_eviction).load()

    assert restored_eviction.capture() == {"entries": []}
    assert restored_quota.list_quotas() == []
    assert restored_quota.get_default_limit_bytes() is None


@pytest.mark.asyncio
async def test_the_stored_document_is_readable(tmp_path: Path):
    """JSON is a deliberate choice here — an operator should be able to
    read why a tenant's cache was evicted."""
    path = tmp_path / "metadata.json"
    quota, eviction = _managers()
    quota.set_quota("tenant-a", 4096)
    eviction.pin([_key(1)])

    await _persister(LocalArtifactStore(path), quota, eviction).save()

    body = json.loads(path.read_text())
    assert body["version"] == 1
    assert body["components"]["quotas"]["limits"] == {"tenant-a": 4096}
    assert body["components"]["pins"]["entries"][0]["count"] == 1
    assert (
        body["components"]["pins"]["entries"][0]["key"]["chunk_hash_hex"]
        == _key(1).chunk_hash.hex()
    )


# -- What the state actually buys --------------------------------------------


@pytest.mark.asyncio
async def test_restored_quotas_arm_eviction(tmp_path: Path):
    """Without restored quotas the coordinator comes back unable to
    enforce anything until the controller re-syncs."""
    store = LocalArtifactStore(tmp_path / "metadata.json")
    quota, eviction = _managers()
    quota.set_quota("", 0)  # A zero quota evicts the whole salt.
    await _persister(store, quota, eviction).save()

    restored_quota, restored_eviction = _managers()
    _persister(store, restored_quota, restored_eviction).load()
    _l2_store(restored_eviction, _key(1))

    assert restored_eviction.compute_eviction_plan() == {"": [_key(1)]}


@pytest.mark.asyncio
async def test_a_restored_pin_survives_restored_quotas(tmp_path: Path):
    """The pair has to be restored together: quotas arm eviction, pins
    are the only thing holding it off."""
    store = LocalArtifactStore(tmp_path / "metadata.json")
    quota, eviction = _managers()
    quota.set_quota("", 0)
    eviction.pin([_key(1)])
    await _persister(store, quota, eviction).save()

    restored_quota, restored_eviction = _managers()
    _persister(store, restored_quota, restored_eviction).load()
    _l2_store(restored_eviction, _key(1))
    _l2_store(restored_eviction, _key(2))

    assert restored_eviction.compute_eviction_plan() == {"": [_key(2)]}


# -- Failure paths -----------------------------------------------------------


def test_a_missing_file_starts_with_no_state(tmp_path: Path):
    quota, eviction = _managers()

    _persister(LocalArtifactStore(tmp_path / "absent.json"), quota, eviction).load()

    assert eviction.capture() == {"entries": []}
    assert quota.get_default_limit_bytes() is None


def test_a_corrupt_file_starts_with_no_state(tmp_path: Path, logged):
    path = tmp_path / "metadata.json"
    path.write_text("{not json at all")
    quota, eviction = _managers()

    messages = logged(metadata_persister.logger)
    _persister(LocalArtifactStore(path), quota, eviction).load()

    assert eviction.capture() == {"entries": []}
    assert any("Ignoring metadata" in m for m in messages)


def test_an_unsupported_version_starts_with_no_state(tmp_path: Path, logged):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"version": 99, "saved_at": 0.0, "components": {}}))
    quota, eviction = _managers()

    messages = logged(metadata_persister.logger)
    _persister(LocalArtifactStore(path), quota, eviction).load()

    assert any("unsupported metadata version 99" in m for m in messages)


def test_a_pin_with_an_invalid_key_starts_with_no_state(tmp_path: Path, logged):
    """A key whose stored fields violate an ObjectKey invariant is a
    corrupt document, not a startup failure."""
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "saved_at": 0.0,
                "components": {
                    "pins": {
                        "entries": [
                            {
                                "key": {
                                    "chunk_hash_hex": "aa",
                                    "model_name": "m@x",
                                    "kv_rank": 0,
                                },
                                "count": 1,
                            }
                        ]
                    }
                },
            }
        )
    )
    quota, eviction = _managers()

    messages = logged(metadata_persister.logger)
    _persister(LocalArtifactStore(path), quota, eviction).load()

    assert eviction.capture() == {"entries": []}
    assert any("Ignoring metadata" in m for m in messages)


@pytest.mark.asyncio
async def test_an_unwritable_path_is_logged_not_raised(tmp_path: Path, logged):
    blocker = tmp_path / "blocked"
    blocker.write_bytes(b"")
    quota, eviction = _managers()

    messages = logged(metadata_persister.logger)
    await _persister(
        LocalArtifactStore(blocker / "metadata.json"), quota, eviction
    ).save()

    assert any("Failed to write metadata" in m for m in messages)


@pytest.mark.asyncio
async def test_a_failed_write_leaves_no_temporary_file(tmp_path: Path):
    path = tmp_path / "metadata.json"
    quota, eviction = _managers()
    await _persister(LocalArtifactStore(path), quota, eviction).save()

    assert [p.name for p in tmp_path.iterdir()] == ["metadata.json"]


# -- Change-driven persistence ------------------------------------------------


@pytest.mark.asyncio
async def test_persist_writes_the_current_state(tmp_path: Path):
    path = tmp_path / "metadata.json"
    quota, eviction = _managers()
    persister = _persister(LocalArtifactStore(path), quota, eviction)
    eviction.pin([_key(1)])

    await persister.save()

    assert (
        json.loads(path.read_text())["components"]["pins"]["entries"][0]["count"] == 1
    )


@pytest.mark.asyncio
async def test_an_unconfigured_store_discards_writes(tmp_path: Path):
    quota, eviction = _managers()
    eviction.pin([_key(1)])

    await _persister(NullArtifactStore(), quota, eviction).save()

    assert list(tmp_path.iterdir()) == []


def test_an_unconfigured_store_loads_nothing():
    quota, eviction = _managers()

    _persister(NullArtifactStore(), quota, eviction).load()

    assert eviction.capture() == {"entries": []}


# -- Coordinator wiring -------------------------------------------------------


def _coordinator(metadata_path: str) -> TestClient:
    """A coordinator with no timers at all, so only change-driven writes fire."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        snapshot_interval=0.0,
        metadata_path=metadata_path,
    )
    return TestClient(create_app(config))


def _pin(client: TestClient, token_ids: list[int]) -> None:
    resp = client.post(
        "/cache/pins",
        json={
            "model_name": "m",
            "world_size": 1,
            "token_ids": token_ids,
            "cache_salt": "",
        },
    )
    assert resp.status_code == 200


def test_a_pin_is_durable_when_its_request_returns(tmp_path: Path):
    """No timer runs here, and the file is read straight after the
    response: the write has to be synchronous with the request."""
    path = tmp_path / "metadata.json"

    with _coordinator(str(path)) as client:
        _pin(client, list(range(256)))
        assert json.loads(path.read_text())["components"]["pins"]["entries"]


def test_a_quota_survives_a_restart(tmp_path: Path):
    path = tmp_path / "metadata.json"

    with _coordinator(str(path)) as client:
        assert (
            client.put("/quota/config", json={"default_limit_gb": 2}).status_code == 200
        )
        assert path.is_file()

    with _coordinator(str(path)) as restarted:
        assert restarted.get("/quota/config").json()["default_limit_gb"] == 2


class TestPersistenceTypeDispatch:
    """The persistence_type flag is what routes a component to its artifact."""

    def test_a_controller_without_durable_state_is_skipped(self):
        """Startup passes every controller it built, so one that persists
        nothing must cost nothing rather than need excluding by hand."""
        collected = ControllerRegistry([PrefetchManager()]).durable_components()

        assert collected == {
            PersistenceType.CHECKPOINT: [],
            PersistenceType.METADATA: [],
        }

    def test_components_are_collected_into_the_artifact_they_name(self):
        """The whole point of the flag: a caller routes state without
        knowing which controller produced it or what it holds."""
        collected = ControllerRegistry(
            [FleetEvictionController(), PrefetchManager()]
        ).durable_components()

        assert {
            persistence_type: [c.name for c in components]
            for persistence_type, components in collected.items()
        } == {
            PersistenceType.CHECKPOINT: ["lru_order"],
            PersistenceType.METADATA: ["pins", "quotas"],
        }

    def test_the_controller_advertises_every_component_it_owns(self):
        """Callers wire durability through this one call, so a component
        added inside the controller needs no change outside it."""
        controller = FleetEvictionController()

        components = controller.get_durable_components()

        assert {c.name for c in components} == {"pins", "quotas", "lru_order"}

    def test_operator_intent_and_derived_state_are_told_apart(self):
        """Pins and quotas are authored and irreplaceable; the policy's
        order is derived from the stream and rides with the checkpoint."""
        controller = FleetEvictionController()

        by_type = {
            c.name: c.persistence_type for c in controller.get_durable_components()
        }

        assert by_type == {
            "pins": PersistenceType.METADATA,
            "quotas": PersistenceType.METADATA,
            "lru_order": PersistenceType.CHECKPOINT,
        }

    def test_registering_checkpoint_state_as_metadata_is_rejected(self, tmp_path: Path):
        """A mis-declared component would be rewritten on every operator
        change and reloaded ahead of the replay that owns it, so this fails
        at startup rather than producing a quietly wrong document."""
        persister = MetadataPersister(LocalArtifactStore(tmp_path / "metadata.json"))
        policy = FleetEvictionController().policy

        with pytest.raises(ValueError, match="checkpoint state"):
            persister.register(policy)
