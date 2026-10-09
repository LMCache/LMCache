# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's ``/instances/usage`` endpoints."""

# Standard
from typing import cast

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ModuleMemoryCapacity, ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType
from lmcache.v1.mp_coordinator.views.server_config import ServerConfigRegistry

GIB = 1 << 30


@pytest.fixture
def client() -> TestClient:
    """A coordinator with both background loops disabled.

    Carries a per-instance seq allocator: ``config`` batches share the seq
    space with placement batches, so reusing a number gets the second batch
    dropped as a duplicate.
    """
    app = create_app(
        MPCoordinatorConfig(health_check_interval=0, eviction_check_interval=0)
    )
    test_client = TestClient(app)
    test_client.seqs = {}  # type: ignore[attr-defined]
    return test_client


def _take_seqs(client: TestClient, instance_id: str, count: int) -> int:
    """Reserve ``count`` consecutive seq numbers and return the first."""
    seqs: dict[str, int] = client.seqs  # type: ignore[attr-defined]
    first = seqs.get(instance_id, 0) + 1
    seqs[instance_id] = first + count - 1
    return first


def _ctx(client: TestClient) -> CoordinatorContext:
    """Return the coordinator context behind ``client``.

    ``TestClient.app`` is a bare ASGI callable, so the cast happens here once.

    Args:
        client: Test client wrapping a coordinator app.

    Returns:
        That app's :class:`CoordinatorContext`.
    """
    return cast("FastAPI", client.app).state.ctx


def _config_batches(
    instance_id: str,
    modules: list[dict[str, object]],
    incarnation: int,
    revision: int,
    first_seq: int,
) -> list[dict[str, object]]:
    """One ``config`` batch per compartment, all at the same revision."""
    return [
        {
            "instance_id": instance_id,
            "incarnation": incarnation,
            "seq": first_seq + offset,
            "event_type": "config",
            "tier": module["tier"],
            "backend": module["backend"],
            "shared": module.get("shared", False),
            "entries": [],
            "capacity_bytes": module.get("capacity_bytes", 0),
            "capacity_revision": revision,
        }
        for offset, module in enumerate(modules)
    ]


def _declare(
    client: TestClient,
    instance_id: str,
    modules: list[dict[str, object]],
    incarnation: int = 1,
    revision: int = 1,
) -> None:
    """Declare ``modules`` as ``config`` batches on the event stream."""
    first_seq = _take_seqs(client, instance_id, max(len(modules), 1))
    response = client.post(
        "/events",
        json={
            "batches": _config_batches(
                instance_id, modules, incarnation, revision, first_seq
            )
        },
    )
    assert response.status_code == 200, response.text


def _register(
    client: TestClient,
    instance_id: str,
    modules: list[dict[str, object]],
) -> None:
    """Register an instance, then declare ``modules`` as a report.

    Registration carries no capacity: it arrives on the event stream, so a
    declaring server takes both steps.
    """
    response = client.post(
        "/instances",
        json={
            "instance_id": instance_id,
            "ip": "10.0.0.1",
            "http_port": 8000,
        },
    )
    assert response.status_code == 200
    if modules:
        _declare(client, instance_id, modules)


def _ingest(
    client: TestClient,
    instance_id: str,
    tier: Tier,
    backend: str,
    size_bytes: int,
    index: int,
    shared: bool = False,
) -> None:
    """Push one STORE batch through the ingest gate."""
    seq = _take_seqs(client, instance_id, 1)
    key = ObjectKey(
        chunk_hash=bytes([index]) * 32,
        model_name="model",
        kv_rank=0,
        cache_salt="tenant",
    )
    _ctx(client).event_gate.ingest(
        CacheEventBatch(
            instance_id=instance_id,
            incarnation=1,
            seq=seq,
            event_type=CacheEventType.STORE,
            tier=tier,
            backend=backend,
            shared=shared,
            ts=1.0,
            entries=[
                CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size_bytes)
            ],
        )
    )


def _module(body: dict, tier: str, backend: str) -> dict:
    """Pick one module out of an instance status body."""
    found = [
        m for m in body["modules"] if m["tier"] == tier and m["backend"] == backend
    ]
    assert len(found) == 1, f"expected one {tier}/{backend}, got {found}"
    return found[0]


class TestInstanceMemory:
    def test_joins_usage_to_declared_capacity(self, client: TestClient) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)

        body = client.get("/instances/mp-1/usage").json()
        module = _module(body, "l1", "dram")
        assert module["used_bytes"] == 10 * GIB
        assert module["capacity_bytes"] == 40 * GIB
        assert module["usage_ratio"] == pytest.approx(0.25)
        assert body["registered"] is True
        assert body["declared_capacity"] is True

    def test_undeclared_capacity_reports_no_ratio(self, client: TestClient) -> None:
        # A null ratio means unknown, not empty.
        _register(client, "mp-1", [])
        _ingest(client, "mp-1", Tier.L2, "fs", 7 * GIB, index=1)

        body = client.get("/instances/mp-1/usage").json()
        module = _module(body, "l2", "fs")
        assert module["used_bytes"] == 7 * GIB
        assert module["capacity_bytes"] == 0
        assert module["usage_ratio"] is None
        assert body["declared_capacity"] is False

    def test_declared_but_unused_module_is_reported_empty(
        self, client: TestClient
    ) -> None:
        # Declared but idle reads as 0%, not as unknown.
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        module = _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")
        assert module["used_bytes"] == 0
        assert module["usage_ratio"] == pytest.approx(0.0)

    def test_ratio_above_one_is_not_clamped(self, client: TestClient) -> None:
        # Over-full signals a misconfigured cap; clamping would hide it.
        _register(
            client, "mp-1", [{"tier": "l1", "backend": "dram", "capacity_bytes": GIB}]
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 3 * GIB, index=1)
        assert _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")[
            "usage_ratio"
        ] == pytest.approx(3.0)

    def test_unknown_instance_is_404(self, client: TestClient) -> None:
        assert client.get("/instances/nobody/usage").status_code == 404

    def test_deregistered_instance_keeps_surviving_l2_bytes(
        self, client: TestClient
    ) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l2", "backend": "fs", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L2, "fs", 5 * GIB, index=1)
        assert client.delete("/instances/mp-1").status_code == 204

        body = client.get("/instances/mp-1/usage").json()
        assert body["registered"] is False
        # Capacity went with the departed process; the bytes did not.
        assert body["declared_capacity"] is False
        module = _module(body, "l2", "fs")
        assert module["used_bytes"] == 5 * GIB
        assert module["usage_ratio"] is None


class TestFleetMemory:
    def test_lists_every_instance(self, client: TestClient) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        _register(
            client,
            "mp-2",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 80 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)
        _ingest(client, "mp-2", Tier.L1, "dram", 60 * GIB, index=2)

        body = client.get("/instances/usage").json()
        ratios = {
            entry["instance_id"]: _module(entry, "l1", "dram")["usage_ratio"]
            for entry in body["instances"]
        }
        assert ratios == {"mp-1": pytest.approx(0.25), "mp-2": pytest.approx(0.75)}

    def test_shared_pool_is_reported_once_not_per_mount(
        self, client: TestClient
    ) -> None:
        shared = {
            "tier": "l2",
            "backend": "s3",
            "capacity_bytes": 100 * GIB,
            "shared": True,
        }
        _register(client, "mp-1", [shared])
        _register(client, "mp-2", [shared])
        _ingest(client, "mp-1", Tier.L2, "s3", 25 * GIB, index=1, shared=True)
        _ingest(client, "mp-2", Tier.L2, "s3", 25 * GIB, index=1, shared=True)

        body = client.get("/instances/usage").json()
        assert len(body["shared_modules"]) == 1
        pool = body["shared_modules"][0]
        assert pool["used_bytes"] == 25 * GIB
        assert pool["usage_ratio"] == pytest.approx(0.25)
        # Counted once, never onto the mounting instances.
        for entry in body["instances"]:
            assert entry["modules"] == []

    def test_disagreeing_shared_capacity_reads_as_undeclared(
        self, client: TestClient
    ) -> None:
        # Picking either would make the answer depend on registration order.
        _register(
            client,
            "mp-1",
            [
                {
                    "tier": "l2",
                    "backend": "s3",
                    "capacity_bytes": 100 * GIB,
                    "shared": True,
                }
            ],
        )
        _register(
            client,
            "mp-2",
            [
                {
                    "tier": "l2",
                    "backend": "s3",
                    "capacity_bytes": 999 * GIB,
                    "shared": True,
                }
            ],
        )
        _ingest(client, "mp-1", Tier.L2, "s3", 25 * GIB, index=1, shared=True)

        pool = client.get("/instances/usage").json()["shared_modules"][0]
        assert pool["capacity_bytes"] == 0
        assert pool["usage_ratio"] is None

    def test_empty_fleet(self, client: TestClient) -> None:
        assert client.get("/instances/usage").json() == {
            "instances": [],
            "shared_modules": [],
        }


class TestRegistration:
    """Registration establishes membership only, never capacity."""

    def test_registration_alone_declares_nothing(self, client: TestClient) -> None:
        response = client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "10.0.0.1", "http_port": 8000},
        )
        assert response.status_code == 200
        assert client.get("/instances/mp-1/usage").json()["declared_capacity"] is False

    def test_capacity_fields_are_not_accepted_at_registration(
        self, client: TestClient
    ) -> None:
        # Guards the one-path invariant: if these are ever reintroduced on
        # the register body, capacity has two sources again.
        response = client.post(
            "/instances",
            json={
                "instance_id": "mp-1",
                "ip": "10.0.0.1",
                "http_port": 8000,
                "memory_modules": [
                    {"tier": "l1", "backend": "dram", "capacity_bytes": GIB}
                ],
            },
        )
        assert response.status_code == 200
        assert client.get("/instances/mp-1/usage").json()["declared_capacity"] is False

    def test_tier_all_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/events",
            json={
                "batches": _config_batches(
                    "mp-1",
                    [{"tier": "all", "backend": "dram", "capacity_bytes": GIB}],
                    incarnation=1,
                    revision=1,
                    first_seq=1,
                )
            },
        )
        assert response.status_code == 422

    def test_repeated_compartment_in_one_declaration_takes_the_last(
        self, client: TestClient
    ) -> None:
        # Per-compartment batches means the registry never sees the whole
        # list, so it cannot reject a duplicate the way a single whole-list
        # report could. It upserts, and the last batch wins. The emitter
        # derives one entry per medium and adapter, so this is unreachable
        # from a correct producer.
        _register(client, "mp-1", [])
        _declare(
            client,
            "mp-1",
            [
                {"tier": "l1", "backend": "dram", "capacity_bytes": GIB},
                {"tier": "l1", "backend": "dram", "capacity_bytes": 2 * GIB},
            ],
            revision=2,
        )
        module = _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")
        assert module["capacity_bytes"] == 2 * GIB


class TestCapacityDeclarations:
    """``config`` batches accumulate into declarations, fenced by the gate."""

    def _capacity(self, client: TestClient, instance_id: str) -> int:
        """Read back one instance's declared L1 capacity."""
        body = client.get(f"/instances/{instance_id}/usage").json()
        return _module(body, "l1", "dram")["capacity_bytes"]

    def _l1(self, capacity_bytes: int) -> list[dict[str, object]]:
        """One L1/dram compartment at ``capacity_bytes``."""
        return [{"tier": "l1", "backend": "dram", "capacity_bytes": capacity_bytes}]

    def test_a_later_declaration_supersedes_the_earlier_one(
        self, client: TestClient
    ) -> None:
        _register(client, "mp-1", self._l1(40 * GIB))
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)
        assert _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")[
            "usage_ratio"
        ] == pytest.approx(0.25)

        # A device was added at runtime: same server, bigger pool.
        _declare(client, "mp-1", self._l1(80 * GIB), revision=2)
        assert _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")[
            "usage_ratio"
        ] == pytest.approx(0.125)

    def test_a_stale_revision_cannot_regress_the_topology(
        self, client: TestClient
    ) -> None:
        _register(client, "mp-1", [])
        _declare(client, "mp-1", self._l1(80 * GIB), revision=5)
        _declare(client, "mp-1", self._l1(10 * GIB), revision=3)
        assert self._capacity(client, "mp-1") == 80 * GIB

    def test_the_same_revision_extends_the_declaration(
        self, client: TestClient
    ) -> None:
        # One declaration is several batches sharing a revision, so an equal
        # revision adds a compartment rather than being ignored. This is how
        # a multi-compartment server declares at all.
        _register(client, "mp-1", [])
        _declare(client, "mp-1", self._l1(40 * GIB), revision=1)
        _declare(
            client,
            "mp-1",
            [{"tier": "l2", "backend": "fs", "capacity_bytes": 90 * GIB}],
            revision=1,
        )
        backends = {
            (m["tier"], m["backend"])
            for m in client.get("/instances/mp-1/usage").json()["modules"]
        }
        assert backends == {("l1", "dram"), ("l2", "fs")}

    def test_a_new_declaration_retires_compartments_it_omits(
        self, client: TestClient
    ) -> None:
        # The replacement for wholesale replacement: a dropped adapter is
        # simply absent from the next revision, so it stops being reported.
        _register(
            client,
            "mp-1",
            [
                {"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB},
                {"tier": "l2", "backend": "fs", "capacity_bytes": 90 * GIB},
            ],
        )
        _declare(client, "mp-1", self._l1(40 * GIB), revision=2)
        backends = {
            (m["tier"], m["backend"])
            for m in client.get("/instances/mp-1/usage").json()["modules"]
        }
        assert backends == {("l1", "dram")}

    def test_a_new_incarnation_supersedes_a_higher_revision(
        self, client: TestClient
    ) -> None:
        # A restarted server's revision counter goes back to 1. Without
        # incarnation in the order, every report it ever sends would look
        # stale and its capacity would never be learned again.
        _register(client, "mp-1", [])
        _declare(client, "mp-1", self._l1(80 * GIB), incarnation=1, revision=9)
        _declare(client, "mp-1", self._l1(16 * GIB), incarnation=2, revision=1)
        assert self._capacity(client, "mp-1") == 16 * GIB

    def test_an_older_incarnation_is_dropped_by_the_gate(
        self, client: TestClient
    ) -> None:
        # The reverse, and now the gate's job rather than the registry's: a
        # straggler from the previous process is fenced before it arrives.
        _register(client, "mp-1", [])
        _declare(client, "mp-1", self._l1(16 * GIB), incarnation=2, revision=1)
        _declare(client, "mp-1", self._l1(80 * GIB), incarnation=1, revision=99)
        assert self._capacity(client, "mp-1") == 16 * GIB

    def test_config_and_placement_batches_ride_the_same_request(
        self, client: TestClient
    ) -> None:
        _register(client, "mp-1", self._l1(40 * GIB))
        _ingest(client, "mp-1", Tier.L1, "dram", 20 * GIB, index=1)
        module = _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")
        assert module["used_bytes"] == 20 * GIB
        assert module["usage_ratio"] == pytest.approx(0.5)


class TestRouting:
    """``/instances/usage`` is a literal segment on a templated collection."""

    def test_usage_is_not_captured_as_an_instance_id(self, client: TestClient) -> None:
        # Routers are discovered alphabetically, so instances_api registers
        # first. If it ever declares GET /instances/{instance_id}, that route
        # would swallow "usage" and this fleet read would break.
        _register(client, "mp-1", [])
        body = client.get("/instances/usage").json()
        assert "instances" in body and "shared_modules" in body
        assert [e["instance_id"] for e in body["instances"]] == ["mp-1"]

    def test_an_instance_literally_named_usage_still_resolves(
        self, client: TestClient
    ) -> None:
        # The per-instance route carries a distinct /usage suffix, so the
        # collection route cannot shadow it even for this id.
        _register(client, "usage", [])
        assert client.get("/instances/usage/usage").json()["instance_id"] == "usage"


class TestDeparture:
    """A departing process takes its L1 pool with it."""

    def test_deregistration_fences_l1_bytes(self, client: TestClient) -> None:
        # Only the stale-eviction loop used to fence, and it never sees an
        # instance that left cleanly -- so its L1 bytes lingered for good.
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)
        assert (
            _module(client.get("/instances/mp-1/usage").json(), "l1", "dram")[
                "used_bytes"
            ]
            == 10 * GIB
        )

        assert client.delete("/instances/mp-1").status_code == 204
        assert client.get("/instances/mp-1/usage").status_code == 404

    def test_deregistration_keeps_l2_bytes(self, client: TestClient) -> None:
        # L2 outlives the reporter: it is storage the fleet shares, and
        # leaves only through a DELETE event.
        _register(
            client,
            "mp-1",
            [{"tier": "l2", "backend": "fs", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L2, "fs", 5 * GIB, index=1)
        assert client.delete("/instances/mp-1").status_code == 204

        body = client.get("/instances/mp-1/usage").json()
        assert body["registered"] is False
        assert body["declared_capacity"] is False
        module = _module(body, "l2", "fs")
        assert module["used_bytes"] == 5 * GIB
        assert module["usage_ratio"] is None


class TestDurableComponent:
    """The registry persists itself like every other event consumer."""

    def _registry(self) -> ServerConfigRegistry:
        """A registry holding one two-compartment declaration."""
        registry = ServerConfigRegistry()
        for offset, (tier, backend, capacity, shared) in enumerate(
            [
                (Tier.L1, "dram", 40 * GIB, False),
                (Tier.L2, "s3", 0, True),
            ]
        ):
            registry.consume(
                CacheEventBatch(
                    instance_id="mp-1",
                    incarnation=5,
                    seq=offset + 1,
                    event_type=CacheEventType.CONFIG,
                    tier=tier,
                    backend=backend,
                    shared=shared,
                    capacity_bytes=capacity,
                    capacity_revision=3,
                )
            )
        return registry

    def test_it_names_itself_and_where_it_belongs(self) -> None:
        registry = ServerConfigRegistry()
        assert registry.name == "server_config"
        assert registry.persistence_type == PersistenceType.CHECKPOINT

    def test_capture_is_plain_data(self) -> None:
        # Domain objects would make every artifact writer understand what a
        # section means.
        captured = self._registry().capture()
        assert captured == {
            "declarations": [
                (
                    "mp-1",
                    5,
                    3,
                    [("l1", "dram", 40 * GIB, False), ("l2", "s3", 0, True)],
                )
            ]
        }

    def test_restore_round_trips_the_declaration(self) -> None:
        restored = ServerConfigRegistry()
        restored.restore(self._registry().capture())
        assert restored.get("mp-1") == (
            ModuleMemoryCapacity(Tier.L1, "dram", 40 * GIB, False),
            ModuleMemoryCapacity(Tier.L2, "s3", 0, True),
        )

    def test_restore_keeps_the_stamp_so_a_straggler_cannot_win(self) -> None:
        # Without the stamp a restored registry starts from scratch and
        # accepts a report from before the capture, regressing the topology
        # it just loaded.
        restored = ServerConfigRegistry()
        restored.restore(self._registry().capture())
        restored.consume(
            CacheEventBatch(
                instance_id="mp-1",
                incarnation=5,
                seq=99,
                event_type=CacheEventType.CONFIG,
                tier=Tier.L1,
                backend="dram",
                capacity_bytes=1 * GIB,
                capacity_revision=2,
            )
        )
        assert restored.get("mp-1")[0].capacity_bytes == 40 * GIB

    def test_restore_refuses_a_non_empty_registry(self) -> None:
        registry = self._registry()
        with pytest.raises(ValueError, match="requires an empty registry"):
            registry.restore(registry.capture())


class TestCheckpointWiring:
    """The registry must be in the coordinator's checkpoint set.

    Declaring itself ``CHECKPOINT`` is not enough on its own -- it also
    has to be discovered, or it would advertise durability and never be
    captured.
    """

    def test_the_registry_is_a_checkpoint_component(self, client: TestClient) -> None:
        # First Party
        from lmcache.v1.mp_coordinator.persistence.durable_component import (
            PersistenceType,
        )

        durable = _ctx(client).views.durable_components()

        names = {c.name for c in durable[PersistenceType.CHECKPOINT]}
        assert "server_config" in names

    def test_a_declaration_survives_capture_and_restore(
        self, client: TestClient
    ) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 64 * GIB}],
        )
        registry = _ctx(client).views.get(ServerConfigRegistry)
        restored = ServerConfigRegistry()
        restored.restore(registry.capture())
        assert restored.get("mp-1") == registry.get("mp-1")
