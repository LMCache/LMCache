# SPDX-License-Identifier: Apache-2.0
"""Tests for the health-check eviction logic and what a reaped
instance takes down with it."""

# Standard
import time

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _register(client: TestClient, instance_id: str = "node-a") -> None:
    client.post(
        "/instances",
        json={"instance_id": instance_id, "ip": "127.0.0.1", "http_port": 8080},
    )


def _instance_ids(client: TestClient) -> set[str]:
    return {i["instance_id"] for i in client.get("/instances").json()["instances"]}


def test_health_loop_reaps_only_instances_whose_heartbeat_lapsed():
    config = MPCoordinatorConfig(
        health_check_interval=0.05,
        instance_timeout=0.3,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client, "fresh")
        _register(client, "old")

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and "old" in _instance_ids(client):
            client.put("/instances/fresh/heartbeat")
            time.sleep(0.05)

        assert _instance_ids(client) == {"fresh"}


def test_health_loop_keeps_instances_that_heartbeat():
    config = MPCoordinatorConfig(
        health_check_interval=0.05,
        instance_timeout=0.3,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client, "a")
        # Several health passes, each well inside the timeout.
        for _ in range(12):
            client.put("/instances/a/heartbeat")
            time.sleep(0.05)

        assert _instance_ids(client) == {"a"}


def _store_batch(instance_id: str, seq: int, tier: str, backend: str, h: str) -> dict:
    return {
        "instance_id": instance_id,
        "incarnation": 1,
        "seq": seq,
        "event_type": "store",
        "tier": tier,
        "backend": backend,
        "entries": [
            {
                "key": {
                    "chunk_hash_hex": h,
                    "model_name": "m",
                    "kv_rank": 0,
                    "cache_salt": "user-a",
                },
                "size_bytes": 1000,
            }
        ],
    }


def _total_gb(client: TestClient, tier: str) -> float:
    return client.get("/quota", params={"tier": tier}).json()["total_gb"]


def test_reaped_instance_loses_its_l1_usage_but_not_its_l2():
    """The health loop hands each reaped instance to the ingest gate, so
    every consumer fences it: its L1 was that process's memory, while its
    L2 sits on storage the fleet shares."""
    config = MPCoordinatorConfig(
        health_check_interval=0.05,
        instance_timeout=0.1,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client, "node-a")
        client.post(
            "/events",
            json={
                "batches": [
                    _store_batch("node-a", 1, "l1", "dram", "aa"),
                    _store_batch("node-a", 2, "l2", "fs", "bb"),
                ]
            },
        )
        assert _total_gb(client, "l1") > 0.0

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _total_gb(client, "l1") > 0.0:
            time.sleep(0.05)

        assert _total_gb(client, "l1") == 0.0
        assert not client.get("/instances").json()["instances"]
        # L2 bytes outlive the reporter and leave only via DELETE events.
        assert _total_gb(client, "l2") > 0.0


def test_reaped_instance_drops_its_capacity_declaration():
    """A heartbeat timeout forgets the instance's capacity declaration, as
    ``DELETE /instances`` does, so a crashed server leaves the fleet
    memory view."""
    config = MPCoordinatorConfig(
        health_check_interval=0.05,
        instance_timeout=0.5,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        client.post(
            "/instances",
            json={"instance_id": "node-a", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.post(
            "/events",
            json={
                "batches": [
                    {
                        "instance_id": "node-a",
                        "incarnation": 1,
                        "seq": 1,
                        "event_type": "config",
                        "tier": "l1",
                        "backend": "dram",
                        "entries": [],
                        "capacity_bytes": 1 << 30,
                        "capacity_revision": 1,
                    }
                ]
            },
        )
        assert client.get("/instances/node-a/usage").json()["declared_capacity"]

        deadline = time.monotonic() + 10.0
        while (
            time.monotonic() < deadline and client.get("/instances").json()["instances"]
        ):
            time.sleep(0.05)

        assert not client.get("/instances").json()["instances"]
        assert client.get("/instances/node-a/usage").status_code == 404


def test_late_l1_events_after_deregistration_are_dropped():
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    with TestClient(create_app(config)) as client:
        _register(client)
        client.post(
            "/events", json={"batches": [_store_batch("node-a", 1, "l1", "dram", "aa")]}
        )
        client.delete("/instances/node-a")

        late = _store_batch("node-a", 2, "l1", "dram", "bb")
        assert client.post("/events", json={"batches": [late]}).json()["stale"] == 1
        assert _total_gb(client, "l1") == 0.0

        _register(client)
        again = _store_batch("node-a", 3, "l1", "dram", "cc")
        assert client.post("/events", json={"batches": [again]}).json()["applied"] == 1


def test_restored_instance_that_never_returns_loses_its_l1(tmp_path):
    config = MPCoordinatorConfig(
        checkpoint_path=str(tmp_path / "checkpoint"),
        checkpoint_interval=0.0,
        health_check_interval=0.05,
        instance_timeout=0.5,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client)
        client.post(
            "/events",
            json={
                "batches": [
                    _store_batch("node-a", 1, "l1", "dram", "aa"),
                    _store_batch("node-a", 2, "l2", "fs", "bb"),
                ]
            },
        )

    with TestClient(create_app(config)) as client:
        assert _total_gb(client, "l1") > 0.0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _total_gb(client, "l1") > 0.0:
            time.sleep(0.05)

        assert _total_gb(client, "l1") == 0.0
        assert _total_gb(client, "l2") > 0.0


def test_restored_instance_that_re_registers_keeps_its_l1(tmp_path):
    config = MPCoordinatorConfig(
        checkpoint_path=str(tmp_path / "checkpoint"),
        checkpoint_interval=0.0,
        health_check_interval=0.05,
        instance_timeout=1.0,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client)
        client.post(
            "/events", json={"batches": [_store_batch("node-a", 1, "l1", "dram", "aa")]}
        )

    with TestClient(create_app(config)) as client:
        _register(client)
        time.sleep(0.6)
        client.put("/instances/node-a/heartbeat")
        time.sleep(0.7)

        assert _total_gb(client, "l1") > 0.0
