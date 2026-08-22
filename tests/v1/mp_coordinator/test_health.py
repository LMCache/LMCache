# SPDX-License-Identifier: Apache-2.0
"""Tests for the health-check eviction logic and what a reaped
instance takes down with it."""

# Standard
import time

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app, evict_stale
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance


def _instance(instance_id: str, heartbeat: float) -> MPInstance:
    return MPInstance(
        instance_id=instance_id,
        ip="127.0.0.1",
        http_port=8080,
        registration_time=heartbeat,
        last_heartbeat_time=heartbeat,
    )


def test_evict_stale_removes_only_expired():
    registry = InstanceRegistry()
    now = time.monotonic()
    registry.register(_instance("fresh", now))
    registry.register(_instance("old", now - 100.0))

    evicted = evict_stale(registry, instance_timeout=30.0)

    assert evicted == ["old"]
    assert registry.contains("fresh")
    assert not registry.contains("old")


def test_evict_stale_noop_when_all_fresh():
    registry = InstanceRegistry()
    registry.register(_instance("a", time.monotonic()))
    assert evict_stale(registry, instance_timeout=30.0) == []
    assert registry.contains("a")


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
        client.post(
            "/instances",
            json={"instance_id": "node-a", "ip": "127.0.0.1", "http_port": 8080},
        )
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
