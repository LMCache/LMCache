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


def _freeze_coordinator(client: TestClient, seconds: float) -> None:
    """Block the coordinator's event loop, as a GC pause or SIGSTOP would:
    nothing it runs -- heartbeats, health passes -- gets a turn."""
    assert client.portal is not None
    client.portal.call(time.sleep, seconds)


def _heartbeat_for(client: TestClient, instance_id: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        client.put(f"/instances/{instance_id}/heartbeat")
        time.sleep(0.05)


def _heartbeat_until_gone(
    client: TestClient, alive: str, dead: str, timeout: float = 10.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and dead in _instance_ids(client):
        client.put(f"/instances/{alive}/heartbeat")
        time.sleep(0.05)


_STALL_CONFIG = MPCoordinatorConfig(
    health_check_interval=0.05,
    instance_timeout=0.5,
    eviction_check_interval=0.0,
)


def test_a_coordinator_freeze_is_not_held_against_instances():
    """Instances that had no chance to heartbeat while the coordinator was
    frozen are not evicted by the first health passes after it resumes;
    one that still does not heartbeat goes once the grace ends."""
    with TestClient(create_app(_STALL_CONFIG)) as client:
        _register(client, "alive")
        _register(client, "dead")

        _freeze_coordinator(client, 1.0)
        time.sleep(0.15)  # several health passes, well inside the grace

        assert _instance_ids(client) == {"alive", "dead"}

        # The grace is only a chance to heartbeat; it does not last.
        _heartbeat_until_gone(client, alive="alive", dead="dead")
        assert _instance_ids(client) == {"alive"}


def test_a_second_freeze_does_not_shield_an_instance_that_stopped():
    """A grace does not carry over: an instance that has not heartbeat
    since the first freeze is already out of time when the second one
    starts, so recurring freezes cannot keep a dead instance registered."""
    with TestClient(create_app(_STALL_CONFIG)) as client:
        _register(client, "alive")
        _register(client, "dead")
        _freeze_coordinator(client, 1.0)
        time.sleep(0.15)
        assert _instance_ids(client) == {"alive", "dead"}

        client.put("/instances/alive/heartbeat")
        _freeze_coordinator(client, 1.0)
        time.sleep(0.15)

        assert _instance_ids(client) == {"alive"}


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


def test_a_freeze_leaves_restored_instances_their_full_window(tmp_path):
    """A freeze right after a restore does not use up the restored
    instances' time to register again: they get the full timeout from when
    the coordinator runs again."""
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
            "/events", json={"batches": [_store_batch("node-a", 1, "l1", "dram", "aa")]}
        )

    with TestClient(create_app(config)) as client:
        _freeze_coordinator(client, 1.0)
        time.sleep(0.15)
        assert _total_gb(client, "l1") > 0.0

        _register(client)
        _heartbeat_for(client, "node-a", 0.7)  # past the grace
        assert _total_gb(client, "l1") > 0.0


def test_a_freeze_starting_inside_the_restore_window_extends_it(tmp_path):
    """A freeze that begins before the restored instances' window to
    register again ends still leaves them the timeout after it, even when
    the health pass it delays was due only after that window ended."""
    config = MPCoordinatorConfig(
        checkpoint_path=str(tmp_path / "checkpoint"),
        checkpoint_interval=0.0,
        health_check_interval=0.2,
        instance_timeout=0.5,
        eviction_check_interval=0.0,
    )
    with TestClient(create_app(config)) as client:
        _register(client)
        client.post(
            "/events", json={"batches": [_store_batch("node-a", 1, "l1", "dram", "aa")]}
        )

    started = time.monotonic()
    with TestClient(create_app(config)) as client:
        # Inside the 0.5 s window, but the pass after it is due beyond it.
        time.sleep(max(0.0, started + 0.4 - time.monotonic()))
        _freeze_coordinator(client, 1.0)
        time.sleep(0.05)
        assert _total_gb(client, "l1") > 0.0

        _register(client)
        _heartbeat_for(client, "node-a", 0.7)  # past the grace
        assert _total_gb(client, "l1") > 0.0
