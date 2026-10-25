# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the fleet memory-usage API.

Drives a uvicorn-served coordinator over real sockets: register, declare
capacities and publish usage to ``POST /events``, read
``GET /instances/usage``. Bodies use the same ``CacheEventsRequest`` dump
``HttpCacheEventSink`` sends, so the wire encoding under test is the real
one.
"""

# Standard
from typing import cast
import socket as _socket
import threading
import time

# Third Party
import pytest
import requests
import uvicorn

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest

GIB = 1 << 30


def _free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until_up(base_url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/healthz", timeout=0.5).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.05)
    raise RuntimeError("coordinator did not come up")


@pytest.fixture
def coordinator():
    """A live coordinator on a free port; yields its base URL."""
    port = _free_port()
    config = MPCoordinatorConfig(
        host="127.0.0.1",
        port=port,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config), host=config.host, port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_until_up(base)
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _declare(
    base: str,
    instance_id: str,
    modules: list[dict[str, object]],
    incarnation: int = 1,
    revision: int = 1,
    first_seq: int = 1,
) -> None:
    """Declare ``modules`` as ``config`` batches on the event stream."""
    batches = [
        CacheEventBatch(
            instance_id=instance_id,
            incarnation=incarnation,
            seq=first_seq + offset,
            event_type=CacheEventType.CONFIG,
            tier=Tier(module["tier"]),
            backend=str(module["backend"]),
            shared=bool(module.get("shared", False)),
            capacity_bytes=cast("int", module.get("capacity_bytes", 0)),
            capacity_revision=revision,
        )
        for offset, module in enumerate(modules)
    ]
    body = CacheEventsRequest(batches=batches)
    response = requests.post(
        f"{base}/events", json=body.model_dump(mode="json"), timeout=2
    )
    assert response.status_code == 200, response.text


def _register(base: str, instance_id: str, modules: list[dict[str, object]]) -> None:
    """Register an mp server, then declare ``modules`` over the event stream.

    The declaration consumes ``len(modules)`` seq numbers, so callers that
    also publish placements start those at ``len(modules) + 1``.
    """
    response = requests.post(
        f"{base}/instances",
        json={
            "instance_id": instance_id,
            "ip": "127.0.0.1",
            "http_port": 9999,
        },
        timeout=2,
    )
    assert response.status_code == 200, response.text
    if modules:
        _declare(base, instance_id, modules)


def _publish(
    base: str,
    instance_id: str,
    tier: Tier,
    backend: str,
    size_bytes: int,
    index: int,
    shared: bool = False,
    seq: int = 1,
    incarnation: int = 1,
    event_type: CacheEventType = CacheEventType.STORE,
) -> None:
    """Publish one cache-event batch to ``POST /events``."""
    key = ObjectKey(
        chunk_hash=bytes([index]) * 32,
        model_name="model",
        kv_rank=0,
        cache_salt="tenant",
    )
    entry = (
        CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size_bytes)
        if event_type is CacheEventType.STORE
        else CacheEventEntry(key=key.to_encoded_object_key())
    )
    body = CacheEventsRequest(
        batches=[
            CacheEventBatch(
                instance_id=instance_id,
                incarnation=incarnation,
                seq=seq,
                event_type=event_type,
                tier=tier,
                backend=backend,
                shared=shared,
                ts=time.time(),
                entries=[entry],
            )
        ]
    )
    response = requests.post(
        f"{base}/events", json=body.model_dump(mode="json"), timeout=2
    )
    assert response.status_code == 200, response.text


def _module(body: dict, tier: str, backend: str) -> dict:
    """Pick one module out of an instance-status body."""
    found = [
        m for m in body["modules"] if m["tier"] == tier and m["backend"] == backend
    ]
    assert len(found) == 1, f"expected one {tier}/{backend}, got {body['modules']}"
    return found[0]


def _instance(fleet: dict, instance_id: str) -> dict:
    """Pick one instance out of a fleet body."""
    found = [i for i in fleet["instances"] if i["instance_id"] == instance_id]
    assert len(found) == 1, f"expected one {instance_id}, got {fleet['instances']}"
    return found[0]


def test_usage_and_capacity_join_over_real_http(coordinator) -> None:
    """Both halves arrive on the event stream and are joined on read."""
    _register(
        coordinator,
        "mp-1",
        [
            {"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB},
            {"tier": "l2", "backend": "fs", "capacity_bytes": 200 * GIB},
        ],
    )
    _register(
        coordinator,
        "mp-2",
        [{"tier": "l1", "backend": "dram", "capacity_bytes": 80 * GIB}],
    )

    # Placement seqs continue past the config batches the declaration used.
    _publish(coordinator, "mp-1", Tier.L1, "dram", 10 * GIB, index=1, seq=3)
    _publish(coordinator, "mp-1", Tier.L2, "fs", 50 * GIB, index=2, seq=4)
    _publish(coordinator, "mp-2", Tier.L1, "dram", 60 * GIB, index=3, seq=2)

    fleet = requests.get(f"{coordinator}/instances/usage", timeout=2).json()
    assert [i["instance_id"] for i in fleet["instances"]] == ["mp-1", "mp-2"]

    mp1 = _instance(fleet, "mp-1")
    assert _module(mp1, "l1", "dram")["usage_ratio"] == pytest.approx(0.25)
    assert _module(mp1, "l2", "fs")["usage_ratio"] == pytest.approx(0.25)
    assert _module(mp2 := _instance(fleet, "mp-2"), "l1", "dram")[
        "usage_ratio"
    ] == pytest.approx(0.75)
    assert mp2["registered"] is True

    # The per-instance endpoint agrees with the fleet view.
    single = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    assert single == mp1


def test_undeclared_capacity_serializes_as_null_over_the_wire(coordinator) -> None:
    """Undeclared capacity (``capacity_bytes == 0``) serializes as ``null``.

    A numeric stand-in would read as real occupancy.
    """
    _register(coordinator, "mp-1", [])
    _publish(coordinator, "mp-1", Tier.L2, "fs", 7 * GIB, index=1)

    body = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    module = _module(body, "l2", "fs")
    assert module["used_bytes"] == 7 * GIB
    assert module["capacity_bytes"] == 0
    assert module["usage_ratio"] is None
    assert body["declared_capacity"] is False
    # Raw text, so a client-side default cannot mask a missing null.
    assert '"usage_ratio":null' in requests.get(
        f"{coordinator}/instances/mp-1/usage", timeout=2
    ).text.replace(" ", "")


def test_shared_pool_counted_once_across_mounts_over_real_http(coordinator) -> None:
    """One bucket mounted by two servers is one bucket, not two."""
    shared = {
        "tier": "l2",
        "backend": "s3",
        "capacity_bytes": 100 * GIB,
        "shared": True,
    }
    _register(coordinator, "mp-1", [shared])
    _register(coordinator, "mp-2", [shared])

    # Both servers report the same shared placement, at a seq past the
    # config batch each declaration consumed.
    _publish(coordinator, "mp-1", Tier.L2, "s3", 25 * GIB, index=1, shared=True, seq=2)
    _publish(coordinator, "mp-2", Tier.L2, "s3", 25 * GIB, index=1, shared=True, seq=2)

    fleet = requests.get(f"{coordinator}/instances/usage", timeout=2).json()
    assert len(fleet["shared_modules"]) == 1
    pool = fleet["shared_modules"][0]
    assert pool["used_bytes"] == 25 * GIB
    assert pool["usage_ratio"] == pytest.approx(0.25)
    # Attributed to neither mounting server.
    for entry in fleet["instances"]:
        assert entry["modules"] == []


def test_restart_fences_l1_but_keeps_l2_over_real_http(coordinator) -> None:
    """An incarnation bump voids the reporter's L1; its L2 bytes outlive it."""
    _register(
        coordinator,
        "mp-1",
        [
            {"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB},
            {"tier": "l2", "backend": "fs", "capacity_bytes": 200 * GIB},
        ],
    )
    _publish(
        coordinator, "mp-1", Tier.L1, "dram", 10 * GIB, index=1, seq=3, incarnation=1
    )
    _publish(
        coordinator, "mp-1", Tier.L2, "fs", 50 * GIB, index=2, seq=4, incarnation=1
    )

    before = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    assert _module(before, "l1", "dram")["used_bytes"] == 10 * GIB

    # Restart: same instance, higher incarnation, seq restarts at 1.
    _publish(
        coordinator, "mp-1", Tier.L1, "dram", 2 * GIB, index=5, seq=1, incarnation=2
    )

    after = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    assert _module(after, "l1", "dram")["used_bytes"] == 2 * GIB
    assert _module(after, "l2", "fs")["used_bytes"] == 50 * GIB


def test_delete_releases_bytes_over_real_http(coordinator) -> None:
    """A DELETE carries no size, so the coordinator must release what it admitted."""
    _register(
        coordinator,
        "mp-1",
        [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
    )
    _publish(coordinator, "mp-1", Tier.L1, "dram", 10 * GIB, index=1, seq=2)
    _publish(
        coordinator,
        "mp-1",
        Tier.L1,
        "dram",
        0,
        index=1,
        seq=3,
        event_type=CacheEventType.DELETE,
    )

    body = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    module = _module(body, "l1", "dram")
    assert module["used_bytes"] == 0
    assert module["usage_ratio"] == pytest.approx(0.0)


def test_unknown_instance_is_404_over_real_http(coordinator) -> None:
    assert (
        requests.get(f"{coordinator}/instances/nobody/usage", timeout=2).status_code
        == 404
    )


def test_producer_declaration_yields_real_ratios_over_real_http(coordinator) -> None:
    """What an MP server declares is what the coordinator can join against.

    Drives the real producer path: a ``StorageManager`` capacity snapshot
    through ``CacheEventSubscriber``, whose ``config`` batches are POSTed
    verbatim. Drift between the producer's compartment identity and the
    event stream's ``(tier, backend)`` surfaces here as a missing ratio.
    """
    # First Party
    from lmcache.v1.distributed.api import CapacitySnapshot, ModuleMemoryCapacity
    from lmcache.v1.mp_coordinator.cache_events import (
        CacheEventSink,
        CacheEventSubscriber,
    )
    from lmcache.v1.mp_observability.event import Event, EventType

    # Shape _build_capacities() returns for a hybrid Device-DAX server.
    produced = (
        ModuleMemoryCapacity(Tier.L1, "devdax", 100 * GIB, False),
        ModuleMemoryCapacity(Tier.L1, "dram", 10 * GIB, False),
        ModuleMemoryCapacity(Tier.L2, "fs", 200 * GIB, False),
        ModuleMemoryCapacity(Tier.L2, "s3", 0, True),
    )
    _register(coordinator, "mp-1", [])

    captured: list[CacheEventBatch] = []

    class _CapturingSink(CacheEventSink):
        def publish(self, batches: list[CacheEventBatch]) -> None:
            captured.extend(batches)

    subscriber = CacheEventSubscriber(
        sink=_CapturingSink(),
        instance_id="mp-1",
        incarnation=1,
        flush_interval=0.0,
    )
    subscriber.get_subscriptions()[EventType.SM_CAPACITY_CHANGED](
        Event(
            event_type=EventType.SM_CAPACITY_CHANGED,
            metadata={"snapshot": CapacitySnapshot(modules=produced)},
        )
    )
    subscriber.flush()
    assert len(captured) == len(produced), captured

    response = requests.post(
        f"{coordinator}/events",
        json=CacheEventsRequest(batches=captured).model_dump(mode="json"),
        timeout=2,
    )
    assert response.status_code == 200, response.text

    # Usage arrives on the event stream.
    _publish(coordinator, "mp-1", Tier.L1, "devdax", 25 * GIB, index=1, seq=5)
    _publish(coordinator, "mp-1", Tier.L1, "dram", 5 * GIB, index=2, seq=6)
    _publish(coordinator, "mp-1", Tier.L2, "fs", 50 * GIB, index=3, seq=7)
    _publish(coordinator, "mp-1", Tier.L2, "s3", 9 * GIB, index=4, seq=8, shared=True)

    status = requests.get(f"{coordinator}/instances/mp-1/usage", timeout=2).json()
    assert status["declared_capacity"] is True

    # Every declared compartment joins, including both hybrid L1 mediums.
    assert _module(status, "l1", "devdax")["usage_ratio"] == pytest.approx(0.25)
    assert _module(status, "l1", "dram")["usage_ratio"] == pytest.approx(0.5)
    assert _module(status, "l2", "fs")["usage_ratio"] == pytest.approx(0.25)

    # The uncapped shared bucket is fleet-scoped and has no denominator.
    pool = requests.get(f"{coordinator}/instances/usage", timeout=2).json()[
        "shared_modules"
    ]
    assert [(p["backend"], p["used_bytes"], p["usage_ratio"]) for p in pool] == [
        ("s3", 9 * GIB, None)
    ]
