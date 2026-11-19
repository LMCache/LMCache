# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's ``GET /status`` snapshot."""

# Standard
import asyncio

# Third Party
from fastapi.testclient import TestClient
import httpx

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _config() -> MPCoordinatorConfig:
    return MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)


def _store_batch(instance_id: str, seq: int) -> dict:
    return {
        "instance_id": instance_id,
        "incarnation": 1,
        "seq": seq,
        "event_type": "store",
        "tier": "l1",
        "backend": "dram",
        "entries": [
            {
                "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0},
                "size_bytes": 100,
            }
        ],
    }


def test_status_reports_every_section_of_a_fresh_coordinator():
    with TestClient(create_app(_config())) as client:
        response = client.get("/status")

    assert response.status_code == 200
    status = response.json()
    assert set(status) == {
        "event_source",
        "ingest",
        "consumers",
        "registry",
        "controllers",
        "background_loops",
        "event_loop",
    }
    assert status["event_source"] == {
        "source_name": "http",
        "replay_capability": "none",
    }
    assert status["controllers"] == {
        "FleetEvictionController": True,
        "PrefetchManager": True,
    }
    assert "KeyDirectory" in status["consumers"]


def test_status_reports_each_server_s_stream_and_heartbeat():
    with TestClient(create_app(_config())) as client:
        client.post(
            "/instances",
            json={"instance_id": "node-a", "ip": "127.0.0.1", "http_port": 8080},
        )
        client.post("/events", json={"batches": [_store_batch("node-a", 1)]})
        client.post("/events", json={"batches": [_store_batch("node-a", 3)]})

        status = client.get("/status").json()

    stream = status["ingest"]["streams"]["node-a"]
    assert (stream["last_seq"], stream["missing_batches"]) == (3, 1)
    assert stream["gap_detected"] is True
    assert status["ingest"]["batches_applied"] == 2
    assert set(status["registry"]["seconds_since_heartbeat"]) == {"node-a"}


def test_status_reports_a_failing_loop_s_last_error():
    app = create_app(_config())
    loops = app.state.ctx.background_loops

    async def failing_pass() -> None:
        raise RuntimeError("disk full")

    async def scenario() -> httpx.Response:
        task = loops.start("boom", 0.001, failing_pass)
        while loops.stats()["boom"].passes_failed == 0:
            await asyncio.sleep(0.005)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://coordinator"
        ) as client:
            response = await client.get("/status")
        task.cancel()
        return response

    response = asyncio.run(scenario())

    assert response.status_code == 200
    loop = response.json()["background_loops"]["boom"]
    assert loop["last_error"] == "RuntimeError: disk full"
    assert loop["last_success_timestamp"] is None
    assert loop["seconds_since_last_success"] is None
