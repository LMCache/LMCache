# SPDX-License-Identifier: Apache-2.0
"""Persistence as the coordinator actually wires it: through the app."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
import asyncio
import json
import threading

# Third Party
from fastapi.testclient import TestClient
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
from lmcache.v1.mp_coordinator.persistence.store import LocalArtifactStore
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory


@contextmanager
def _coordinator(checkpoint: Path, metadata: Path) -> Iterator[TestClient]:
    """Run a coordinator over the given artifacts, stopping it cleanly."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        # Writes happen on the clean stop, so the tests need no timer.
        checkpoint_interval=0.0,
        checkpoint_path=str(checkpoint),
        metadata_path=str(metadata),
    )
    with TestClient(create_app(config)) as client:
        yield client


def _store_one_key(client: TestClient) -> None:
    """Report one L2 store, as an MP server would."""
    response = client.post(
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
    assert response.status_code == 200


class TestBothArtifacts:
    def test_a_restart_resumes_the_directory_and_the_operator_intent(
        self, tmp_path: Path
    ):
        """The two halves are stored apart but must come back together:
        the directory says what is cached, the metadata what may not be
        evicted."""
        checkpoint, metadata = tmp_path / "checkpoint", tmp_path / "metadata.json"

        with _coordinator(checkpoint, metadata) as client:
            _store_one_key(client)
            assert (
                client.put("/quota/config", json={"default_limit_gb": 2}).status_code
                == 200
            )

        assert checkpoint.is_file() and metadata.is_file()
        with _coordinator(checkpoint, metadata) as restarted:
            assert restarted.get("/directory/stats").json()["num_keys"] == 1
            assert restarted.get("/quota/config").json()["default_limit_gb"] == 2

    def test_operator_intent_is_durable_before_the_response(self, tmp_path: Path):
        """A 200 from a quota call has to mean the change survives, not
        that it will at the next checkpoint tick."""
        checkpoint, metadata = tmp_path / "checkpoint", tmp_path / "metadata.json"

        with _coordinator(checkpoint, metadata) as client:
            client.put("/quota/tenant-a", json={"limit_gb": 1})

            document = json.loads(metadata.read_text())
            assert document["components"]["quotas"]["limits"] == {
                "tenant-a": 1073741824
            }

    def test_persistence_off_by_default(self, tmp_path: Path):
        """Unconfigured paths must not create files or break the app."""
        config = MPCoordinatorConfig(
            health_check_interval=0.0, eviction_check_interval=0.0
        )

        with TestClient(create_app(config)) as client:
            _store_one_key(client)
            assert client.get("/directory/stats").json()["num_keys"] == 1

        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_periodic_write", [False, True])
async def test_shutdown_drains_periodic_checkpoint_before_final_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_periodic_write: bool
) -> None:
    """A slow periodic write must finish before shutdown replaces its file."""
    checkpoint = tmp_path / "checkpoint"
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        checkpoint_interval=0.001,
        checkpoint_path=str(checkpoint),
    )
    write_started = threading.Event()
    release_write = threading.Event()
    write_finished = threading.Event()
    writes: list[str] = []
    original_open_write = LocalArtifactStore.open_write

    @contextmanager
    def blocked_open_write(store: LocalArtifactStore) -> Iterator[BinaryIO]:
        periodic = not writes
        label = "periodic" if periodic else "final"
        writes.append(f"{label} started")
        try:
            with original_open_write(store) as stream:
                if periodic:
                    # Pause after the real temporary file is opened, with
                    # the old snapshot captured but not yet written.
                    write_started.set()
                    assert release_write.wait(timeout=10.0)
                    if fail_periodic_write:
                        raise OSError("injected checkpoint write failure")
                yield stream
        finally:
            writes.append(f"{label} finished")
            if periodic:
                write_finished.set()

    monkeypatch.setattr(LocalArtifactStore, "open_write", blocked_open_write)
    app = create_app(config)
    running = asyncio.Event()
    stop = asyncio.Event()

    async def run_coordinator() -> None:
        async with app.router.lifespan_context(app):
            running.set()
            await stop.wait()

    task = asyncio.create_task(run_coordinator())
    try:
        await asyncio.wait_for(running.wait(), timeout=5.0)
        assert await asyncio.to_thread(write_started.wait, 5.0)
        # The final checkpoint must include this event, which arrived
        # after the periodic writer captured an empty directory.
        app.state.ctx.event_gate.ingest(
            CacheEventBatch(
                instance_id="node-a",
                incarnation=1,
                seq=1,
                event_type=CacheEventType.STORE,
                tier=Tier.L2,
                backend="fs",
                entries=[
                    CacheEventEntry(
                        key=ObjectKey(
                            chunk_hash=b"a", model_name="m", kv_rank=0
                        ).to_encoded_object_key(),
                        size_bytes=1024,
                    )
                ],
            )
        )
        stop.set()
        # Give shutdown a bounded chance to run while the I/O is held.
        # wait() leaves the lifespan task running when the bound expires.
        await asyncio.wait({task}, timeout=0.2)
    finally:
        stop.set()
        release_write.set()
        await asyncio.wait_for(task, timeout=5.0)
        assert await asyncio.to_thread(write_finished.wait, 5.0)

    restarted = create_app(config)
    restored_keys = restarted.state.ctx.views.get(KeyDirectory).stats().num_keys
    assert (writes, restored_keys) == (
        ["periodic started", "periodic finished", "final started", "final finished"],
        1,
    )


@pytest.mark.asyncio
async def test_shutdown_interrupts_checkpoint_interval(tmp_path: Path) -> None:
    """An idle timer must not delay the final checkpoint until its next tick."""
    checkpoint = tmp_path / "checkpoint"
    app = create_app(
        MPCoordinatorConfig(
            health_check_interval=0.0,
            eviction_check_interval=0.0,
            checkpoint_interval=3600.0,
            checkpoint_path=str(checkpoint),
        )
    )

    async def run_coordinator() -> None:
        async with app.router.lifespan_context(app):
            # Let the timer start waiting for its next tick.
            await asyncio.sleep(0)

    await asyncio.wait_for(run_coordinator(), timeout=5.0)
    assert checkpoint.is_file()
