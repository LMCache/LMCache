# SPDX-License-Identifier: Apache-2.0
"""Persistence as the coordinator actually wires it: through the app."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import json

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


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
