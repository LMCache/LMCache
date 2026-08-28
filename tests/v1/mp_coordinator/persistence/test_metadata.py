# SPDX-License-Identifier: Apache-2.0
"""Tests for the metadata document: operator intent, written on change."""

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.persistence.store import LocalArtifactStore
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_id.to_bytes(4, "big"), model_name="m", kv_rank=0)


def _controller() -> FleetEvictionController:
    return FleetEvictionController(usage_manager=CacheUsageManager())


class TestRoundTrip:
    def test_pins_and_quotas_survive_a_restart(self, tmp_path: Path):
        """Nothing can reconstruct operator intent, so losing it means an
        operator has to notice and re-apply it."""
        store = LocalArtifactStore(tmp_path / "metadata.json")
        live = _controller()
        live.pin([_key(1)])
        live.quota.set_quota("tenant-a", 4096)
        live.quota.set_default_limit_bytes(8192)
        persister = MetadataPersister(store)
        persister.register(live)
        persister.register(live.quota)

        persister.save()

        restored = _controller()
        restarted = MetadataPersister(store)
        restarted.register(restored)
        restarted.register(restored.quota)
        restarted.load()

        assert restored.filter_unpinned([_key(1)]) == []
        assert restored.quota.get_limit_bytes("tenant-a") == 4096
        assert restored.quota.get_default_limit_bytes() == 8192

    def test_the_document_is_readable_json(self, tmp_path: Path):
        """The one person likely to open this file is the operator whose
        intent it holds."""
        store = LocalArtifactStore(tmp_path / "metadata.json")
        controller = _controller()
        controller.quota.set_quota("tenant-a", 4096)
        persister = MetadataPersister(store)
        persister.register(controller.quota)

        persister.save()

        document = json.loads((tmp_path / "metadata.json").read_text())
        assert document["version"] == 1
        assert document["components"]["quotas"]["limits"] == {"tenant-a": 4096}


class TestSurvivableFailures:
    def test_a_missing_document_leaves_the_components_alone(self, tmp_path: Path):
        controller = _controller()
        persister = MetadataPersister(LocalArtifactStore(tmp_path / "absent"))
        persister.register(controller)

        persister.load()

        assert controller.filter_unpinned([_key(1)]) == [_key(1)]

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ("{not json", "unparsable"),
            ("[]", "not an object"),
            ('{"version": 99, "components": {}, "saved_at": 0}', "wrong version"),
            ('{"version": 1, "saved_at": 0}', "no components"),
            ('{"version": 1, "components": {}}', "no timestamp"),
        ],
    )
    def test_a_bad_document_is_ignored_not_fatal(
        self, tmp_path: Path, payload: str, reason: str
    ):
        """An operator can re-apply intent; a coordinator that will not
        boot needs someone paged."""
        path = tmp_path / "metadata.json"
        path.write_text(payload)
        controller = _controller()
        persister = MetadataPersister(LocalArtifactStore(path))
        persister.register(controller)

        persister.load()

        assert controller.filter_unpinned([_key(1)]) == [_key(1)], reason


class TestRegistration:
    def test_checkpoint_state_is_refused(self, tmp_path: Path):
        """Derived state here would be rewritten on every operator change
        and reloaded ahead of the checkpoint that owns it."""
        persister = MetadataPersister(LocalArtifactStore(tmp_path / "metadata.json"))

        with pytest.raises(ValueError, match="checkpoint state"):
            persister.register(_controller().policy)

    def test_a_quota_manager_alone_is_enough(self, tmp_path: Path):
        """Components are independent: registering one does not require
        the others."""
        store = LocalArtifactStore(tmp_path / "metadata.json")
        quota = QuotaManager()
        quota.set_quota("t", 1024)
        persister = MetadataPersister(store)
        persister.register(quota)
        persister.save()

        restored = QuotaManager()
        reader = MetadataPersister(store)
        reader.register(restored)
        reader.load()

        assert restored.get_limit_bytes("t") == 1024
