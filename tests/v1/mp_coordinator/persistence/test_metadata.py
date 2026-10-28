# SPDX-License-Identifier: Apache-2.0
"""Tests for the metadata document: operator intent, written on change."""

# Standard
from collections.abc import Mapping
from pathlib import Path
import json
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType
from lmcache.v1.mp_coordinator.persistence.metadata import (
    MetadataPersister,
    MetadataWriteError,
)
from lmcache.v1.mp_coordinator.persistence.store import LocalArtifactStore
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_id.to_bytes(4, "big"), model_name="m", kv_rank=0)


def _controller() -> FleetEvictionController:
    return FleetEvictionController(
        usage_manager=CacheUsageManager(), registry=InstanceRegistry()
    )


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


class _Unserializable:
    """A metadata component whose capture JSON cannot encode: a bug."""

    persistence_type = PersistenceType.METADATA
    name = "broken"

    def capture(self) -> Mapping[str, object]:
        return {"value": object()}

    def restore(self, state: Mapping[str, object]) -> None:
        pass


class TestWriteFailures:
    def test_a_failed_write_raises_instead_of_logging(self, tmp_path: Path):
        """A pin or quota a restart would lose must not be reported as
        saved, so the caller has to hear about it."""
        controller = _controller()
        controller.pin([_key(1)])
        # A file where the directory should be: the write cannot even start.
        (tmp_path / "blocker").write_text("")
        persister = MetadataPersister(
            LocalArtifactStore(tmp_path / "blocker" / "metadata.json")
        )
        persister.register(controller)

        with pytest.raises(MetadataWriteError, match="blocker"):
            persister.save()

    def test_a_failed_write_leaves_the_previous_document(self, tmp_path: Path):
        """The write is atomic: a reader still finds the last good copy."""
        if os.geteuid() == 0:
            pytest.skip("root can write to a read-only directory")
        path = tmp_path / "metadata.json"
        quota = QuotaManager()
        quota.set_quota("t", 1024)
        persister = MetadataPersister(LocalArtifactStore(path))
        persister.register(quota)
        persister.save()
        tmp_path.chmod(0o500)
        try:
            quota.set_quota("t", 2048)
            with pytest.raises(MetadataWriteError):
                persister.save()
        finally:
            tmp_path.chmod(0o700)

        restored = QuotaManager()
        reader = MetadataPersister(LocalArtifactStore(path))
        reader.register(restored)
        reader.load()
        assert restored.get_limit_bytes("t") == 1024

    def test_a_capture_json_cannot_encode_is_a_bug_not_a_write_error(
        self, tmp_path: Path
    ):
        """Retrying cannot fix a bad capture, so it must not read as a
        storage problem the caller can wait out."""
        persister = MetadataPersister(LocalArtifactStore(tmp_path / "metadata.json"))
        persister.register(_Unserializable())

        with pytest.raises(TypeError):
            persister.save()


class TestStats:
    def test_saves_are_counted_by_result(self, tmp_path: Path):
        unwritable = tmp_path / "dir-in-the-way"
        unwritable.mkdir()
        good = MetadataPersister(LocalArtifactStore(tmp_path / "metadata.json"))
        bad = MetadataPersister(LocalArtifactStore(unwritable))

        good.save()
        with pytest.raises(MetadataWriteError):
            bad.save()

        assert good.stats().writes == {"ok": 1, "error": 0}
        assert bad.stats().writes == {"ok": 0, "error": 1}


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
