# SPDX-License-Identifier: Apache-2.0
"""Tests for the checkpoint: what survives a restart, and what a bad
checkpoint costs."""

# Standard
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    DurableComponent,
    PersistenceType,
)
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.persistence.store import LocalArtifactStore
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager


class _Coordinator:
    """The consumers a coordinator wires up, plus its capture."""

    def __init__(self) -> None:
        self.directory = KeyDirectory()
        self.usage = CacheUsageManager()
        self.controller = FleetEvictionController(usage_manager=self.usage)
        broadcaster = CacheEventBroadcaster()
        broadcaster.register_consumer(self.directory)
        broadcaster.register_consumer(self.usage)
        broadcaster.register_consumer(self.controller)
        quiesce = QuiesceLock()
        self.gate = EventGate(broadcaster, quiesce)
        self.quiesce = quiesce
        durable: list[DurableComponent] = [
            self.directory,
            self.usage,
            self.gate,
            *self.controller.get_durable_components(),
        ]
        self.components = [
            c for c in durable if c.persistence_type is PersistenceType.CHECKPOINT
        ]


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_id.to_bytes(4, "big"), model_name="m", kv_rank=0)


def _store(
    seq: int,
    keys: list[ObjectKey],
    tier: Tier = Tier.L2,
    incarnation: int = 7,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=incarnation,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=tier,
        backend="fs" if tier is Tier.L2 else "dram",
        entries=[
            CacheEventEntry(
                key=k.to_encoded_object_key(),
                size_bytes=1024,
                token_ids=[i, i + 1],
                token_offset=i * 2,
            )
            for i, k in enumerate(keys, start=1)
        ],
    )


class TestRoundTrip:
    def test_a_restarted_coordinator_resumes(self, tmp_path: Path):
        """The whole point: what the fleet cached survives the process
        that learned it."""
        store = LocalArtifactStore(tmp_path / "checkpoint")
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1), _key(2)]))
        live.gate.ingest(_store(seq=2, keys=[_key(3)], tier=Tier.L1))

        save_checkpoint(store, live.quiesce, live.components)
        restarted = _Coordinator()
        load_checkpoint(store, restarted.components)

        assert restarted.directory.stats().num_keys == 3
        assert restarted.usage.get_salt_bytes(Tier.L2, "") == 2048
        assert _capture(restarted) == _capture(live)

    def test_restored_cursors_fence_a_restarted_emitter(self, tmp_path: Path):
        """Placements are only fenceable if the cursor dating them comes
        back with them."""
        store = LocalArtifactStore(tmp_path / "checkpoint")
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1)], tier=Tier.L1))
        save_checkpoint(store, live.quiesce, live.components)

        restarted = _Coordinator()
        load_checkpoint(store, restarted.components)
        assert restarted.directory.lookup([_key(1)])[0], "restored L1 placement"

        restarted.gate.ingest(
            _store(seq=1, keys=[_key(9)], tier=Tier.L1, incarnation=8)
        )

        assert restarted.directory.lookup([_key(1)]) == [[]]


class TestConsistency:
    def test_a_checkpoint_never_holds_a_half_applied_batch(self, tmp_path: Path):
        """A batch reaches the directory and the usage view in turn. A
        save that read them while one was in flight would write a state
        no moment matched -- a key the ledger has no bytes for -- and it
        would look perfectly plausible on restore.
        """
        store = LocalArtifactStore(tmp_path / "checkpoint")
        live = _Coordinator()

        # Widen the window between the two consumers; the real one is
        # microseconds, which a test would clear by luck.
        usage_consume = live.usage.consume

        def slow_usage_consume(batch: CacheEventBatch) -> None:
            time.sleep(0.3)
            usage_consume(batch)

        live.usage.consume = slow_usage_consume  # type: ignore[method-assign]

        with ThreadPoolExecutor(max_workers=2) as pool:
            ingesting = pool.submit(live.gate.ingest, _store(seq=1, keys=[_key(1)]))
            time.sleep(0.05)  # let the batch reach the slow consumer
            saving = pool.submit(save_checkpoint, store, live.quiesce, live.components)
            saving.result(timeout=5.0)
            assert ingesting.result(timeout=5.0)

        restarted = _Coordinator()
        load_checkpoint(store, restarted.components)

        assert restarted.directory.stats().num_keys == 1
        assert restarted.usage.get_salt_bytes(Tier.L2, "") == 1024, (
            "the checkpoint caught the directory without the bytes for it"
        )


class TestSurvivableFailures:
    def test_no_checkpoint_starts_cold(self, tmp_path: Path):
        """A first boot is not an error."""
        restarted = _Coordinator()

        load_checkpoint(LocalArtifactStore(tmp_path / "absent"), restarted.components)

        assert restarted.directory.stats().num_keys == 0

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            (b"not a checkpoint at all", "wrong magic"),
            (b"LMCKPT\0\0\x63\0\0\0rest", "unsupported version"),
            (b"LMCKPT\0\0", "no payload"),
            (b"LMC", "truncated header"),
        ],
    )
    def test_a_bad_checkpoint_is_ignored_not_fatal(
        self, tmp_path: Path, payload: bytes, reason: str
    ):
        """A coordinator that refuses to boot over a corrupt optimization
        is strictly worse than one that starts cold."""
        path = tmp_path / "checkpoint"
        path.write_bytes(payload)
        restarted = _Coordinator()

        load_checkpoint(LocalArtifactStore(path), restarted.components)

        assert restarted.directory.stats().num_keys == 0, reason

    def test_one_unreadable_section_does_not_cost_the_others(self, tmp_path: Path):
        """Sections are independent, so a component that cannot read its
        own must not take the rest of the restore with it."""
        store = LocalArtifactStore(tmp_path / "checkpoint")
        quiesce = QuiesceLock()
        save_checkpoint(store, quiesce, [_Section("first"), _Section("second")])

        readable, unreadable = _Section("first"), _Unreadable("second")
        load_checkpoint(store, [unreadable, readable])

        assert readable.restored == {"value": 1}, "a good section still loaded"

    def test_an_unwritable_checkpoint_does_not_raise(self, tmp_path: Path):
        """A checkpoint is an optimization; failing to write one must not
        take the coordinator down."""
        unwritable = tmp_path / "dir-in-the-way"
        unwritable.mkdir()
        live = _Coordinator()

        save_checkpoint(LocalArtifactStore(unwritable), live.quiesce, live.components)


class _Section:
    """A component with one trivial section."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.restored: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def persistence_type(self) -> PersistenceType:
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        return {"value": 1}

    def restore(self, state: Mapping[str, object]) -> None:
        self.restored = state


class _Unreadable(_Section):
    """A component that cannot read the section it wrote."""

    def restore(self, state: Mapping[str, object]) -> None:
        raise ValueError("cannot read my own section")


def _capture(coordinator: _Coordinator) -> dict[str, Mapping[str, object]]:
    """Read a coordinator's sections, as a checkpoint would."""
    with coordinator.quiesce.quiesced():
        return {c.name: c.capture() for c in coordinator.components}


class TestCapturesAreCopies:
    def test_mutating_a_component_does_not_change_its_capture(self):
        """The quiesce is released before the artifact is encoded, so a
        capture that aliased live state would be serialized mid-mutation.
        Copies are the contract; this is what enforces it.
        """
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1)]))
        live.controller.pin([_key(1)])
        live.controller.quota.set_quota("tenant-a", 4096)
        every_component = [
            live.directory,
            live.usage,
            live.gate,
            *live.controller.get_durable_components(),
        ]

        captured = {c.name: c.capture() for c in every_component}

        # Everything a live coordinator would go on to do while the
        # checkpoint from those captures is still being encoded.
        live.gate.ingest(_store(seq=2, keys=[_key(2), _key(3)]))
        live.controller.pin([_key(2)])
        live.controller.unpin([_key(1)])
        live.controller.quota.set_quota("tenant-b", 8192)
        live.gate.drop_instance("node-a")

        assert len(captured["key_directory"]["keys"]) == 1
        assert len(captured["cache_usage"]["placements"]) == 1
        assert len(captured["lru_order"]["buckets"][""]) == 1
        assert len(captured["pins"]["entries"]) == 1
        assert captured["quotas"]["limits"] == {"tenant-a": 4096}
        assert set(captured["stream_cursors"]["cursors"]) == {"node-a"}
