# SPDX-License-Identifier: Apache-2.0
"""Every piece of persisted state restores itself.

Nothing is rebuilt by re-delivering synthesized events: a restore is a
per-component load, so no component depends on another being restored
first, and no view is left empty because a replay missed it.
"""

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
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
from tests.v1.mp_coordinator.persistence.conftest import capture_consistently


class _Coordinator:
    """The consumers a coordinator wires up, plus their durable components."""

    def __init__(self) -> None:
        self.directory = KeyDirectory()
        self.usage = CacheUsageManager()
        self.controller = FleetEvictionController(usage_manager=self.usage)
        self.broadcaster = CacheEventBroadcaster()
        self.broadcaster.register_consumer(self.directory)
        self.broadcaster.register_consumer(self.usage)
        self.broadcaster.register_consumer(self.controller)
        quiesce = QuiesceLock()
        self.gate = EventGate(self.broadcaster, quiesce)
        self.quiesce = quiesce

    def components(self):
        return [
            self.directory,
            self.usage,
            self.gate,
            *self.controller.get_durable_components(),
        ]

    def capture(self) -> dict[str, object]:
        return dict(capture_consistently(self.quiesce, self.components()))

    def restore(self, captured: dict[str, object]) -> None:
        for component in self.components():
            component.restore(captured[component.name])  # type: ignore[arg-type]


def _key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_id.to_bytes(4, "big"),
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


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
        backend="fs" if tier == Tier.L2 else "dram",
        entries=[
            CacheEventEntry(
                key=k.to_encoded_object_key(),
                size_bytes=1024,
                # Real token content, so bindings ride the round trip too.
                token_ids=[chunk_id, chunk_id + 1, chunk_id + 2],
                token_offset=chunk_id * 3,
            )
            for chunk_id, k in enumerate(keys, start=1)
        ],
    )


class TestRestoreWithoutReplay:
    def test_a_restored_coordinator_matches_the_captured_one(self):
        """The whole state comes back from the sections alone."""
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1), _key(2)]))
        live.gate.ingest(_store(seq=2, keys=[_key(3)], tier=Tier.L1))
        live.controller.pin([_key(1)])
        live.controller.quota.set_quota("tenant-a", 8192)

        captured = live.capture()
        restarted = _Coordinator()
        restarted.restore(captured)

        assert restarted.capture() == captured

    def test_the_usage_view_comes_back_without_being_replayed(self):
        """The view a replay used to rebuild now restores itself, so
        quota enforcement is armed before the first new batch."""
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1), _key(2)]))
        assert live.usage.get_salt_bytes(Tier.L2, "") == 2048

        restarted = _Coordinator()
        restarted.restore(live.capture())

        assert restarted.usage.get_salt_bytes(Tier.L2, "") == 2048
        assert restarted.usage.get_key_bytes(Tier.L2, _key(1)) == 1024

    def test_restored_cursors_fence_a_restarted_emitter(self):
        """Placements are only fenceable if the cursor that dates them
        comes back too."""
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1)], tier=Tier.L1))
        assert live.directory.lookup([_key(1)])[0]

        restarted = _Coordinator()
        restarted.restore(live.capture())
        assert restarted.directory.lookup([_key(1)])[0], "restored L1 placement"

        # The emitter comes back with a higher incarnation.
        restarted.gate.ingest(
            _store(seq=1, keys=[_key(9)], tier=Tier.L1, incarnation=8)
        )

        assert restarted.directory.lookup([_key(1)]) == [[]], (
            "incarnation 8 should have fenced incarnation 7's L1 slice"
        )

    def test_restoring_twice_is_refused(self):
        """Every component guards its own load, so a second restore
        cannot silently double-count."""
        live = _Coordinator()
        live.gate.ingest(_store(seq=1, keys=[_key(1)]))
        captured = live.capture()

        restarted = _Coordinator()
        restarted.restore(captured)

        for component in restarted.components():
            if component.name in ("pins", "quotas", "lru_order"):
                continue  # replacing these is idempotent by construction
            try:
                component.restore(captured[component.name])  # type: ignore[arg-type]
            except ValueError:
                continue
            raise AssertionError(f"{component.name} allowed a second restore")
