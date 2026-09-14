# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP server's memory-capacity declaration.

Capacity comes from config, not from the lazily grown heap, and a tier
spanning several mediums declares one compartment per medium.
"""

# Standard
from typing import cast
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType, ModuleMemoryCapacity, Tier
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    get_configured_capacity_bytes,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.l1_manager_protocol import L1ManagerProtocol
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import Event, EventType

GIB = 1 << 30


class _FakeAdapterConfig:
    """Stands in for an ``L2AdapterConfigBase``."""

    def __init__(self, shared: bool) -> None:
        self.shared = shared


class _FakeDescriptor:
    """Stands in for an ``AdapterDescriptor``."""

    def __init__(self, type_name: str, shared: bool) -> None:
        self.type_name = type_name
        self.config = _FakeAdapterConfig(shared)


class _FakeUsage:
    """Stands in for an ``AdapterUsage``."""

    def __init__(self, capacity_bytes: int) -> None:
        self.total_capacity_bytes = capacity_bytes


class _FakeAdapter:
    """An L2 adapter that reports a fixed capacity, or raises."""

    def __init__(self, capacity_bytes: int, fail: bool = False) -> None:
        self._capacity_bytes = capacity_bytes
        self._fail = fail

    def get_usage(self) -> _FakeUsage:
        if self._fail:
            raise RuntimeError("adapter unavailable")
        return _FakeUsage(self._capacity_bytes)


class _RecordingBus:
    """Captures what the publish path emits."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


class _StorageManagerStub:
    """Stands in for a ``StorageManager``, minus its pinned-memory ``__init__``."""

    def __init__(
        self,
        l1: dict[L1BackendType, int],
        adapters: list[tuple[_FakeDescriptor, _FakeAdapter]],
    ) -> None:
        # A real config, so the stub exercises the actual L1 derivation.
        self._l1_config = _config_yielding(l1)
        self._adapters = adapters
        self._lifecycle_lock = threading.Lock()
        self._event_bus = _RecordingBus()

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        return StorageManager._build_capacities(cast("StorageManager", self))

    def _publish_capacity_changed(self) -> None:
        StorageManager._publish_capacity_changed(cast("StorageManager", self))

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, _FakeDescriptor, _FakeAdapter]]:
        return [
            (index, desc, adapter)
            for index, (desc, adapter) in enumerate(self._adapters)
        ]


def _capacities(
    l1: dict[L1BackendType, int],
    adapters: list[tuple[_FakeDescriptor, _FakeAdapter]],
) -> list[ModuleMemoryCapacity]:
    """Run ``StorageManager._build_capacities`` against fakes.

    Args:
        l1: Configured L1 capacity per backing medium.
        adapters: The L2 adapters to report, as ``(descriptor, adapter)``.

    Returns:
        The capacities the method assembles.
    """
    return _StorageManagerStub(l1, adapters)._build_capacities()


def _config_yielding(capacities: dict[L1BackendType, int]) -> L1ManagerConfig:
    """Build a config whose derived capacity equals ``capacities``.

    Args:
        capacities: The per-medium result the config should produce.

    Returns:
        A matching :class:`L1ManagerConfig`.

    Raises:
        ValueError: If the combination is not expressible as one tier.
    """
    if set(capacities) == {L1BackendType.GDS}:
        return L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=0, use_lazy=True),
            gds_l1_config=GdsL1Config(
                size_in_bytes=capacities[L1BackendType.GDS],
                file_location="/tmp/gds-slab",
            ),
        )
    if L1BackendType.DEVDAX in capacities:
        return L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=capacities.get(L1BackendType.DRAM, 0),
                devdax_path="/dev/dax0.0",
                devdax_size_in_bytes=capacities[L1BackendType.DEVDAX],
                use_lazy=False,
                shm_name="",
            )
        )
    if set(capacities) <= {L1BackendType.DRAM}:
        return L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=capacities.get(L1BackendType.DRAM, 0), use_lazy=True
            )
        )
    raise ValueError(f"not expressible as one tier: {capacities}")


class TestConfiguredL1Capacity:
    """The single derivation of "how large is L1", from config alone."""

    def _config(self, **memory: object) -> L1ManagerConfig:
        """Build an L1ManagerConfig with the given memory-config fields."""
        defaults: dict[str, object] = {
            "size_in_bytes": 0,
            "devdax_path": None,
            "use_lazy": True,
        }
        defaults.update(memory)
        return L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(**defaults)  # type: ignore[arg-type]
        )

    def test_cpu_tier_reports_configured_size_not_grown_heap(self) -> None:
        # The lazy allocator grows, so its current heap is not the capacity.
        config = self._config(size_in_bytes=40 * GIB)
        assert get_configured_capacity_bytes(config) == {L1BackendType.DRAM: 40 * GIB}

    def test_unconfigured_tier_reports_nothing_rather_than_zero(self) -> None:
        assert get_configured_capacity_bytes(self._config()) == {}

    def test_pure_devdax_reports_one_medium(self) -> None:
        # An unset devdax size means the whole tier is Device-DAX.
        config = self._config(
            size_in_bytes=100 * GIB,
            devdax_path="/dev/dax0.0",
            use_lazy=False,
            shm_name="",
        )
        assert get_configured_capacity_bytes(config) == {
            L1BackendType.DEVDAX: 100 * GIB
        }

    def test_hybrid_devdax_splits_into_two_mediums(self) -> None:
        # L1 events tag placements per medium, so capacity must too.
        config = self._config(
            size_in_bytes=10 * GIB,
            devdax_path="/dev/dax0.0",
            devdax_size_in_bytes=100 * GIB,
            use_lazy=False,
            shm_name="",
        )
        assert get_configured_capacity_bytes(config) == {
            L1BackendType.DEVDAX: 100 * GIB,
            L1BackendType.DRAM: 10 * GIB,
        }

    def test_gds_tier_wins_over_the_dram_config(self) -> None:
        config = L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=40 * GIB, use_lazy=True),
            gds_l1_config=GdsL1Config(
                size_in_bytes=8 * GIB, file_location="/tmp/gds-slab"
            ),
        )
        assert get_configured_capacity_bytes(config) == {L1BackendType.GDS: 8 * GIB}

    def test_matches_the_devdax_manager_arena_split(self) -> None:
        # Mirrors DevDaxL1MemoryManager.__init__; catches drift in that split.
        memory_config = L1MemoryManagerConfig(
            size_in_bytes=10 * GIB,
            devdax_path="/dev/dax0.0",
            devdax_size_in_bytes=100 * GIB,
            use_lazy=False,
            shm_name="",
        )
        devdax_size = memory_config.devdax_size_in_bytes or memory_config.size_in_bytes
        local_size = (
            memory_config.size_in_bytes if memory_config.devdax_size_in_bytes else 0
        )
        derived = get_configured_capacity_bytes(
            L1ManagerConfig(memory_config=memory_config)
        )
        assert derived[L1BackendType.DEVDAX] == devdax_size
        assert derived[L1BackendType.DRAM] == local_size

    def test_total_matches_what_usage_telemetry_reports(self) -> None:
        # usage_telemetry sums the same derivation; drift would split the views.
        memory_config = L1MemoryManagerConfig(
            size_in_bytes=10 * GIB,
            devdax_path="/dev/dax0.0",
            devdax_size_in_bytes=100 * GIB,
            use_lazy=False,
            shm_name="",
        )
        config = L1ManagerConfig(memory_config=memory_config)
        assert sum(get_configured_capacity_bytes(config).values()) == (
            memory_config.size_in_bytes + memory_config.devdax_size_in_bytes
        )


class TestStorageManagerCapacities:
    def test_reports_l1_per_medium(self) -> None:
        found = _capacities(
            {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}, []
        )
        assert {(c.tier, c.backend, c.capacity_bytes) for c in found} == {
            (Tier.L1, "devdax", 100 * GIB),
            (Tier.L1, "dram", 10 * GIB),
        }
        assert all(c.shared is False for c in found)

    def test_reports_each_l2_adapter_with_its_shared_flag(self) -> None:
        found = _capacities(
            {L1BackendType.DRAM: 40 * GIB},
            [
                (_FakeDescriptor("fs", shared=False), _FakeAdapter(200 * GIB)),
                (_FakeDescriptor("s3", shared=True), _FakeAdapter(4000 * GIB)),
            ],
        )
        by_backend = {c.backend: c for c in found}
        assert by_backend["fs"].tier == Tier.L2
        assert by_backend["fs"].shared is False
        assert by_backend["s3"].shared is True
        assert by_backend["s3"].capacity_bytes == 4000 * GIB

    def test_adapter_without_a_configured_cap_reports_zero(self) -> None:
        # fs / mooncake / p2p / sagemaker return 0: undeclared, never "full".
        found = _capacities(
            {}, [(_FakeDescriptor("fs", shared=False), _FakeAdapter(0))]
        )
        assert [c.capacity_bytes for c in found] == [0]

    def test_failing_adapter_is_omitted_not_reported_wrong(self) -> None:
        found = _capacities(
            {},
            [
                (_FakeDescriptor("fs", shared=False), _FakeAdapter(0, fail=True)),
                (_FakeDescriptor("s3", shared=False), _FakeAdapter(9 * GIB)),
            ],
        )
        assert [c.backend for c in found] == ["s3"]

    def test_server_with_nothing_configured_declares_nothing(self) -> None:
        assert _capacities({}, []) == []


@pytest.mark.parametrize(
    "capacities",
    [
        {L1BackendType.DRAM: 40 * GIB},
        {L1BackendType.GDS: 8 * GIB},
        {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB},
    ],
)
def test_backend_names_match_the_cache_event_vocabulary(capacities: dict) -> None:
    # Capacity joins usage on (tier, backend); these strings must match events.
    found = _capacities(capacities, [])
    assert {c.backend for c in found} == {b.value for b in capacities}


class TestReportStatusSharesTheSource:
    """``report_status`` and the capacity API must report the same size."""

    def _l1_manager(self, configured: dict[L1BackendType, int]) -> L1Manager:
        """Build an L1Manager over a fake memory manager.

        Args:
            configured: Capacity the fake reports per backing medium.

        Returns:
            A manager whose ``report_status`` is callable.
        """

        class _MemoryManager:
            def get_memory_usage(self) -> tuple[int, int]:
                # The grown heap, deliberately unequal to the configured total.
                return (1 * GIB, 3 * GIB)

            def memcheck(self) -> bool:
                return True

        manager = L1Manager.__new__(L1Manager)
        manager._memory_manager = cast("L1ManagerProtocol", _MemoryManager())
        # report_status reads the total precomputed at construction.
        manager._configured_capacity_bytes = sum(
            get_configured_capacity_bytes(_config_yielding(configured)).values()
        )
        manager._objects = {}
        manager._write_ttl_seconds = 600
        manager._read_ttl_seconds = 600
        # report_status is lock-guarded; the pinned-memory __init__ is skipped.
        manager._lock = threading.Lock()
        return manager

    def test_status_reports_configured_separately_from_grown_heap(self) -> None:
        manager = self._l1_manager({L1BackendType.DRAM: 40 * GIB})
        status = manager.report_status()
        assert status["memory_configured_bytes"] == 40 * GIB
        # The pre-existing field keeps its old meaning for old consumers.
        assert status["memory_total_bytes"] == 3 * GIB

    def test_status_sums_a_hybrid_tier(self) -> None:
        manager = self._l1_manager(
            {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}
        )
        assert manager.report_status()["memory_configured_bytes"] == 110 * GIB

    def test_status_and_capacity_report_agree(self) -> None:
        # The point of one shared derivation: the two consumers of it --
        # report_status and the coordinator capacity report -- cannot drift.
        configured = {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}
        manager = self._l1_manager(configured)
        reported = sum(
            module.capacity_bytes
            for module in _capacities(configured, [])
            if module.tier == Tier.L1
        )
        assert manager.report_status()["memory_configured_bytes"] == reported

    def test_unconfigured_tier_reports_zero_not_the_heap(self) -> None:
        manager = self._l1_manager({})
        assert manager.report_status()["memory_configured_bytes"] == 0


class TestCapacityChangePublishing:
    """Runtime reconfiguration announces the new topology on the bus.

    Without this, a coordinator declaration made at registration keeps the
    boot capacity for the life of the process.
    """

    def _stub(self) -> _StorageManagerStub:
        """A stub wired with the bus plumbing the publish path needs."""
        return _StorageManagerStub({L1BackendType.DRAM: 40 * GIB}, [])

    def test_publish_emits_one_event_per_call(self) -> None:
        # The snapshot carries no revision: the cache-event subscriber
        # numbers declarations as it emits them, on the one bus thread.
        stub = self._stub()
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        assert len(stub._event_bus.events) == 2
        assert all(
            not hasattr(e.metadata["snapshot"], "revision")
            for e in stub._event_bus.events
        )

    def test_published_event_carries_the_whole_declaration(self) -> None:
        # A delta would leave the coordinator permanently wrong if dropped.
        stub = self._stub()
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        snapshot = stub._event_bus.events[0].metadata["snapshot"]
        assert [(m.tier, m.backend, m.capacity_bytes) for m in snapshot.modules] == [
            (Tier.L1, "dram", 40 * GIB)
        ]

    def test_event_type_is_the_capacity_one(self) -> None:
        stub = self._stub()
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        assert stub._event_bus.events[0].event_type == EventType.SM_CAPACITY_CHANGED

    def test_snapshot_carries_the_whole_topology(self) -> None:
        stub = self._stub()
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        snapshot = stub._event_bus.events[0].metadata["snapshot"]
        assert len(snapshot.modules) == 1

    def test_publish_capacity_declares_without_any_reconfiguration(self) -> None:
        # The only path by which a server that never reconfigures reaches
        # the coordinator at all.
        stub = self._stub()
        StorageManager.publish_capacity(cast("StorageManager", stub))
        assert len(stub._event_bus.events) == 1
        snapshot = stub._event_bus.events[0].metadata["snapshot"]
        assert [(m.tier, m.backend) for m in snapshot.modules] == [(Tier.L1, "dram")]


class TestCapacityPublishTakesNoLock:
    """Publishing must not queue behind adapter lifecycle work."""

    def test_publish_does_not_take_the_lifecycle_lock(self) -> None:
        # add/delete/reconfigure hold _lifecycle_lock, and delete can hold
        # it for its full timeout. Publishing runs on every registration, so
        # taking that lock would stall registration behind a teardown.
        stub = _StorageManagerStub({L1BackendType.DRAM: 8 * GIB}, [])
        stub._lifecycle_lock.acquire()
        try:
            StorageManager.publish_capacity(cast("StorageManager", stub))
        finally:
            stub._lifecycle_lock.release()
        assert len(stub._event_bus.events) == 1
