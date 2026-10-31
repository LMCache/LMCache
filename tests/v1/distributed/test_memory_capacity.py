# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP server's memory-capacity declaration.

CPU and GDS capacity comes from config rather than the lazily grown heap;
Device-DAX capacity follows its active arenas. Each L1 manager has one
medium; the server declares one compartment per medium.
"""

# Standard
from dataclasses import replace
from typing import cast
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ModuleMemoryCapacity,
    Tier,
)
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    get_configured_capacity_bytes,
    l1_medium_name,
)
from lmcache.v1.distributed.error import L1ReconfigureError
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager.l1_manager_protocol import (
    L1MemoryManagerProtocol,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
)
from lmcache.v1.mp_observability.event import Event, EventType

GIB = 1 << 30


class _FakeAdapterConfig:
    """Stands in for an ``L2AdapterConfigBase``."""

    def __init__(self, shared: bool) -> None:
        self.shared = shared


class _FakeDescriptor:
    """Stands in for an ``L2AdapterDescriptor``."""

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


class _FakeL1Manager:
    """An L1 manager of one medium that reports a fixed capacity."""

    def __init__(self, backend_type: L1BackendType, capacity_bytes: int) -> None:
        self.backend_type = backend_type
        self._capacity_bytes = capacity_bytes

    def get_capacity_bytes(self) -> int:
        return self._capacity_bytes

    def get_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        return []

    def get_devdax_arena_status(
        self, device_path: str, offset_in_bytes: int = 0
    ) -> DevDaxArenaStatus:
        for status in self.get_devdax_arena_statuses():
            if status.device_path == device_path:
                return status
        raise L1ReconfigureError(404, f"no Device-DAX arena mapped at {device_path}")


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
        l1: list[_FakeL1Manager],
        adapters: list[tuple[_FakeDescriptor, _FakeAdapter]],
    ) -> None:
        self._l1_managers = list(l1)
        self._adapters = adapters
        self._adapter_descriptors = {
            index: desc for index, (desc, _adapter) in enumerate(adapters)
        }
        self._lifecycle_lock = threading.Lock()
        self._adapters_lock = threading.Lock()
        self._capacity_publish_lock = threading.Lock()
        self._event_bus = _RecordingBus()

    def _devdax_l1(self) -> _FakeL1Manager:
        """The stub's first fake stands in for the Device-DAX L1 of the routes."""
        return self._l1_managers[0]

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        return StorageManager._build_capacities(cast("StorageManager", self))

    def _publish_capacity_changed(self) -> None:
        StorageManager._publish_capacity_changed(cast("StorageManager", self))

    def _single_region_adapter_names(self) -> list[str]:
        return StorageManager._single_region_adapter_names(cast("StorageManager", self))

    def _l2_device_owner_names(
        self, _device_path: str, _offset: int = 0, _size: int = 0
    ) -> list[str]:
        return []

    def _l1_devdax_arena_is_active(
        self, device_path: str, offset_in_bytes: int = 0
    ) -> bool:
        return StorageManager._l1_devdax_arena_is_active(
            cast("StorageManager", self), device_path, offset_in_bytes
        )

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, _FakeDescriptor, _FakeAdapter]]:
        return [
            (index, desc, adapter)
            for index, (desc, adapter) in enumerate(self._adapters)
        ]


def _capacities(
    l1: list[_FakeL1Manager],
    adapters: list[tuple[_FakeDescriptor, _FakeAdapter]],
) -> list[ModuleMemoryCapacity]:
    """Run ``StorageManager._build_capacities`` against fakes.

    Args:
        l1: The L1 managers to report, each with its medium and capacity.
        adapters: The L2 adapters to report, as ``(descriptor, adapter)``.

    Returns:
        The capacities the method assembles.
    """
    return _StorageManagerStub(l1, adapters)._build_capacities()


def _dram_config(capacity_bytes: int) -> L1ManagerConfig:
    """Build a lazy DRAM config whose configured capacity is ``capacity_bytes``."""
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=capacity_bytes, use_lazy=True)
    )


class TestConfiguredL1Capacity:
    """The boot-capacity derivation for each L1 configuration."""

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
        assert get_configured_capacity_bytes(config) == 40 * GIB
        assert l1_medium_name(config) == "dram"

    def test_unconfigured_tier_reports_zero(self) -> None:
        assert get_configured_capacity_bytes(self._config()) == 0

    def test_devdax_reports_its_device_size(self) -> None:
        config = self._config(
            size_in_bytes=100 * GIB,
            devdax_path="/dev/dax0.0",
            use_lazy=False,
            shm_name="",
        )
        assert get_configured_capacity_bytes(config) == 100 * GIB
        assert l1_medium_name(config) == "devdax"

    def test_gds_tier_wins_over_the_dram_config(self) -> None:
        config = L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=40 * GIB, use_lazy=True),
            gds_l1_config=GdsL1Config(
                size_in_bytes=8 * GIB, file_location="/tmp/gds-slab"
            ),
        )
        assert get_configured_capacity_bytes(config) == 8 * GIB
        assert l1_medium_name(config) == "gds"


class TestStorageManagerCapacities:
    def test_reports_l1_per_medium(self) -> None:
        # Managers of one medium share a compartment; media stay apart.
        found = _capacities(
            [
                _FakeL1Manager(L1BackendType.DEVDAX, 100 * GIB),
                _FakeL1Manager(L1BackendType.DRAM, 10 * GIB),
                _FakeL1Manager(L1BackendType.DRAM, 6 * GIB),
            ],
            [],
        )
        assert {(c.tier, c.backend, c.capacity_bytes) for c in found} == {
            (Tier.L1, "devdax", 100 * GIB),
            (Tier.L1, "dram", 16 * GIB),
        }
        assert all(c.shared is False for c in found)

    def test_reports_each_l2_adapter_with_its_shared_flag(self) -> None:
        found = _capacities(
            [_FakeL1Manager(L1BackendType.DRAM, 40 * GIB)],
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
            [], [(_FakeDescriptor("fs", shared=False), _FakeAdapter(0))]
        )
        assert [c.capacity_bytes for c in found] == [0]

    def test_failing_adapter_is_omitted_not_reported_wrong(self) -> None:
        found = _capacities(
            [],
            [
                (_FakeDescriptor("fs", shared=False), _FakeAdapter(0, fail=True)),
                (_FakeDescriptor("s3", shared=False), _FakeAdapter(9 * GIB)),
            ],
        )
        assert [c.backend for c in found] == ["s3"]

    def test_server_with_nothing_configured_declares_nothing(self) -> None:
        assert _capacities([], []) == []


@pytest.mark.parametrize(
    "managers",
    [
        [_FakeL1Manager(L1BackendType.DRAM, 40 * GIB)],
        [_FakeL1Manager(L1BackendType.GDS, 8 * GIB)],
        [
            _FakeL1Manager(L1BackendType.DEVDAX, 100 * GIB),
            _FakeL1Manager(L1BackendType.DRAM, 10 * GIB),
        ],
    ],
)
def test_backend_names_match_the_cache_event_vocabulary(
    managers: list[_FakeL1Manager],
) -> None:
    # Capacity joins usage on (tier, backend); these strings must match events.
    found = _capacities(managers, [])
    assert {c.backend for c in found} == {m.backend_type.value for m in managers}


class TestReportStatusSharesTheSource:
    """``report_status`` and the capacity API must report the same size."""

    def _l1_manager(self, configured: int) -> BaseL1Manager:
        """Build an L1Manager over a fake memory manager.

        Args:
            configured: The configured DRAM capacity in bytes.

        Returns:
            A manager whose ``report_status`` is callable.
        """

        class _MemoryManager:
            def get_memory_usage(self) -> tuple[int, int]:
                # The grown heap, deliberately unequal to the configured total.
                return (1 * GIB, 3 * GIB)

            def memcheck(self) -> bool:
                return True

        class _FakeMediumL1Manager(BaseL1Manager):
            @property
            def backend_type(self) -> L1BackendType:
                return L1BackendType.DRAM

            def get_memory_tags(self) -> MemoryTags:
                return MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)

        # No pinned memory is touched: the fake only answers usage queries.
        return _FakeMediumL1Manager(
            _dram_config(configured), cast("L1MemoryManagerProtocol", _MemoryManager())
        )

    def test_status_reports_configured_separately_from_grown_heap(self) -> None:
        manager = self._l1_manager(40 * GIB)
        status = manager.report_status()
        assert status["memory_configured_bytes"] == 40 * GIB
        assert status["backend"] == "dram"
        # The pre-existing field keeps its old meaning for old consumers.
        assert status["memory_total_bytes"] == 3 * GIB

    def test_status_and_capacity_report_agree(self) -> None:
        # Both consumers use the same point-in-time capacity derivation.
        manager = self._l1_manager(10 * GIB)
        reported = sum(
            module.capacity_bytes
            for module in _capacities([cast(_FakeL1Manager, manager)], [])
            if module.tier == Tier.L1
        )
        assert manager.report_status()["memory_configured_bytes"] == reported

    def test_unconfigured_tier_reports_zero_not_the_heap(self) -> None:
        manager = self._l1_manager(0)
        assert manager.report_status()["memory_configured_bytes"] == 0


class TestCapacityChangePublishing:
    """Runtime reconfiguration announces the new topology on the bus.

    Without this, a coordinator declaration made at registration keeps the
    boot capacity for the life of the process.
    """

    def _stub(self) -> _StorageManagerStub:
        """A stub wired with the bus plumbing the publish path needs."""
        return _StorageManagerStub([_FakeL1Manager(L1BackendType.DRAM, 40 * GIB)], [])

    def test_published_event_carries_the_whole_declaration(self) -> None:
        # A delta would leave the coordinator permanently wrong if dropped.
        stub = self._stub()
        StorageManager._publish_capacity_changed(cast("StorageManager", stub))
        snapshot = stub._event_bus.events[0].metadata["snapshot"]
        assert [(m.tier, m.backend, m.capacity_bytes) for m in snapshot.modules] == [
            (Tier.L1, "dram", 40 * GIB)
        ]

    def test_reconfigure_publishes_post_operation_capacity(self) -> None:
        active = DevDaxArenaStatus(
            "/dev/dax0.1", 4096, 0, 4096, 0, DevDaxArenaState.ACTIVE, False
        )

        class _ReconfigurableL1Manager(_FakeL1Manager):
            def __init__(self) -> None:
                super().__init__(L1BackendType.DEVDAX, 4096)

            def get_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
                return [active] if self._capacity_bytes == 8192 else []

            def add_devdax_device(
                self, _device_path: str, _size_in_bytes: int, _offset: int = 0
            ) -> DevDaxArenaStatus:
                self._capacity_bytes = 8192
                return active

            def remove_devdax_device(
                self, _device_path: str, _mode: object, _offset: int = 0
            ) -> DevDaxArenaStatus:
                if self._capacity_bytes == 4096:
                    raise ValueError("Device-DAX arena is not mapped")
                self._capacity_bytes = 4096
                return replace(active, state=DevDaxArenaState.REMOVED)

        stub = _StorageManagerStub([_ReconfigurableL1Manager()], [])

        StorageManager.add_l1_devdax_device(
            cast("StorageManager", stub), active.device_path, 4096
        )
        StorageManager.remove_l1_devdax_device(
            cast("StorageManager", stub), active.device_path
        )
        with pytest.raises(ValueError, match="not mapped"):
            StorageManager.remove_l1_devdax_device(
                cast("StorageManager", stub), active.device_path
            )

        events = stub._event_bus.events
        assert [event.event_type for event in events] == [
            EventType.SM_CAPACITY_CHANGED,
            EventType.SM_CAPACITY_CHANGED,
        ]
        assert [
            next(
                module.capacity_bytes
                for module in event.metadata["snapshot"].modules
                if module.tier is Tier.L1 and module.backend == "devdax"
            )
            for event in events
        ] == [8192, 4096]

    def test_remove_transition_is_published_when_total_is_unchanged(self) -> None:
        class _FailingRemoveL1Manager(_FakeL1Manager):
            def __init__(self) -> None:
                super().__init__(L1BackendType.DEVDAX, 40 * GIB)
                self._target_state = DevDaxArenaState.ACTIVE

            def get_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
                return [
                    DevDaxArenaStatus(
                        device_path="/dev/dax0.1",
                        size_in_bytes=20 * GIB,
                        used_bytes=0,
                        free_bytes=20 * GIB,
                        active_allocations=0,
                        state=self._target_state,
                        is_primary=False,
                    )
                ]

            def remove_devdax_device(
                self, _device_path: str, _mode: object, _offset: int = 0
            ) -> None:
                self._target_state = DevDaxArenaState.DRAINING
                raise RuntimeError("cleanup failed after drain started")

        stub = _StorageManagerStub([_FailingRemoveL1Manager()], [])

        with pytest.raises(RuntimeError, match="cleanup failed"):
            StorageManager.remove_l1_devdax_device(
                cast("StorageManager", stub), "/dev/dax0.1"
            )

        assert len(stub._event_bus.events) == 1
        event = stub._event_bus.events[0]
        assert event.event_type == EventType.SM_CAPACITY_CHANGED
        # A concurrent same-sized add can keep the aggregate unchanged. The
        # target arena's ACTIVE -> DRAINING transition still needs an event to
        # supersede any older declaration already queued by that add.
        assert event.metadata["snapshot"].modules[0].capacity_bytes == 40 * GIB

    def test_publish_capacity_declares_without_any_reconfiguration(self) -> None:
        # The only path by which a server that never reconfigures reaches
        # the coordinator at all.
        stub = self._stub()
        StorageManager.publish_capacity(cast("StorageManager", stub))
        assert len(stub._event_bus.events) == 1
        snapshot = stub._event_bus.events[0].metadata["snapshot"]
        assert [(m.tier, m.backend) for m in snapshot.modules] == [(Tier.L1, "dram")]


class TestCapacityPublishDoesNotTakeLifecycleLock:
    """Publishing must not queue behind adapter lifecycle work."""

    def test_publish_does_not_take_the_lifecycle_lock(self) -> None:
        # Add/delete hold _lifecycle_lock, and delete can hold it for its full
        # timeout. Publishing runs on every registration, so taking that lock
        # would stall registration behind a teardown.
        stub = _StorageManagerStub([_FakeL1Manager(L1BackendType.DRAM, 8 * GIB)], [])
        stub._lifecycle_lock.acquire()
        try:
            StorageManager.publish_capacity(cast("StorageManager", stub))
        finally:
            stub._lifecycle_lock.release()
        assert len(stub._event_bus.events) == 1
