# SPDX-License-Identifier: Apache-2.0
"""Tests for the distributed storage memory usage snapshot API."""

# Standard
from dataclasses import dataclass
from typing import Any, cast
import threading

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.storage_usage import StorageHealth, StorageUsageSnapshot
from lmcache.v1.distributed.tiers import Tier


class _FakeL1Manager:
    def __init__(
        self,
        used_bytes: int,
        capacity_bytes: int,
        health: object = True,
    ) -> None:
        self.used_bytes = used_bytes
        self.capacity_bytes = capacity_bytes
        self.health = health
        self.usage_error: Exception | None = None
        self.health_error: Exception | None = None

    def get_memory_usage(self) -> tuple[int, int]:
        if self.usage_error is not None:
            raise self.usage_error
        return self.used_bytes, self.capacity_bytes

    def memcheck(self) -> object:
        if self.health_error is not None:
            raise self.health_error
        return self.health


class _FakeAdapter:
    def __init__(
        self,
        used_bytes: int,
        capacity_bytes: int,
        health: object = True,
    ) -> None:
        self.used_bytes = used_bytes
        self.capacity_bytes = capacity_bytes
        self.health = health
        self.usage_error: Exception | None = None
        self.health_error: Exception | None = None

    def get_usage(self) -> AdapterUsage:
        if self.usage_error is not None:
            raise self.usage_error
        return AdapterUsage(
            total_bytes_used=self.used_bytes,
            total_capacity_bytes=self.capacity_bytes,
        )

    def report_status(self) -> dict[str, object]:
        if self.health_error is not None:
            raise self.health_error
        return {"is_healthy": self.health}


@dataclass(frozen=True)
class _FakeAdapterConfig:
    eviction_config: EvictionConfig | None = None


@dataclass(frozen=True)
class _FakeAdapterDescriptor:
    index: int
    type_name: str
    config: _FakeAdapterConfig


def _make_storage_manager(
    l1_manager: _FakeL1Manager,
    adapters: list[tuple[int, _FakeAdapterDescriptor, _FakeAdapter]],
) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager_any = cast(Any, manager)
    manager_any._l1_manager = l1_manager
    manager_any._l1_eviction_trigger_watermark = 0.8
    manager_any._l1_eviction_policy = "LRU"
    manager_any._lifecycle_lock = threading.Lock()
    manager_any._adapters_lock = threading.Lock()
    manager_any._l2_adapters = {
        adapter_id: adapter for adapter_id, _descriptor, adapter in adapters
    }
    manager_any._adapter_descriptors = {
        adapter_id: descriptor for adapter_id, descriptor, _adapter in adapters
    }
    return manager


def _descriptor(
    adapter_id: int,
    backend_type: str = "fake",
    eviction_config: EvictionConfig | None = None,
) -> _FakeAdapterDescriptor:
    return _FakeAdapterDescriptor(
        index=adapter_id,
        type_name=backend_type,
        config=_FakeAdapterConfig(eviction_config=eviction_config),
    )


def test_memory_usage_snapshot_reports_healthy_l1_and_l2() -> None:
    l2_eviction = EvictionConfig(eviction_policy="noop", trigger_watermark=0.9)
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=20, capacity_bytes=100),
        [
            (
                3,
                _descriptor(3, backend_type="disk", eviction_config=l2_eviction),
                _FakeAdapter(used_bytes=40, capacity_bytes=200),
            )
        ],
    )

    snapshot = manager.get_memory_usage_snapshot()

    assert snapshot.l1.tier is Tier.L1
    assert snapshot.l1.adapter_id is None
    assert snapshot.l1.backend_type is None
    assert snapshot.l1.used_bytes == 20
    assert snapshot.l1.capacity_bytes == 100
    assert snapshot.l1.trigger_watermark == 0.8
    assert snapshot.l1.eviction_policy == "LRU"
    assert snapshot.l1.health is StorageHealth.OK
    assert snapshot.l1.collection_errors == ()

    assert len(snapshot.l2) == 1
    l2 = snapshot.l2[0]
    assert l2.tier is Tier.L2
    assert l2.adapter_id == 3
    assert l2.backend_type == "disk"
    assert l2.used_bytes == 40
    assert l2.capacity_bytes == 200
    assert l2.trigger_watermark == 0.9
    assert l2.eviction_policy == "noop"
    assert l2.health is StorageHealth.OK
    assert l2.collection_errors == ()


def test_unknown_capacity_preserves_known_usage() -> None:
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=11, capacity_bytes=0),
        [(0, _descriptor(0), _FakeAdapter(used_bytes=22, capacity_bytes=0))],
    )

    snapshot = manager.get_memory_usage_snapshot()

    assert snapshot.l1.used_bytes == 11
    assert snapshot.l1.capacity_bytes is None
    assert snapshot.l2[0].used_bytes == 22
    assert snapshot.l2[0].capacity_bytes is None


def test_adapter_usage_failure_does_not_hide_health_or_other_adapters() -> None:
    broken = _FakeAdapter(used_bytes=10, capacity_bytes=100, health=False)
    broken.usage_error = RuntimeError("usage failed")
    healthy = _FakeAdapter(used_bytes=30, capacity_bytes=100, health=True)
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=1, capacity_bytes=10),
        [
            (0, _descriptor(0), broken),
            (1, _descriptor(1), healthy),
        ],
    )

    snapshot = manager.get_memory_usage_snapshot()

    assert snapshot.l2[0].used_bytes is None
    assert snapshot.l2[0].capacity_bytes is None
    assert snapshot.l2[0].health is StorageHealth.FAILED
    assert snapshot.l2[0].collection_errors == ("usage_unavailable",)
    assert snapshot.l2[1].used_bytes == 30
    assert snapshot.l2[1].health is StorageHealth.OK
    assert snapshot.l2[1].collection_errors == ()


def test_adapter_status_failure_preserves_usage() -> None:
    adapter = _FakeAdapter(used_bytes=70, capacity_bytes=100)
    adapter.health_error = RuntimeError("status failed")
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=1, capacity_bytes=10),
        [(0, _descriptor(0), adapter)],
    )

    snapshot = manager.get_memory_usage_snapshot()

    assert snapshot.l2[0].used_bytes == 70
    assert snapshot.l2[0].capacity_bytes == 100
    assert snapshot.l2[0].health is StorageHealth.UNKNOWN
    assert snapshot.l2[0].collection_errors == ("health_unavailable",)


def test_duplicate_backend_types_keep_stable_adapter_ids_and_order() -> None:
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=1, capacity_bytes=10),
        [
            (9, _descriptor(9, backend_type="disk"), _FakeAdapter(9, 100)),
            (2, _descriptor(2, backend_type="disk"), _FakeAdapter(2, 100)),
        ],
    )

    snapshot = manager.get_memory_usage_snapshot()

    assert [entry.adapter_id for entry in snapshot.l2] == [2, 9]
    assert [entry.backend_type for entry in snapshot.l2] == ["disk", "disk"]
    assert [entry.used_bytes for entry in snapshot.l2] == [2, 9]


def test_each_memory_usage_snapshot_reads_current_runtime_state() -> None:
    l1 = _FakeL1Manager(used_bytes=10, capacity_bytes=100)
    adapter = _FakeAdapter(used_bytes=20, capacity_bytes=200)
    manager = _make_storage_manager(l1, [(0, _descriptor(0), adapter)])

    first = manager.get_memory_usage_snapshot()
    l1.used_bytes = 30
    replacement = _FakeAdapter(used_bytes=40, capacity_bytes=400)
    manager_any = cast(Any, manager)
    with manager_any._adapters_lock:
        manager_any._l2_adapters = {2: replacement}
        manager_any._adapter_descriptors = {2: _descriptor(2, "replacement")}
    second = manager.get_memory_usage_snapshot()

    assert first.l1.used_bytes == 10
    assert [entry.adapter_id for entry in first.l2] == [0]
    assert first.l2[0].used_bytes == 20
    assert second.l1.used_bytes == 30
    assert [entry.adapter_id for entry in second.l2] == [2]
    assert second.l2[0].backend_type == "replacement"
    assert second.l2[0].used_bytes == 40


def test_l1_usage_and_health_failures_are_collected_independently() -> None:
    l1 = _FakeL1Manager(used_bytes=10, capacity_bytes=100, health=False)
    l1.usage_error = RuntimeError("usage failed")
    manager = _make_storage_manager(l1, [])

    usage_failed = manager.get_memory_usage_snapshot()

    assert usage_failed.l1.used_bytes is None
    assert usage_failed.l1.health is StorageHealth.FAILED
    assert usage_failed.l1.collection_errors == ("usage_unavailable",)

    l1.usage_error = None
    l1.health_error = RuntimeError("health failed")
    health_failed = manager.get_memory_usage_snapshot()

    assert health_failed.l1.used_bytes == 10
    assert health_failed.l1.capacity_bytes == 100
    assert health_failed.l1.health is StorageHealth.UNKNOWN
    assert health_failed.l1.collection_errors == ("health_unavailable",)


def test_snapshot_holds_adapter_lifecycle_until_collection_finishes() -> None:
    entered_usage = threading.Event()
    release_usage = threading.Event()
    mutation_attempted = threading.Event()
    mutation_complete = threading.Event()

    class _BlockingAdapter(_FakeAdapter):
        def get_usage(self) -> AdapterUsage:
            entered_usage.set()
            assert release_usage.wait(timeout=2)
            return super().get_usage()

    adapter = _BlockingAdapter(used_bytes=20, capacity_bytes=100)
    manager = _make_storage_manager(
        _FakeL1Manager(used_bytes=1, capacity_bytes=10),
        [(0, _descriptor(0), adapter)],
    )
    manager_any = cast(Any, manager)
    snapshot_result: list[StorageUsageSnapshot] = []

    snapshot_thread = threading.Thread(
        target=lambda: snapshot_result.append(manager.get_memory_usage_snapshot())
    )

    def mutate_adapters() -> None:
        mutation_attempted.set()
        with manager_any._lifecycle_lock:
            with manager_any._adapters_lock:
                manager_any._l2_adapters.clear()
                manager_any._adapter_descriptors.clear()
        mutation_complete.set()

    mutation_thread = threading.Thread(target=mutate_adapters)
    snapshot_thread.start()
    assert entered_usage.wait(timeout=2)
    mutation_thread.start()
    assert mutation_attempted.wait(timeout=2)
    assert not mutation_complete.wait(timeout=0.1)

    release_usage.set()
    snapshot_thread.join(timeout=2)
    mutation_thread.join(timeout=2)

    assert not snapshot_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert mutation_complete.is_set()
    assert len(snapshot_result) == 1
    assert snapshot_result[0].l2[0].adapter_id == 0
    assert snapshot_result[0].l2[0].used_bytes == 20
