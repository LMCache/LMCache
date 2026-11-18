# SPDX-License-Identifier: Apache-2.0
"""StorageManager wiring of the L2 health monitor.

An adapter type whose probe results the test scripts is registered, so the
manager can be built from a normal config. What is checked is what operators
see: ``report_status`` and the adapters' probe traffic, across startup,
runtime add/delete and ``close``.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2HealthConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.health import HealthProbe, ProbeStatus
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager

_TYPE = "scripted_probe_mock"


class _ScriptedProbeConfig(MockL2AdapterConfig):
    """Config of the scripted adapter type."""

    @classmethod
    def from_dict(cls, d: dict) -> "_ScriptedProbeConfig":
        return cls(max_size_gb=0.01, mock_bandwidth_gb=10.0)


class _ScriptedAdapter(MockL2Adapter):
    """A mock adapter that supports probing; ``backend_up`` is the script."""

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self.backend_up = True
        self.probe_calls = 0

    @property
    def supports_health_probe(self) -> bool:
        return True

    def probe_health(self, timeout_s: float) -> HealthProbe:
        self.probe_calls += 1
        if self.backend_up:
            return HealthProbe(ProbeStatus.OK)
        return HealthProbe(ProbeStatus.UNREACHABLE, "scripted outage")


# Every adapter built by the factory, in creation order.
_created: list[_ScriptedAdapter] = []


def _build(config: L2AdapterConfigBase, l1_memory_desc=None) -> _ScriptedAdapter:
    assert isinstance(config, MockL2AdapterConfig)
    adapter = _ScriptedAdapter(config)
    _created.append(adapter)
    return adapter


register_l2_adapter_type(_TYPE, _ScriptedProbeConfig)
register_l2_adapter_factory(_TYPE, _build)


def _manager_config(num_adapters: int, **health) -> StorageManagerConfig:
    l1 = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=False,
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    return StorageManagerConfig(
        l1_manager_config=l1,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            [_ScriptedProbeConfig(0.01, 10.0) for _ in range(num_adapters)]
        ),
        l2_health_config=L2HealthConfig(
            check_interval_s=0.05,
            probe_timeout_s=0.5,
            failure_threshold=2,
            **health,
        ),
    )


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.02)


@pytest.fixture
def manager():
    _created.clear()
    created = StorageManager(_manager_config(1))
    yield created
    created.close()


def test_status_shows_when_a_backend_goes_down_and_comes_back(manager):
    assert manager.report_status()["l2_degraded"] is False

    _created[0].backend_up = False
    _wait_until(lambda: manager.report_status()["l2_degraded"])

    health = manager.report_status()["l2_health"]
    assert health["enabled"]
    assert health["adapters"][0]["state"] == "unavailable"
    assert health["adapters"][0]["reason"] == "scripted outage"

    _created[0].backend_up = True
    _wait_until(lambda: not manager.report_status()["l2_degraded"])
    assert manager.report_status()["l2_health"]["adapters"][0]["state"] == "healthy"


def test_an_adapter_added_at_runtime_is_monitored_until_it_is_deleted(manager):
    adapter_id = manager.add_l2_adapter(_ScriptedProbeConfig(0.01, 10.0))
    added = _created[-1]
    added.backend_up = False
    _wait_until(lambda: manager.report_status()["l2_degraded"])

    manager.delete_l2_adapter(adapter_id)

    assert not manager.report_status()["l2_degraded"]
    calls_after_delete = added.probe_calls
    time.sleep(0.3)
    assert added.probe_calls == calls_after_delete


def test_close_stops_the_probing():
    _created.clear()
    manager = StorageManager(_manager_config(1))
    _wait_until(lambda: _created[0].probe_calls >= 2)

    manager.close()

    calls_at_close = _created[0].probe_calls
    time.sleep(0.3)
    assert _created[0].probe_calls == calls_at_close


def test_adapters_that_do_not_support_probing_are_left_alone():
    _created.clear()
    config = _manager_config(0)
    config.l2_adapter_config = L2AdaptersConfig(
        [MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)]
    )
    manager = StorageManager(config)
    try:
        time.sleep(0.3)
        status = manager.report_status()
        assert status["l2_health"]["adapters"] == []
        assert status["l2_degraded"] is False
    finally:
        manager.close()
