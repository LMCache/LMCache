# SPDX-License-Identifier: Apache-2.0
"""Integration tests for StorageManager runtime policy updates."""

# Standard
from typing import cast

# Third Party
import pytest

pytest.importorskip("sortedcontainers")

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.runtime_policy import (
    RuntimeL2EvictionUpdate,
    RuntimePolicyError,
    RuntimePolicyTunables,
    RuntimePolicyUpdate,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

try:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    pytest.skip(
        "Skipping because StorageManager cannot be imported", allow_module_level=True
    )

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


@pytest.fixture
def basic_l1_config() -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def storage_manager(basic_l1_config: L1ManagerConfig):
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=basic_l1_config,
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
                trigger_watermark=0.8,
                eviction_ratio=0.2,
            ),
        )
    )
    yield manager
    manager.close()


@pytest.fixture
def storage_manager_with_l2(basic_l1_config: L1ManagerConfig):
    adapter_config = MockL2AdapterConfig(
        max_size_gb=0.01,
        mock_bandwidth_gb=10.0,
    )
    adapter_config.eviction_config = EvictionConfig(
        eviction_policy="LRU",
        trigger_watermark=0.7,
        eviction_ratio=0.3,
    )
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=basic_l1_config,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[adapter_config]),
        )
    )
    yield manager
    manager.close()


def _capability(
    policy: dict[str, object],
    name: str,
) -> dict[str, object]:
    """Return one capability entry from the JSON-shaped status mapping."""
    capabilities = cast(dict[str, object], policy["capabilities"])
    return cast(dict[str, object], capabilities[name])


def _tunable(
    capability: dict[str, object],
    name: str,
) -> dict[str, object]:
    """Return one eviction tunable from a capability entry."""
    tunables = cast(dict[str, object], capability["runtime_tunables"])
    return cast(dict[str, object], tunables[name])


def test_runtime_policy_reports_capabilities(storage_manager: StorageManager) -> None:
    policy = storage_manager.get_runtime_policy()

    assert policy["version"] == 0
    assert _capability(policy, "store_policy")["current"] == "default"
    assert _capability(policy, "prefetch_policy")["current"] == "default"
    assert _capability(policy, "l1_eviction")["policy"] == "LRU"
    assert (
        _tunable(_capability(policy, "l1_eviction"), "eviction_ratio")["current"] == 0.2
    )


def test_runtime_policy_update_changes_selectors_and_l1_tunables(
    storage_manager: StorageManager,
) -> None:
    result = storage_manager.update_runtime_policy(
        RuntimePolicyUpdate(
            expected_version=0,
            store_policy="skip_l1",
            prefetch_policy="retain",
            l1_eviction=RuntimePolicyTunables(
                trigger_watermark=0.9,
                eviction_ratio=0.1,
            ),
        )
    )

    assert result.status == "updated"
    assert result.version == 1
    assert result.applied == (
        "store_policy",
        "prefetch_policy",
        "l1_eviction.trigger_watermark",
        "l1_eviction.eviction_ratio",
    )

    policy = storage_manager.get_runtime_policy()
    assert policy["version"] == 1
    assert _capability(policy, "store_policy")["current"] == "skip_l1"
    assert _capability(policy, "prefetch_policy")["current"] == "retain"
    assert (
        _tunable(_capability(policy, "l1_eviction"), "trigger_watermark")["current"]
        == 0.9
    )
    assert storage_manager.report_status()["runtime_policy_version"] == 1


def test_runtime_policy_rejects_stale_version_without_mutation(
    storage_manager: StorageManager,
) -> None:
    storage_manager.update_runtime_policy(RuntimePolicyUpdate(store_policy="skip_l1"))

    with pytest.raises(RuntimePolicyError, match="version changed") as exc_info:
        storage_manager.update_runtime_policy(
            RuntimePolicyUpdate(
                expected_version=0,
                prefetch_policy="retain",
            )
        )

    assert exc_info.value.code == "version_conflict"
    assert exc_info.value.status_code == 409
    assert storage_manager.get_runtime_policy()["version"] == 1
    assert (
        _capability(storage_manager.get_runtime_policy(), "prefetch_policy")["current"]
        == "default"
    )


def test_runtime_policy_rejects_eviction_policy_class_change(
    storage_manager: StorageManager,
) -> None:
    with pytest.raises(RuntimePolicyError) as exc_info:
        storage_manager.update_runtime_policy(
            RuntimePolicyUpdate(
                l1_eviction=RuntimePolicyTunables(policy="noop"),
            )
        )

    assert exc_info.value.code == "state_migration_required"
    assert storage_manager.get_runtime_policy()["version"] == 0


def test_runtime_policy_updates_configured_l2_tunables(
    storage_manager_with_l2: StorageManager,
) -> None:
    result = storage_manager_with_l2.update_runtime_policy(
        RuntimePolicyUpdate(
            l2_eviction=(
                RuntimeL2EvictionUpdate(
                    adapter_id=0,
                    tunables=RuntimePolicyTunables(
                        trigger_watermark=0.95,
                        eviction_ratio=0.05,
                    ),
                ),
            ),
        )
    )

    assert result.version == 1
    assert result.applied == (
        "l2_eviction[0].trigger_watermark",
        "l2_eviction[0].eviction_ratio",
    )
    capabilities = cast(
        dict[str, object], storage_manager_with_l2.get_runtime_policy()["capabilities"]
    )
    l2 = cast(list[dict[str, object]], capabilities["l2_eviction"])[0]
    assert l2["adapter_id"] == 0
    assert _tunable(l2, "trigger_watermark")["current"] == 0.95
    assert _tunable(l2, "eviction_ratio")["current"] == 0.05


def test_runtime_policy_rejects_startup_only_field(
    storage_manager: StorageManager,
) -> None:
    with pytest.raises(RuntimePolicyError) as exc_info:
        storage_manager.update_runtime_policy(
            RuntimePolicyUpdate(restart_required_fields=("chunk_size",))
        )

    assert exc_info.value.code == "restart_required"
    assert exc_info.value.field == "chunk_size"
