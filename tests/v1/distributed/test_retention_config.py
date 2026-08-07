# SPDX-License-Identifier: Apache-2.0
"""Config validation for retention: watermark bound and adapter count."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    validate_storage_manager_config,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig


def _adapter(evicting: bool = True) -> MockL2AdapterConfig:
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    if evicting:
        config.eviction_config = EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
    return config


def _config(
    fraction: float, adapters: list[L2AdapterConfigBase]
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 * 1024 * 1024,
                use_lazy=False,
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters),
        retention_max_fraction=fraction,
    )


def test_fraction_must_stay_below_watermark():
    with pytest.raises(ValueError, match="trigger_watermark"):
        validate_storage_manager_config(_config(0.8, [_adapter()]))


def test_single_eviction_enabled_adapter_is_accepted():
    validate_storage_manager_config(_config(0.7, [_adapter()]))


def test_multiple_eviction_enabled_adapters_are_rejected():
    with pytest.raises(ValueError, match="one eviction-enabled"):
        validate_storage_manager_config(_config(0.5, [_adapter(), _adapter()]))


def test_disabled_retention_ignores_adapter_count():
    validate_storage_manager_config(_config(0.0, [_adapter(), _adapter()]))


def test_non_evicting_adapters_do_not_count():
    validate_storage_manager_config(
        _config(0.7, [_adapter(), _adapter(evicting=False)])
    )
