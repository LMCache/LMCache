# SPDX-License-Identifier: Apache-2.0

"""Configuration tests for stable multi-disk placement."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig


def _storage_config(
    adapters: list[L2AdapterConfigBase],
    *,
    store_policy: str,
    prefetch_policy: str,
) -> StorageManagerConfig:
    """Build the smallest storage-manager config used by these tests."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20,
                use_lazy=False,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters),
        store_policy=store_policy,
        prefetch_policy=prefetch_policy,
    )


@pytest.mark.parametrize(
    ("store_policy", "prefetch_policy"),
    [("striped", "default"), ("default", "striped")],
)
def test_striped_store_and_prefetch_policies_must_be_paired(
    tmp_path: Path,
    store_policy: str,
    prefetch_policy: str,
) -> None:
    adapter = FSNativeL2AdapterConfig(str(tmp_path / "disk"))

    with pytest.raises(ValueError, match="requires both"):
        _storage_config(
            [adapter],
            store_policy=store_policy,
            prefetch_policy=prefetch_policy,
        )


def test_striped_policy_rejects_non_fs_native_adapter() -> None:
    adapter = MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)

    with pytest.raises(ValueError, match="only fs_native"):
        _storage_config(
            [adapter],
            store_policy="striped",
            prefetch_policy="striped",
        )


def test_striped_policy_accepts_multiple_fs_native_adapters(tmp_path: Path) -> None:
    adapters = [
        FSNativeL2AdapterConfig(str(tmp_path / "disk-a")),
        FSNativeL2AdapterConfig(str(tmp_path / "disk-b")),
    ]

    config = _storage_config(
        adapters,
        store_policy="striped",
        prefetch_policy="striped",
    )

    assert config.l2_adapter_config.adapters == adapters
