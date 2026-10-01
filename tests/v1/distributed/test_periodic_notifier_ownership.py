# SPDX-License-Identifier: Apache-2.0
"""Check notifier startup and shutdown around the P2P adapter lifecycle."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.v1.distributed import storage_manager as storage_manager_mod
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager


def test_periodic_notifier_lifecycle() -> None:
    """Create the notifier before P2P registration and stop it after cleanup."""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4096,
                use_lazy=False,
                init_size_in_bytes=4096,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            [P2PL2AdapterConfig("tcp://peer:5555", "peer:7600")]
        ),
    )
    notifier = MagicMock()
    notifier.get.side_effect = lambda: notifier if notifier.create.called else None

    with (
        patch.object(storage_manager_mod, "L1Manager"),
        patch.object(storage_manager_mod, "L1EvictionController"),
        patch.object(storage_manager_mod, "L2EvictionController"),
        patch.object(storage_manager_mod, "StoreController"),
        patch.object(storage_manager_mod, "PrefetchController"),
        patch.object(storage_manager_mod, "PeriodicEventNotifier", notifier),
        patch.object(storage_manager_mod, "get_event_bus"),
        patch.object(storage_manager_mod, "register_gauge"),
        patch.object(p2p_mod, "PeriodicEventNotifier", notifier),
        patch.object(p2p_mod.RequestClientFactory, "create"),
        patch.object(p2p_mod, "get_transfer_channel_context"),
    ):
        manager = StorageManager(config)
        try:
            notifier.create.assert_called_once()
            notifier.shutdown.assert_not_called()
        finally:
            manager.close()

    assert [call[0] for call in notifier.mock_calls if call[0] != "get"] == [
        "create",
        "register_fd",
        "register_fd",
        "unregister_fd",
        "unregister_fd",
        "shutdown",
    ]
