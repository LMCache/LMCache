# SPDX-License-Identifier: Apache-2.0
"""Tests for PeriodicEventNotifier ownership in the distributed storage path."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed import storage_manager as storage_manager_mod
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.platform import HAS_EVENTFD


def _p2p_config() -> P2PL2AdapterConfig:
    return P2PL2AdapterConfig(
        peer_mq_server_url="tcp://peer:5555",
        peer_transfer_channel_server_url="peer:7600",
    )


def _storage_config(
    adapters: list[L2AdapterConfigBase],
    interval_ms: int,
) -> StorageManagerConfig:
    return StorageManagerConfig(
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
        l2_adapter_config=L2AdaptersConfig(adapters),
        periodic_notifier_interval_ms=interval_ms,
    )


@contextmanager
def _manager_with_mocked_runtime(
    config: StorageManagerConfig,
) -> Iterator[tuple[StorageManager, MagicMock, MagicMock]]:
    l1_manager = MagicMock()
    notifier = MagicMock()
    notifier_class = MagicMock()
    notifier_class.get.side_effect = lambda: (
        notifier if notifier_class.create.called else None
    )

    transfer_context = MagicMock()
    transfer_context.get_transfer_channel_client.return_value = MagicMock()

    with (
        patch.object(storage_manager_mod, "L1Manager", return_value=l1_manager),
        patch.object(storage_manager_mod, "L1EvictionController"),
        patch.object(storage_manager_mod, "L2EvictionController"),
        patch.object(storage_manager_mod, "StoreController"),
        patch.object(storage_manager_mod, "PrefetchController"),
        patch.object(storage_manager_mod, "PeriodicEventNotifier", notifier_class),
        patch.object(storage_manager_mod, "get_event_bus"),
        patch.object(storage_manager_mod, "register_gauge"),
        patch.object(p2p_mod, "PeriodicEventNotifier", notifier_class),
        patch.object(p2p_mod, "MessageQueueClient", return_value=MagicMock()),
        patch.object(
            p2p_mod,
            "get_transfer_channel_context",
            return_value=transfer_context,
        ),
    ):
        manager = StorageManager(config)
        try:
            yield manager, notifier_class, notifier
        finally:
            if notifier_class.shutdown.call_count == 0:
                manager.close()


def test_initial_p2p_adapter_uses_storage_manager_interval() -> None:
    config = _storage_config([_p2p_config()], interval_ms=73)

    with _manager_with_mocked_runtime(config) as (
        _manager,
        notifier_class,
        _notifier,
    ):
        notifier_class.create.assert_called_once_with(
            interval_ms=73,
            use_eventfd=HAS_EVENTFD,
        )


def test_runtime_p2p_adapter_does_not_change_existing_interval() -> None:
    config = _storage_config([], interval_ms=91)

    with _manager_with_mocked_runtime(config) as (
        manager,
        notifier_class,
        _notifier,
    ):
        manager.add_l2_adapter(_p2p_config())

        notifier_class.create.assert_called_once_with(
            interval_ms=91,
            use_eventfd=HAS_EVENTFD,
        )


def test_p2p_fds_are_unregistered_before_notifier_shutdown() -> None:
    config = _storage_config([_p2p_config()], interval_ms=5)
    lifecycle_calls: list[tuple[str, int | None]] = []

    with _manager_with_mocked_runtime(config) as (
        manager,
        notifier_class,
        notifier,
    ):
        notifier.unregister_fd.side_effect = lambda fd: lifecycle_calls.append(
            ("unregister", fd)
        )
        notifier_class.shutdown.side_effect = lambda: lifecycle_calls.append(
            ("shutdown", None)
        )

        manager.close()

        assert lifecycle_calls[-1] == ("shutdown", None)
        assert [name for name, _fd in lifecycle_calls].count("unregister") == 2


@pytest.mark.parametrize("interval_ms", [0, -1])
def test_storage_manager_config_rejects_nonpositive_notifier_interval(
    interval_ms: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="periodic_notifier_interval_ms must be greater than 0",
    ):
        _storage_config([], interval_ms=interval_ms)
