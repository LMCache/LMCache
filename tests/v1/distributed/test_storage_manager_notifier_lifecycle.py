# SPDX-License-Identifier: Apache-2.0
"""Teardown order of the storage manager, its peer adapters and the notifier.

Peer L2 adapters register their event fds with the process-wide
``PeriodicEventNotifier`` and unregister them when they close. Shutting the
notifier down frees its native instance, so an adapter that still holds it
must close first. The native notifier has no use-after-shutdown check (the
call is undefined behavior and can hang), so these tests swap in a
stand-in that fails loudly instead.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
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
from lmcache.v1.distributed.p2p import P2PClient
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


class _NotifierUsedAfterShutdown(RuntimeError):
    """A notifier instance was used after ``shutdown`` freed it."""


class _StrictNotifier:
    """Stand-in for the native ``PeriodicEventNotifier`` singleton.

    Mirrors its static ``create``/``get``/``shutdown`` API. Unlike the native
    class, an instance that ``shutdown`` released raises on any later use
    instead of touching freed memory.
    """

    _instance: _StrictNotifier | None = None
    shutdown_count = 0
    # fds still registered when the last shutdown ran.
    fds_at_shutdown: set[int] = set()

    def __init__(self) -> None:
        self._alive = True
        self.fds: set[int] = set()

    @classmethod
    def reset(cls) -> None:
        """Forget any instance and shutdown history from a previous test."""
        cls._instance = None
        cls.shutdown_count = 0
        cls.fds_at_shutdown = set()

    @classmethod
    def create(cls, interval_ms: int, use_eventfd: bool) -> None:
        """Create the singleton; a second call is a no-op."""
        if cls._instance is None:
            cls._instance = cls()

    @classmethod
    def get(cls) -> _StrictNotifier | None:
        """Return the singleton, or None if it does not exist."""
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Release the singleton; idempotent."""
        instance = cls._instance
        if instance is None:
            return
        cls.shutdown_count += 1
        cls.fds_at_shutdown = set(instance.fds)
        instance._alive = False
        cls._instance = None

    def register_fd(self, fd: int) -> None:
        """Start signaling ``fd``."""
        self._check_alive()
        self.fds.add(fd)

    def unregister_fd(self, fd: int) -> None:
        """Stop signaling ``fd``."""
        self._check_alive()
        self.fds.discard(fd)

    def _check_alive(self) -> None:
        if not self._alive:
            raise _NotifierUsedAfterShutdown("notifier used after shutdown")


@pytest.fixture
def strict_notifier() -> Iterator[type[_StrictNotifier]]:
    """Use the strict stand-in as the notifier and fake the peer's endpoints."""
    _StrictNotifier.reset()
    with (
        patch.object(storage_manager_mod, "PeriodicEventNotifier", _StrictNotifier),
        patch.object(p2p_mod, "PeriodicEventNotifier", _StrictNotifier),
        patch.object(p2p_mod, "P2PClient", return_value=MagicMock(spec=P2PClient)),
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=MagicMock()),
    ):
        yield _StrictNotifier
    _StrictNotifier.reset()


def _peer_config() -> P2PL2AdapterConfig:
    return P2PL2AdapterConfig("grpc://peer:9555", "peer:7600")


def _storage_manager_config(
    adapters: list[L2AdapterConfigBase],
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=32 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters),
    )


def _spy_on_l1_close(sm: StorageManager, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each L1 manager close in the returned list."""
    closed: list[str] = []
    for manager in sm.l1_managers():
        original = manager.close

        def recording_close(original: Callable[[], None] = original) -> None:
            closed.append("l1")
            original()

        monkeypatch.setattr(manager, "close", recording_close)
    return closed


@pytest.mark.parametrize("added_at", ["construction", "runtime"])
def test_close_shuts_notifier_down_after_peer_adapters(
    strict_notifier: type[_StrictNotifier],
    monkeypatch: pytest.MonkeyPatch,
    added_at: str,
) -> None:
    """Closing a storage manager that still holds a peer adapter closes the
    adapter (unregistering its fds) before the notifier shuts down, so the
    adapter closes cleanly and the L1 managers are released."""
    if added_at == "construction":
        sm = StorageManager(_storage_manager_config([_peer_config()]))
    else:
        sm = StorageManager(_storage_manager_config([]))
        sm.add_l2_adapter(_peer_config())
    notifier = strict_notifier.get()
    assert notifier is not None
    assert notifier.fds, "the peer adapter should have registered its event fds"
    l1_closed = _spy_on_l1_close(sm, monkeypatch)

    sm.close()

    assert strict_notifier.shutdown_count == 1
    assert strict_notifier.fds_at_shutdown == set()
    # L1 stays open only if an adapter failed to close.
    assert l1_closed == ["l1"]


def test_construction_failure_closes_peer_adapters_before_the_notifier(
    strict_notifier: type[_StrictNotifier],
) -> None:
    """If construction fails after a peer adapter was built, the unwind closes
    the adapter while the notifier is still running."""
    with (
        patch.object(
            storage_manager_mod,
            "QuotaManager",
            side_effect=RuntimeError("late construction failure"),
        ),
        pytest.raises(RuntimeError, match="late construction failure"),
    ):
        StorageManager(_storage_manager_config([_peer_config()]))

    assert strict_notifier.shutdown_count == 1
    assert strict_notifier.fds_at_shutdown == set()


def test_peer_adapter_closes_after_notifier_shutdown(
    strict_notifier: type[_StrictNotifier],
) -> None:
    """A peer adapter whose notifier was already shut down still closes: it
    does not use the released notifier instance."""
    adapter = p2p_mod.P2PL2Adapter(_peer_config())
    strict_notifier.shutdown()

    adapter.close()
