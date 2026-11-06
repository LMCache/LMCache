# SPDX-License-Identifier: Apache-2.0
"""A lending peer adapter over an in-memory fake owner (``lending_mock``), and
a REMOTE-wanting config whose adapter cannot lend (``nonlending_mock``)."""

# Standard
from typing import Optional
from unittest.mock import MagicMock, patch
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    LendingPeerL2AdapterBase,
    PeerL2AdapterBase,
    PeerL2AdapterConfigBase,
)
from lmcache.v1.distributed.memory_manager import (
    BorrowedMemoryObj,
    BorrowedTensorMemoryObj,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.memory_management import MemoryObj


class LendingMockL2AdapterConfig(PeerL2AdapterConfigBase):
    """Config of the lending mock; the peer URL is never dialed."""

    def __init__(self, lookup_timeout_s: float = 10.0) -> None:
        super().__init__("tcp://fake-owner:0", lookup_timeout_s)

    @classmethod
    def from_dict(cls, d: dict) -> "LendingMockL2AdapterConfig":
        return cls(lookup_timeout_s=float(d.get("lookup_timeout_s", 10.0)))

    @classmethod
    def help(cls) -> str:
        return "Lending mock adapter config: lookup_timeout_s (float, optional)"

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE),)


class NonLendingMockL2AdapterConfig(MockL2AdapterConfig):
    """Wants a REMOTE L1 like a lender, but its adapter cannot lend."""

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE),)


class LocalFirstLendingMockL2AdapterConfig(LendingMockL2AdapterConfig):
    """A lender's config that forgot to want a REMOTE L1 first."""

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL),)


class _Future:
    def __init__(self, value: object) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> object:
        return self._value


class _FakeOwner:
    """The peer's request server: stores tensors, counts read locks."""

    def __init__(self) -> None:
        self.objects: dict[ObjectKey, MemoryObj] = {}
        self.locks: dict[ObjectKey, int] = {}
        self._lookups: dict[int, list[ObjectKey]] = {}
        self._next_task = 0

    def p2p_lookup_and_lock(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> _Future:
        task_id = self._next_task
        self._next_task += 1
        self._lookups[task_id] = keys
        return _Future(task_id)

    def p2p_query_lookup_results(self, task_id: int) -> _Future:
        addresses = []
        for i, key in enumerate(self._lookups.pop(task_id)):
            obj = self.objects.get(key)
            if obj is None:
                addresses.append(TransferChannelAddress(offset=-1, size=0))
            else:
                self.locks[key] = self.locks.get(key, 0) + 1
                addresses.append(TransferChannelAddress(offset=i, size=obj.get_size()))
        return _Future(addresses)

    def p2p_unlock_objects(self, keys: list[ObjectKey]) -> _Future:
        for key in keys:
            if self.locks.get(key, 0) > 1:
                self.locks[key] -= 1
            else:
                self.locks.pop(key, None)
        return _Future(None)

    def close(self) -> None:
        pass


class LendingMockL2Adapter(LendingPeerL2AdapterBase):
    """Lends views of the fake owner's tensors; a view's lease is its lock."""

    def __init__(self, config: LendingMockL2AdapterConfig) -> None:
        self._owner = _FakeOwner()
        # Reservations a lookup took at the owner and no view consumed yet.
        self._reserved: dict[ObjectKey, int] = {}
        self._live_views = 0
        self._views_lock = threading.Lock()
        self.lookup_count = 0
        # While set, lookup results are withheld so several requests can have
        # lookups of the same key outstanding at once.
        self.hold_lookups = False
        with (
            patch.object(
                p2p_mod.RequestClientFactory, "create", return_value=self._owner
            ),
            patch.object(p2p_mod, "PeriodicEventNotifier") as notifier,
        ):
            notifier.get.return_value = MagicMock()
            super().__init__(config)

    # -- test helpers -------------------------------------------------------

    def debug_has_key(self, key: ObjectKey) -> bool:
        return key in self._owner.objects

    def debug_get_locked_key_count(self) -> int:
        return len(self._owner.locks)

    # -- store: the fake owner keeps the bytes -------------------------------

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        for key, obj in zip(keys, objects, strict=True):
            self._owner.objects[key] = obj
        return super().submit_store_task(keys, objects)

    # -- lookup: the peer base's RPC path, plus an immediate wake-up ----------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        self.lookup_count += 1
        task_id = super().submit_lookup_and_lock_task(keys, group_layout_descs)
        self._lookup_efd.notify()
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        if self.hold_lookups:
            return None
        return super().query_lookup_and_lock_result(task_id)

    def release_lookups(self) -> None:
        """Stop withholding lookup results and wake the prefetch loop."""
        self.hold_lookups = False
        self._lookup_efd.notify()

    def _process_lookup_result(
        self, keys: list[ObjectKey], addresses: list[TransferChannelAddress]
    ) -> Bitmap:
        bitmap = Bitmap(len(keys))
        for i, (key, address) in enumerate(zip(keys, addresses, strict=True)):
            if address.is_valid():
                bitmap.set(i)
                self._reserved[key] = self._reserved.get(key, 0) + 1
        return bitmap

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Return one reservation per key; the controller never names a lent key."""
        for key in keys:
            self._reserved[key] -= 1
        super().submit_unlock(keys)

    # -- lending ------------------------------------------------------------

    def take_borrowed_objects(
        self, keys: list[ObjectKey], layouts: dict[int, MemoryLayoutDesc]
    ) -> dict[ObjectKey, BorrowedMemoryObj]:
        views: dict[ObjectKey, BorrowedMemoryObj] = {}
        for key in keys:
            if not self._reserved.get(key, 0):
                continue
            self._reserved[key] -= 1
            stored = self._owner.objects[key]
            ended: list[bool] = []
            with self._views_lock:
                self._live_views += 1

            def lease_is_valid(ended: list[bool] = ended) -> bool:
                return not ended

            def end_lease(key: ObjectKey = key, ended: list[bool] = ended) -> None:
                if not ended:
                    ended.append(True)
                    with self._views_lock:
                        self._live_views -= 1
                    PeerL2AdapterBase.submit_unlock(self, [key])

            views[key] = BorrowedTensorMemoryObj(
                stored.raw_data,
                stored.meta,
                lease_is_valid=lease_is_valid,
                on_release=end_lease,
            )
        return views

    def get_active_borrow_count(self) -> int:
        with self._views_lock:
            return self._live_views

    # -- the rest of the peer contract --------------------------------------

    def submit_load_task(self, keys: list[ObjectKey], objects: list[MemoryObj]):
        raise NotImplementedError("a lender never loads")

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        raise NotImplementedError("a lender never loads")

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def _register_peer_region(self) -> None:
        pass

    def _unregister_peer_region(self) -> None:
        pass


def _create_lending_mock_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a LendingMockL2Adapter from config."""
    return LendingMockL2Adapter(config)  # type: ignore[arg-type]


def _create_nonlending_mock_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a plain MockL2Adapter from a REMOTE-wanting config."""
    return MockL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_type("lending_mock", LendingMockL2AdapterConfig)
register_l2_adapter_factory("lending_mock", _create_lending_mock_adapter)
register_l2_adapter_type("nonlending_mock", NonLendingMockL2AdapterConfig)
register_l2_adapter_factory("nonlending_mock", _create_nonlending_mock_adapter)
register_l2_adapter_type(
    "localfirst_lending_mock", LocalFirstLendingMockL2AdapterConfig
)
register_l2_adapter_factory("localfirst_lending_mock", _create_lending_mock_adapter)
