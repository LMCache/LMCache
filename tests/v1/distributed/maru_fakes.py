# SPDX-License-Identifier: Apache-2.0

"""Stateful in-memory fakes of the maru runtime for MaruL1Manager tests.

``FakeMaruHandler`` models the MaruServer directory semantics the manager
relies on (per-key pins, dup-skip store, pinned-delete refusal) so contract
tests exercise real behavior instead of tautological mocks. Assertions should
stay on the manager's observable results plus the pin/free bookkeeping here.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
    MaruL1Config,
)
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.maru_l1_manager import MaruL1Manager


class RecordingListener(L1ManagerListener):
    """Records every ``on_l1_keys_*`` firing as ``(event, keys)``.

    Shared by the maru-specific and the cross-backend conformance suites so
    both assert the same listener contract. ``kinds()`` returns the key lists
    of each firing of one event, in order.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, list[ObjectKey]]] = []

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        self.events.append(("reserved_read", list(keys)))

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        self.events.append(("read_finished", list(keys)))

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        self.events.append(("reserved_write", list(keys)))

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        self.events.append(("write_finished", list(keys)))

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        self.events.append(("finish_write_and_reserve_read", list(keys)))

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        self.events.append(("deleted_by_manager", list(keys)))

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.events.append(("accessed", list(keys)))

    def kinds(self, name: str) -> list[list[ObjectKey]]:
        """Return the key list of every recorded firing of event ``name``."""
        return [keys for event, keys in self.events if event == name]


@dataclass
class FakeMemoryInfo:
    """Stand-in for maru's MemoryInfo (only the fields the manager reads)."""

    view: bytes
    region_id: int
    page_index: int


class FakeMaruHandler:
    """Dict-backed MaruServer: directory + per-key pin counts.

    ``fail_*`` knobs make the next matching RPC raise; ``retrieve_none``
    models the real transport-failure mode (batch_retrieve returns all None
    instead of raising).
    """

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
        self.store_map: dict[str, tuple[int, int, int]] = {}  # key -> (rid, pid, size)
        self.pins: dict[str, int] = {}
        self.unpin_log: list[str] = []
        self.fail_store_keys: set[str] = set()  # keys batch_store reports False for
        self.fail_pin = False
        self.fail_retrieve = False
        self.retrieve_none = False
        self.fail_exists = False
        self.fail_store = False
        self.fail_delete = False

    def batch_pin(self, keys: list[str]) -> list[bool]:
        if self.fail_pin:
            raise RuntimeError("rpc fail")
        out = []
        for ks in keys:
            if ks in self.store_map:
                self.pins[ks] = self.pins.get(ks, 0) + 1
                out.append(True)
            else:
                out.append(False)
        return out

    def batch_unpin(self, keys: list[str]) -> list[bool]:
        out = []
        for ks in keys:
            self.unpin_log.append(ks)
            if self.pins.get(ks, 0) > 0:
                self.pins[ks] -= 1
                out.append(True)
            else:
                out.append(False)
        return out

    def batch_retrieve(self, keys: list[str]) -> list[FakeMemoryInfo | None]:
        if self.fail_retrieve:
            raise RuntimeError("rpc fail")
        if self.retrieve_none:
            return [None] * len(keys)
        out: list[FakeMemoryInfo | None] = []
        for ks in keys:
            ent = self.store_map.get(ks)
            out.append(
                FakeMemoryInfo(view=b"x" * ent[2], region_id=ent[0], page_index=ent[1])
                if ent
                else None
            )
        return out

    def batch_store(self, keys: list[str], handles: list) -> list[bool]:
        if self.fail_store:
            raise RuntimeError("rpc fail")
        out = []
        for ks, h in zip(keys, handles, strict=True):
            if ks in self.fail_store_keys:
                out.append(False)
            elif ks in self.store_map:
                out.append(True)  # dup-skip is still a success
            else:
                self.store_map[ks] = (h.region_id, h.page_index, h.size)
                out.append(True)
        return out

    def batch_exists(self, keys: list[str]) -> list[bool]:
        if self.fail_exists:
            raise RuntimeError("rpc fail")
        return [ks in self.store_map for ks in keys]

    def exists(self, key: str) -> bool:
        return key in self.store_map

    def delete(self, key: str) -> bool:
        if self.fail_delete:
            raise RuntimeError("rpc fail")
        if self.pins.get(key, 0) > 0:
            return False  # pinned: refused
        return self.store_map.pop(key, None) is not None

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_stats(self) -> dict:
        return {
            "store_regions": {
                "total_pool_size": 16 * self.chunk_size,
                "total_allocated_pages": len(self.store_map),
            }
        }

    def close(self) -> None:
        pass


class FakeCxlAdapter:
    """Page-pool stand-in for maru_lmcache.CxlMemoryAdapter."""

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
        self.oom = False  # set True to make allocation fail
        self.resolve_none = False  # get_by_location returns None (pool miss)
        self.fail_handle = False  # create_store_handle raises
        self.freed: list[int] = []  # addresses returned via free (abort_alloc)
        self._next_pid = 0
        self._pool: dict[tuple[int, int], MagicMock] = {}

    def _page(self, rid: int, pid: int) -> MagicMock:
        obj = MagicMock(name=f"cxl-page-{rid}-{pid}")
        obj.metadata.address = (rid << 32) | pid
        obj.metadata.phy_size = self.chunk_size
        return obj

    def batched_allocate(self, shapes, dtypes, batch_size, fmt=None, at=None):
        if self.oom:
            return None
        out = []
        for _ in range(batch_size):
            rid, pid = 0, self._next_pid
            self._next_pid += 1
            obj = self._page(rid, pid)
            self._pool[(rid, pid)] = obj
            out.append(obj)
        return out

    def create_store_handle(self, memory_obj) -> SimpleNamespace:
        if self.fail_handle:
            raise RuntimeError("bad handle")
        addr = memory_obj.metadata.address
        return SimpleNamespace(
            region_id=addr >> 32, page_index=addr & 0xFFFFFFFF, size=self.chunk_size
        )

    def get_by_location(self, region_id, page_index, actual_size, single_token_size):
        if self.resolve_none:
            return None
        # A page stored by another instance is materialized on demand
        # (mirrors the eager-map view of a peer region).
        key = (region_id, page_index)
        if key not in self._pool:
            self._pool[key] = self._page(region_id, page_index)
        return self._pool[key]

    def free(self, memory_obj, allocator_type=None) -> None:
        addr = memory_obj.metadata.address
        if addr in self.freed:
            # Mirror the real allocator's double-free detection.
            raise AssertionError(f"double free of page {addr}")
        self.freed.append(addr)

    def close(self) -> None:
        pass


def make_maru_manager(
    chunk_size: int = 64,
) -> tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter]:
    """Build a MaruL1Manager wired to fresh fakes (post-init_layout state)."""
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=0,
            use_lazy=False,
            maru_config=MaruL1Config(
                server_url="maru://localhost:5555",
                pool_size_bytes=1 << 20,
                instance_id="test",
            ),
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    manager = MaruL1Manager(cfg)
    handler = FakeMaruHandler(chunk_size)
    adapter = FakeCxlAdapter(chunk_size)
    allocator = manager._allocator
    allocator._handler = handler
    allocator._cxl_adapter = adapter
    allocator._single_token_size = 16
    return manager, handler, adapter
