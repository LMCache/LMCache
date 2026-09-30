# SPDX-License-Identifier: Apache-2.0
"""Borrow read-locked peer CXL bytes through the existing P2P control plane."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import cast
import json
import math
import threading
import time
import weakref

# Third Party
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import CxlArenaDescriptor, L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import (
    BorrowedObject,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.transfer_channel.api import MemoryRegionAddress
from lmcache.v1.memory_allocators.devdax_memory_allocator import CxlPeerMapping
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

logger = init_logger(__name__)


def _create_cxl_peer_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    return CxlPeerL2Adapter(cast(CxlPeerL2AdapterConfig, config))


@dataclass
class CxlPeerL2AdapterConfig(P2PL2AdapterConfig):
    """Configure one peer whose slab is visible through a local CXL device.

    Args:
        peer_mq_server_url: Peer lookup/unlock endpoint.
        local_device_path: Local path to the shared pool.
        local_arena: This node's owned slab.
        peer_arena: Peer slab, verified through its mapped identity header.
        lookup_timeout_s: Existing P2P lookup deadline in seconds.

    Raises:
        ValueError: If pools differ or owned slab ranges overlap.
    """

    peer_mq_server_url: str
    local_device_path: str
    local_arena: CxlArenaDescriptor
    peer_arena: CxlArenaDescriptor
    lookup_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lookup_timeout_s) or self.lookup_timeout_s <= 0:
            raise ValueError("lookup_timeout_s must be positive and finite")
        super().__init__(self.peer_mq_server_url, "", self.lookup_timeout_s)
        local, peer = self.local_arena, self.peer_arena
        if local.pool_id != peer.pool_id or (
            local.offset < peer.offset + peer.alignment + peer.size
            and peer.offset < local.offset + local.alignment + local.size
        ):
            raise ValueError("CXL peers require the same pool and disjoint slabs")

    @classmethod
    def from_dict(cls, d: dict) -> "CxlPeerL2AdapterConfig":
        """Decode a runtime adapter configuration.

        Args:
            d: Endpoint/path plus local_arena and peer_arena descriptor objects.

        Returns:
            Validated CXL peer configuration.

        Raises:
            ValueError: If required fields or arena identities are invalid.
        """
        d = {**d, "peer_rpc_url": d.get("peer_rpc_url", d.get("peer_mq_server_url"))}
        for name in ("peer_rpc_url", "local_device_path"):
            if not isinstance(d.get(name), str) or not d[name]:
                raise ValueError(f"{name} must be a non-empty string")
        timeout = float(d.get("lookup_timeout_s", 30.0))
        return cls(
            d["peer_rpc_url"],
            d["local_device_path"],
            CxlArenaDescriptor.from_json(json.dumps(d.get("local_arena"))),
            CxlArenaDescriptor.from_json(json.dumps(d.get("peer_arena"))),
            timeout,
        )

    @classmethod
    def help(cls) -> str:
        """Return the required fields for a shared-pool CXL peer."""
        return (
            "CXL peer: peer_rpc_url (or peer_mq_server_url), local_device_path, "
            "local_arena, peer_arena"
        )


class CxlPeerL2Adapter(P2PL2Adapter):
    """Reuse P2P lookups and turn their reservations into temporary L1 views.

    Args:
        config: Same-pool peer configuration.

    Borrow lookup state is scoped by task ID. GPU callbacks only enqueue
    releases; a worker sends the existing unlock RPC outside L1 locks.
    """

    def __init__(self, config: CxlPeerL2AdapterConfig) -> None:
        super().__init__(config)
        self._borrowed_lookups: dict[
            int, dict[ObjectKey, tuple[MemoryRegionAddress, float]]
        ] = {}
        self._borrow_lock = threading.Lock()
        self._active_borrows = 0
        self._live_views = 0
        self._releases = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cxl-unlock"
        )

    def register_peer_region(self) -> None:
        """Map and GPU-register the configured peer slab during construction.

        Uses the same lifecycle as transfer-channel registration, with the
        existing CXL mapping supplying region access instead of a copy client.

        Raises:
            ValueError: If the mapped slab identity does not match the peer.
            RuntimeError: If the mapping or GPU registration fails.
            OSError: If the shared device cannot be opened or mapped.
        """
        config = cast(CxlPeerL2AdapterConfig, self._config)
        self._mapping = CxlPeerMapping(config.local_device_path, config.peer_arena)

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        """Submit an owner lookup and start a conservative local TTL clock.

        Args:
            keys: Object keys to look up, without duplicates.
            group_layout_descs: Expected layouts indexed by object-group ID.

        Returns:
            The local lookup task ID.

        Raises:
            ValueError: If keys contain duplicates, which cannot identify
                independent reservations within one result dictionary.
        """
        if len(set(keys)) != len(keys):
            raise ValueError("CXL lookup keys must be unique")
        return super().submit_lookup_and_lock_task(keys, group_layout_descs)

    def take_borrowed_objects(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        layouts: dict[int, MemoryLayoutDesc],
    ) -> dict[ObjectKey, BorrowedObject]:
        """Move selected reservations into views of peer bytes.

        Args:
            task_id: Completed lookup task.
            keys: Selected keys, at most once each.
            layouts: Expected layout by object-group ID.

        Returns:
            Per-key (tensor view, validity predicate, release callback).
            Invalid ranges/layouts are omitted and remain
            releasable through release_lookup. No payload capacity is used.
        """
        hits = self._borrowed_lookups.get(task_id, {})
        objects: dict[ObjectKey, BorrowedObject] = {}
        for key in keys:
            hit = hits.get(key)
            if hit is None:
                continue
            address, expires = hit
            if time.monotonic() >= expires:
                continue
            layout = layouts[key.object_group_id]
            size = sum(
                s.numel() * d.itemsize
                for s, d in zip(layout.shapes, layout.dtypes, strict=True)
            )
            if not layout.shapes or size != address.size:
                continue
            try:
                view = self._mapping.view(address.offset, address.size)
            except ValueError:
                continue
            metadata = MemoryObjMetadata(
                shape=layout.shapes[0],
                dtype=layout.dtypes[0],
                address=address.offset,
                phy_size=size,
                ref_count=1,
                fmt=MemoryFormat.KV_2LTD,
                shapes=layout.shapes,
                dtypes=layout.dtypes,
            )
            obj = TensorMemoryObj(view, metadata, parent_allocator=None)
            with self._borrow_lock:
                self._active_borrows += 1
                self._live_views += 1
            release = weakref.finalize(
                obj, self._release_view, weakref.ref(obj), key, expires
            )
            objects[key] = (
                obj,
                partial(self._view_is_valid, release, expires),
                release,
            )
            del hits[key]
        if not hits:
            self._borrowed_lookups.pop(task_id, None)
        return objects

    def release_lookup(self, task_id: L2TaskId, keys: list[ObjectKey]) -> None:
        """Return unused reservations from one lookup; repeated calls are inert.

        Args:
            task_id: Completed local lookup identity.
            keys: Keys to release; adopted shadows own their own reservations.
        """
        hits = self._borrowed_lookups.get(task_id, {})
        release = {}
        for key in keys:
            hit = hits.pop(key, None)
            if hit is not None:
                release[key] = hit[1]
        if not hits:
            self._borrowed_lookups.pop(task_id, None)
        if release:
            self._queue_unlock(release)

    def get_active_borrow_count(self) -> int:
        """Return views and pending unlock operations that must drain."""
        with self._borrow_lock:
            return self._active_borrows

    def unregister_peer_region(self) -> None:
        """Drain owner releases, then unregister and unmap the peer slab.

        Called by the shared ``close`` lifecycle before RPC teardown.

        Raises:
            RuntimeError: If L1 still owns live shadows; callers must drain them.
            BufferError: If an exported tensor still references the mapping.
        """
        with self._borrow_lock:
            if self._live_views:
                raise RuntimeError("Cannot close a CXL peer with live shadows")
        for task_id, hits in list(self._borrowed_lookups.items()):
            self.release_lookup(task_id, list(hits))
        self._releases.shutdown(wait=True)
        self._mapping.close()

    def report_status(self) -> dict:
        """Return peer identity and outstanding borrow count for observability."""
        return {
            **super().report_status(),
            "type": "CxlPeerL2Adapter",
            "is_healthy": self._mapping.is_current(),
            "active_borrows": self.get_active_borrow_count(),
            "pool_id": self._mapping.arena.pool_id,
        }

    def _process_lookup_result(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        addresses: list[MemoryRegionAddress],
        started_at: float,
    ) -> Bitmap:
        """Retain task-scoped borrows without populating the copy-address cache."""
        result = Bitmap(len(keys))
        hits: dict[ObjectKey, tuple[MemoryRegionAddress, float]] = {}
        for i, (key, address) in enumerate(zip(keys, addresses, strict=True)):
            if not address.is_valid():
                continue
            expires = started_at + address.read_ttl_seconds
            if (
                address.cxl_arena != self._mapping.arena
                or not self._mapping.is_current()
                or time.monotonic() >= expires
            ):
                self._queue_unlock({key: expires})
            else:
                result.set(i)
                hits[key] = (address, expires)
        if hits:
            self._borrowed_lookups[task_id] = hits
        return result

    def _view_is_valid(self, release: weakref.finalize, expires: float) -> bool:
        """An adopted reservation ends at release, owner expiry, or owner restart."""
        return (
            release.alive and time.monotonic() < expires and self._mapping.is_current()
        )

    def _release_view(
        self,
        reference: weakref.ReferenceType[TensorMemoryObj],
        key: ObjectKey,
        expires: float,
    ) -> None:
        obj = reference()
        if obj is not None:
            obj.invalidate()
            obj.raw_data = torch.empty(0, dtype=torch.uint8)
        with self._borrow_lock:
            self._live_views -= 1
            # Queue before close can observe zero live views and stop the worker.
            self._releases.submit(self._send_unlock, {key: expires})

    def _queue_unlock(self, keys: dict[ObjectKey, float]) -> None:
        with self._borrow_lock:
            self._active_borrows += 1
            self._releases.submit(self._send_unlock, keys)

    def _send_unlock(self, reservations: dict[ObjectKey, float]) -> None:
        try:
            # Expired reservations belong to the owner's TTL cleanup. A late
            # key-only unlock must not consume a subsequent borrow's count.
            if not self._mapping.is_current():
                return
            now = time.monotonic()
            keys = [key for key, expires in reservations.items() if now < expires]
            if not keys:
                return
            super().submit_unlock(keys, wait=True)
        except Exception:
            logger.warning(
                "CXL peer unlock failed; reservation follows owner TTL", exc_info=True
            )
        finally:
            with self._borrow_lock:
                self._active_borrows -= 1


register_l2_adapter_type("cxl_peer", CxlPeerL2AdapterConfig)
register_l2_adapter_factory("cxl_peer", _create_cxl_peer_adapter)
