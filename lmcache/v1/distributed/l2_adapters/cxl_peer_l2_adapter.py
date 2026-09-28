# SPDX-License-Identifier: Apache-2.0
"""Borrow read-locked peer CXL bytes through the existing P2P control plane."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import cast
import json
import math
import threading
import time

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.cxl_types import CxlArenaDescriptor
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.memory_allocators.devdax_memory_allocator import CxlPeerMapping
from lmcache.v1.memory_management import (
    CXLMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
)

logger = init_logger(__name__)


def _create_cxl_peer_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    return CxlPeerL2Adapter(cast(CxlPeerL2AdapterConfig, config))


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

    def __init__(
        self,
        peer_mq_server_url: str,
        local_device_path: str,
        local_arena: CxlArenaDescriptor,
        peer_arena: CxlArenaDescriptor,
        lookup_timeout_s: float = 30.0,
    ) -> None:
        if not math.isfinite(lookup_timeout_s) or lookup_timeout_s <= 0:
            raise ValueError("lookup_timeout_s must be positive and finite")
        super().__init__(peer_mq_server_url, "", lookup_timeout_s)
        if local_arena.pool_id != peer_arena.pool_id or local_arena.overlaps(
            peer_arena
        ):
            raise ValueError("CXL peers require the same pool and disjoint slabs")
        self.local_device_path = local_device_path
        self.local_arena = local_arena
        self.peer_arena = peer_arena

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
        for name in ("peer_mq_server_url", "local_device_path"):
            if not isinstance(d.get(name), str) or not d[name]:
                raise ValueError(f"{name} must be a non-empty string")
        timeout = float(d.get("lookup_timeout_s", 30.0))
        if timeout <= 0:
            raise ValueError("lookup_timeout_s must be positive")
        return cls(
            d["peer_mq_server_url"],
            d["local_device_path"],
            CxlArenaDescriptor.from_json(json.dumps(d.get("local_arena"))),
            CxlArenaDescriptor.from_json(json.dumps(d.get("peer_arena"))),
            timeout,
        )

    @classmethod
    def help(cls) -> str:
        """Return the required fields for a shared-pool CXL peer."""
        return (
            "CXL peer: peer_mq_server_url, local_device_path, local_arena, peer_arena"
        )


class CxlPeerL2Adapter(P2PL2Adapter):
    """Reuse P2P lookups and turn their reservations into temporary L1 views.

    Args:
        config: Same-pool peer configuration.

    Borrow lookup state is scoped by task ID. GPU callbacks only enqueue
    releases; a worker sends the existing unlock RPC outside L1 locks.
    """

    def __init__(self, config: CxlPeerL2AdapterConfig) -> None:
        self._mapping = CxlPeerMapping(config.local_device_path, config.peer_arena)
        try:
            super().__init__(config, use_transfer_channel=False)
        except Exception:
            self._mapping.close()
            raise
        self._borrowed_lookups: dict[
            int, dict[ObjectKey, tuple[TransferChannelAddress, float]]
        ] = {}
        self._lookup_starts: dict[L2TaskId, float] = {}
        self._borrow_lock = threading.Lock()
        self._active_borrows = 0
        self._live_views = 0
        self._releases = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cxl-unlock"
        )

    def supports_borrowing(self) -> bool:
        """Return True: this adapter supplies views without destination buffers."""
        return True

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
        started = time.monotonic()
        task_id = super().submit_lookup_and_lock_task(keys, group_layout_descs)
        self._lookup_starts[task_id] = started
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Collect peer hits and retain each lookup's independent reservations.

        Args:
            task_id: Identity returned by submit_lookup_and_lock_task.

        Returns:
            Same-pool hit bitmap, or None while the peer lookup is pending.
        """
        task = self._lookup_tasks.get(task_id)
        if task is None:
            return None
        result = super().query_lookup_and_lock_result(task_id)
        if result is None:
            return None
        started = self._lookup_starts.pop(task_id)
        hits: dict[ObjectKey, tuple[TransferChannelAddress, float]] = {}
        for i, key in enumerate(task.keys):
            if not result.test(i):
                continue
            address = self._remote_addresses.pop(key)
            expires = started + address.cxl_ttl_seconds
            if (
                address.cxl_arena != self._mapping.arena
                or not self._mapping.is_current()
                or time.monotonic() >= expires
            ):
                result.clear(i)
                self._queue_unlock({key: expires})
            else:
                hits[key] = (address, expires)
        if hits:
            self._borrowed_lookups[task_id] = hits
        return result

    def take_borrowed_objects(
        self,
        task_id: L2TaskId,
        keys: list[ObjectKey],
        layouts: dict[int, MemoryLayoutDesc],
    ) -> dict[ObjectKey, MemoryObj]:
        """Move selected reservations into views of peer bytes.

        Args:
            task_id: Completed lookup task.
            keys: Selected keys, at most once each.
            layouts: Expected layout by object-group ID.

        Returns:
            Borrowed objects; invalid ranges/layouts are omitted and remain
            releasable through release_lookup. No payload capacity is used.
        """
        hits = self._borrowed_lookups.get(task_id, {})
        objects: dict[ObjectKey, MemoryObj] = {}
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
            obj = CXLMemoryObj(
                view,
                metadata,
                partial(self._release_view, key, expires),
                expires,
                self._mapping.is_current,
            )
            with self._borrow_lock:
                self._active_borrows += 1
                self._live_views += 1
            objects[key] = obj
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

    def close(self) -> None:
        """Drain queued releases and close the peer mapping and RPC resources.

        Raises:
            RuntimeError: If L1 still owns live shadows; callers must drain them.
        """
        if self._closed:
            return
        with self._borrow_lock:
            if self._live_views:
                raise RuntimeError("Cannot close a CXL peer with live shadows")
        for task_id, hits in list(self._borrowed_lookups.items()):
            self.release_lookup(task_id, list(hits))
        self._releases.shutdown(wait=True)
        self._mapping.close()
        super().close()

    def report_status(self) -> dict:
        """Return peer identity and outstanding borrow count for observability."""
        return {
            **super().report_status(),
            "type": "CxlPeerL2Adapter",
            "is_healthy": self._mapping.is_current(),
            "active_borrows": self.get_active_borrow_count(),
            "pool_id": self._mapping.arena.pool_id,
        }

    def _release_view(self, key: ObjectKey, expires: float) -> None:
        with self._borrow_lock:
            self._live_views -= 1
        # The view's existing active count covers this queued release.
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
            self._req_client.p2p_unlock_objects(keys).result(timeout=3.0)
        except Exception:
            logger.warning(
                "CXL peer unlock failed; reservation follows owner TTL", exc_info=True
            )
        finally:
            with self._borrow_lock:
                self._active_borrows -= 1


register_l2_adapter_type("cxl_peer", CxlPeerL2AdapterConfig)
register_l2_adapter_factory("cxl_peer", _create_cxl_peer_adapter)
