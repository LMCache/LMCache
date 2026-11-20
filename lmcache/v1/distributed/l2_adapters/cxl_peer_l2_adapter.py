# SPDX-License-Identifier: Apache-2.0
"""Borrow a peer's objects in place from a shared CXL memory pool.

The owner keeps its read lock while this host uses an object; the bytes are
never copied between hosts. This host maps the owner's slab of the pool, a
shadow L1 indexes the borrowed objects, and each one is retrieved from the
mapping straight into GPU memory like any host-memory object. No local L1
capacity is used.
"""

# Standard
from typing import Any, cast
import json
import math

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.common.transfer_channel.api import (
    CxlArenaDescriptor,
    TransferChannelAddress,
)
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    BorrowingPeerL2AdapterBase,
    PeerL2AdapterBase,
    PeerL2AdapterConfigBase,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import CxlPeerMapping
from lmcache.v1.memory_management import BorrowedMemoryObj, Lease

logger = init_logger(__name__)

# The address of a hit this adapter will not borrow.
_NOT_BORROWABLE = TransferChannelAddress(offset=-1, size=0)


def _create_cxl_peer_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    return CxlPeerL2Adapter(cast(CxlPeerL2AdapterConfig, config))


class CxlPeerL2AdapterConfig(PeerL2AdapterConfigBase):
    """Configure one peer whose slab is visible through a local CXL device.

    Args:
        peer_mq_server_url: Peer lookup/unlock endpoint.
        local_device_path: Local path to the shared pool.
        local_arena: This node's owned slab.
        peer_arena: Peer slab, verified through its mapped identity header.
        lookup_timeout_s: P2P lookup deadline in seconds.
        read_ttl_s: The L1 read-lock TTL, the same on every node; bounds a
            lease. The P2P controller sets it from the local CXL L1.
        read_window_s: Seconds a lease must have left, before the owner's
            lock may lapse, for a GPU copy from the view to start. It bounds
            how long a queued copy may take.

    Raises:
        ValueError: If a deadline or TTL is not positive and finite, if
            ``read_window_s`` is not shorter than ``read_ttl_s``, or if pools
            differ or owned slab ranges overlap.
    """

    def __init__(
        self,
        peer_mq_server_url: str,
        local_device_path: str,
        local_arena: CxlArenaDescriptor,
        peer_arena: CxlArenaDescriptor,
        lookup_timeout_s: float = 30.0,
        read_ttl_s: float = 300.0,
        read_window_s: float = 5.0,
    ) -> None:
        for name, value in (
            ("lookup_timeout_s", lookup_timeout_s),
            ("read_ttl_s", read_ttl_s),
            ("read_window_s", read_window_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if read_window_s >= read_ttl_s:
            raise ValueError("read_window_s must be shorter than read_ttl_s")
        super().__init__(peer_mq_server_url, lookup_timeout_s)
        self.local_device_path = local_device_path
        self.local_arena = local_arena
        self.peer_arena = peer_arena
        self.read_ttl_s = read_ttl_s
        self.read_window_s = read_window_s
        local, peer = local_arena, peer_arena
        if local.pool_id != peer.pool_id or (
            local.offset < peer.offset + peer.alignment + peer.size
            and peer.offset < local.offset + local.alignment + local.size
        ):
            raise ValueError("CXL peers require the same pool and disjoint slabs")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CxlPeerL2AdapterConfig":
        """Decode a runtime adapter configuration.

        Args:
            d: ``peer_mq_server_url`` and ``local_device_path`` strings,
                ``local_arena`` and ``peer_arena`` descriptor objects (the
                fields of :class:`CxlArenaDescriptor`), and optional
                ``lookup_timeout_s``, ``read_ttl_s`` and ``read_window_s``
                numbers.

        Returns:
            Validated CXL peer configuration.

        Raises:
            ValueError: If required fields or arena identities are invalid.
        """
        for name in ("peer_mq_server_url", "local_device_path"):
            if not isinstance(d.get(name), str) or not d[name]:
                raise ValueError(f"{name} must be a non-empty string")
        return cls(
            d["peer_mq_server_url"],
            d["local_device_path"],
            CxlArenaDescriptor.from_json(json.dumps(d.get("local_arena"))),
            CxlArenaDescriptor.from_json(json.dumps(d.get("peer_arena"))),
            float(d.get("lookup_timeout_s", 30.0)),
            float(d.get("read_ttl_s", 300.0)),
            float(d.get("read_window_s", 5.0)),
        )

    @classmethod
    def help(cls) -> str:
        """Return the fields of a shared-pool CXL peer."""
        return (
            "CXL peer adapter config fields:\n"
            "- peer_mq_server_url (str): peer request server URL (required)\n"
            "- local_device_path (str): local path of the shared pool (required)\n"
            "- local_arena (object): this node's slab descriptor (required)\n"
            "- peer_arena (object): the peer's slab descriptor (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 30)\n"
            "- read_ttl_s (float): the L1 read-lock TTL of every node in seconds "
            "(optional, default 300)\n"
            "- read_window_s (float): lease time a GPU copy from the view needs "
            "to start, in seconds (optional, default 5)"
        )

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        """Return ``(CXL, SHADOW)``: borrowed objects go into the shadow L1."""
        return (MemoryTags(MemorySegment.CXL, MemoryResidency.SHADOW),)


class CxlPeerL2Adapter(BorrowingPeerL2AdapterBase):
    """Borrow a peer's objects as views of its slab in a shared CXL pool.

    The lease bookkeeping, the load that binds the shadow L1's placeholders
    and the owner unlocks are :class:`BorrowingPeerL2AdapterBase`'s. This
    class maps the peer's slab and binds each placeholder to a view of it. A
    view is trusted only while the mapped slab header still carries the
    identity the peer advertised, so an owner restart ends every lease.

    Args:
        config: Same-pool peer configuration.
    """

    _config: CxlPeerL2AdapterConfig

    def __init__(self, config: CxlPeerL2AdapterConfig) -> None:
        # A read from the mapping is a host/GPU copy queued at retrieve and
        # finished asynchronously; the window keeps the owner's lock held for
        # at least that long after a copy is allowed to start.
        super().__init__(
            config,
            read_window_s=config.read_window_s,
            owner_read_ttl_s=config.read_ttl_s,
        )

    def _register_peer_region(self) -> None:
        """Map and GPU-register the configured peer slab.

        Raises:
            ValueError: If the mapped slab identity does not match the peer.
            RuntimeError: If the mapping or GPU registration fails.
            OSError: If the shared device cannot be opened or mapped.
        """
        self._mapping = CxlPeerMapping(
            self._config.local_device_path, self._config.peer_arena
        )

    def _unregister_peer_region(self) -> None:
        """Drain the unlocks, then unregister and unmap the peer slab.

        Raises:
            RuntimeError: If a lease is still live.
            BufferError: If a view of the mapping is still referenced, e.g.
                by a reader of an object whose lease expired.
        """
        self._drain_borrows_for_close()
        self._mapping.close()

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(CXL, REMOTE)``: a peer's slab of the shared pool."""
        return MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)

    def report_status(self) -> dict[str, Any]:
        """Return the peer endpoint, slab health and outstanding borrow count."""
        return {
            **super().report_status(),
            "type": "CxlPeerL2Adapter",
            "is_healthy": self._mapping.is_current(),
            "pool_id": self._mapping.arena.pool_id,
        }

    def _process_lookup_result(
        self,
        keys: list[ObjectKey],
        addresses: list[TransferChannelAddress],
        started_at: float,
    ) -> Bitmap:
        """Reserve the hits in the mapped slab; return the others at once.

        A hit outside the mapped incarnation of the peer's slab (another
        arena, or the owner restarted) cannot be borrowed, so its owner lock
        is released instead of being held until a load that never binds it.
        """
        current = self._mapping.is_current()
        borrowable: list[TransferChannelAddress] = []
        refused: list[ObjectKey] = []
        for key, address in zip(keys, addresses, strict=True):
            if not address.is_valid():
                borrowable.append(address)
            elif current and address.cxl_arena == self._mapping.arena:
                borrowable.append(address)
            else:
                borrowable.append(_NOT_BORROWABLE)
                refused.append(key)
        if refused and current:
            # A restarted owner holds none of these locks any more.
            PeerL2AdapterBase.submit_unlock(self, refused)
        return super()._process_lookup_result(keys, borrowable, started_at)

    def _bind_borrowed_object(
        self, obj: BorrowedMemoryObj, address: TransferChannelAddress, lease: Lease
    ) -> bool:
        """Bind the placeholder to a view of the peer's bytes; the lease also
        ends when the owner restarts."""
        try:
            view = self._mapping.view(address.offset, address.size)
        except ValueError:
            logger.warning(
                "Peer object at offset %#x (%d bytes) is outside the mapped slab",
                address.offset,
                address.size,
            )
            return False
        lease_is_valid, on_release = lease
        arena = self._mapping.arena
        obj.bind(
            (lambda: lease_is_valid() and self._mapping.is_current(), on_release),
            local_view=view,
            address=address.offset,
            source=f"cxl://{arena.pool_id}/{arena.offset}/{arena.session_id}",
        )
        return True


register_l2_adapter_type("cxl_peer", CxlPeerL2AdapterConfig)
register_l2_adapter_factory("cxl_peer", _create_cxl_peer_adapter)
