# SPDX-License-Identifier: Apache-2.0
"""Borrow a peer's DRAM objects and read them straight into GPU memory.

GPU-direct borrowing (GDR) keeps the owner's read lock while this host uses an
object, and moves the bytes only at retrieve time, over the transfer channel
from the peer's registered L1 into a GPU landing buffer registered as a device
region. Nothing is staged in this host's DRAM L1 and no L1 capacity is used:
the shadow L1 paired with the DRAM L1 indexes the borrowed objects.
"""

# Standard
from functools import partial
from typing import Any, Callable, cast
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.distributed.l2_adapters.lending_peer_l2_adapter import (
    LendingPeerL2AdapterBase,
)
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelClient,
    TransferChannelContext,
    get_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.gdr_context import DeviceRegionRegistrar
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    MemoryObjMetadata,
    PendingRead,
    RemoteBorrowedMemoryObj,
)

logger = init_logger(__name__)


def _create_gdr_peer_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    return GdrPeerL2Adapter(cast(GdrPeerL2AdapterConfig, config))


class TransferChannelRegionRegistrar(DeviceRegionRegistrar):
    """Register GPU landing buffers with the process's transfer channel.

    Installed by the P2P controller in GDR mode, so GPU cache contexts can
    register their landing rings without knowing the transport.

    Args:
        context: The process-global transfer channel context.
        align_bytes: The L1 alignment the context registered with.
    """

    def __init__(self, context: TransferChannelContext, align_bytes: int) -> None:
        self._context = context
        self._align_bytes = align_bytes

    @property
    def align_bytes(self) -> int:
        """Alignment every registered region and offset must honor."""
        return self._align_bytes

    def register(self, ptr: int, size: int, device_index: int) -> int:
        """Register a device buffer; see ``register_device_region``."""
        return self._context.register_device_region(ptr, size, device_index)

    def unregister(self, region_id: int) -> None:
        """Release a device buffer; see ``unregister_device_region``."""
        self._context.unregister_device_region(region_id)


class _TransferChannelRead(PendingRead):
    """One submitted transfer-channel read with a deadline."""

    def __init__(self, client: TransferChannelClient, task_id: int, deadline: float):
        self._client = client
        self._task_id = task_id
        self._deadline = deadline
        self._succeeded: bool | None = None

    def is_finished(self) -> bool:
        if self._succeeded is not None:
            return True
        result = self._client.query_read_status(self._task_id)
        if result.is_finished():
            self._succeeded = all(result.succeeded_mask)
            return True
        if time.monotonic() > self._deadline:
            logger.warning("GDR read task %d timed out", self._task_id)
            self._succeeded = False
            return True
        return False

    def succeeded(self) -> bool:
        return self._succeeded is True


class _FailedRead(PendingRead):
    """A read refused before submission, e.g. because the lease ended."""

    def is_finished(self) -> bool:
        return True

    def succeeded(self) -> bool:
        return False


class GdrPeerL2AdapterConfig(P2PL2AdapterConfig):
    """Config for a GPU-direct borrowing peer: the P2P fields, lent not copied.

    ``load_timeout_s`` bounds each retrieve-time read into GPU memory.
    """

    @classmethod
    def help(cls) -> str:
        return (
            "GDR peer adapter config fields (same as p2p):\n"
            "- peer_mq_server_url (str): peer request server URL (required)\n"
            "- peer_transfer_channel_server_url (str): the peer's transfer channel "
            "server url (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 10)\n"
            "- load_timeout_s (float): deadline of one retrieve-time read into "
            "GPU memory, in seconds (optional, default 10)"
        )

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        """Bind to the shadow L1 paired with the DRAM L1.

        Returns:
            Only ``(DRAM, REMOTE)``: borrowed objects are admitted into the
            shadow L1, never into an L1 that owns payload.
        """
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE),)


class GdrPeerL2Adapter(LendingPeerL2AdapterBase):
    """Lend a peer's DRAM objects as remote handles read at retrieve time.

    Args:
        config: Peer RPC/transfer endpoints and deadlines.

    The region access is the same transfer-channel client a copying P2P
    adapter uses; only the destination differs (a registered GPU landing
    region, chosen by the reader) and the moment of the read (retrieve, not
    lookup).
    """

    _config: GdrPeerL2AdapterConfig

    def _register_peer_region(self) -> None:
        """Connect to the configured peer and import its registered region.

        Raises:
            RuntimeError: If the transfer context is unavailable or setup fails.
        """
        self._tc_context = get_transfer_channel_context()
        self._tc_client = self._tc_context.get_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def _unregister_peer_region(self) -> None:
        """Drain owner releases, then drop the peer's transfer-channel client.

        Raises:
            RuntimeError: If an L1 still indexes live borrowed objects.
        """
        self._drain_borrows_for_close()
        self._tc_context.remove_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def get_memory_tags(self) -> MemoryTags:
        """Return the memory tags of this adapter: a peer's DRAM L1."""
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def report_status(self) -> dict[str, Any]:
        """Return peer endpoints and the outstanding borrow count."""
        return {
            **super().report_status(),
            "is_healthy": True,
            "type": "GdrPeerL2Adapter",
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
        }

    def _accept_address(self, address: TransferChannelAddress) -> bool:
        """Every valid hit is usable: a peer reporting no TTL is not."""
        return address.read_ttl_seconds > 0

    def _make_borrowed_object(
        self,
        address: TransferChannelAddress,
        metadata: MemoryObjMetadata,
        *,
        lease_is_valid: Callable[[], bool],
        on_release: Callable[[], None],
    ) -> BorrowedMemoryObj | None:
        return RemoteBorrowedMemoryObj(
            metadata,
            lease_is_valid=lease_is_valid,
            on_release=on_release,
            read_into=partial(self._read_into, address, lease_is_valid),
        )

    def _read_into(
        self,
        address: TransferChannelAddress,
        lease_is_valid: Callable[[], bool],
        region_id: int,
        offset: int,
    ) -> PendingRead:
        """Start reading one borrowed object into a registered device region."""
        if not lease_is_valid():
            return _FailedRead()
        task_id = self._tc_client.submit_read(
            [TransferChannelAddress(offset=offset, size=address.size)],
            [address],
            local_region=region_id,
        )
        return _TransferChannelRead(
            self._tc_client, task_id, time.monotonic() + self._config.load_timeout_s
        )


register_l2_adapter_type("p2p_gdr", GdrPeerL2AdapterConfig)
register_l2_adapter_factory("p2p_gdr", _create_gdr_peer_adapter)
