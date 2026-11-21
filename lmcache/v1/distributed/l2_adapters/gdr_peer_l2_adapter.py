# SPDX-License-Identifier: Apache-2.0
"""Borrow a peer's DRAM objects and read them straight into GPU memory.

The owner keeps its read lock while this host uses an object; the bytes move
once, at retrieve, over the transfer channel into the GPU staging buffer. A
shadow L1 indexes the borrowed objects and no local L1 capacity is used.
"""

# Standard
from functools import partial
from typing import Any, Callable, cast
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress
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
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    BorrowingPeerL2AdapterBase,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelClient,
    get_transfer_channel_context,
)
from lmcache.v1.memory_management import BorrowedMemoryObj, Lease, PendingRead

logger = init_logger(__name__)

# Pause between attempts to cancel a read that is past its deadline.
_CANCEL_RETRY_S = 0.01


def _create_gdr_peer_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    return GdrPeerL2Adapter(cast(GdrPeerL2AdapterConfig, config))


class _TransferChannelRead(PendingRead):
    """One submitted transfer-channel read.

    Past its deadline it is cancelled, and it is reported finished only once
    the transport confirms the transfer stopped: a read still running could
    land in a staging slot a later batch reuses.
    """

    def __init__(
        self,
        client: TransferChannelClient,
        task_id: int,
        deadline: float,
        on_finished: Callable[[], None],
    ):
        self._client = client
        self._task_id = task_id
        self._deadline = deadline
        self._on_finished = on_finished
        self._succeeded: bool | None = None

    def is_finished(self) -> bool:
        """Return whether the transport stopped the read; latches the result."""
        if self._succeeded is not None:
            return True
        try:
            result = self._client.query_status(self._task_id)
            if result.is_finished():
                self._succeeded = all(result.succeeded_mask)
            elif time.monotonic() > self._deadline:
                if self._client.cancel(self._task_id):
                    logger.warning("GDR read task %d timed out", self._task_id)
                    self._succeeded = False
                else:
                    # Still running; ask again shortly.
                    self._deadline = time.monotonic() + _CANCEL_RETRY_S
        except KeyError:
            # The client was closed, which released the transfer.
            logger.debug("GDR read task %d is gone", self._task_id)
            self._succeeded = False
        except Exception:
            # A transport error: the read failed once it is confirmed stopped.
            if self._stopped_after_error():
                logger.exception("GDR read task %d failed", self._task_id)
                self._succeeded = False
            else:
                time.sleep(_CANCEL_RETRY_S)
        if self._succeeded is None:
            return False
        self._on_finished()
        return True

    def succeeded(self) -> bool:
        return self._succeeded is True

    def _stopped_after_error(self) -> bool:
        """Cancel the read after a transport error; whether it is stopped.

        An error from the cancel itself counts as stopped: nothing more can
        stop the transfer, and waiting would never end.
        """
        try:
            return self._client.cancel(self._task_id)
        except Exception:
            return True


class _FailedRead(PendingRead):
    """A read refused before submission, e.g. because the lease ended."""

    def is_finished(self) -> bool:
        return True

    def succeeded(self) -> bool:
        return False


class GdrPeerL2AdapterConfig(P2PL2AdapterConfig):
    """Config for a GPU-direct borrowing peer: the P2P fields.

    ``load_timeout_s`` bounds each retrieve-time read into GPU memory; a
    read past it is cancelled.
    """

    read_ttl_s: float = 300.0
    """The L1 read-lock TTL, the same on every node; bounds a lease. The P2P
    controller sets it from the local L1."""

    @classmethod
    def help(cls) -> str:
        return (
            "GDR peer adapter config fields (same as p2p):\n"
            "- peer_mq_server_url (str): peer request server URL (required)\n"
            "- peer_transfer_channel_server_url (str): the peer's transfer channel "
            "server url (required)\n"
            "- lookup_timeout_s (float): lookup result deadline in seconds "
            "(optional, default 10)\n"
            "- load_timeout_s (float): seconds before a retrieve-time read into "
            "GPU memory is cancelled and fails (optional, default 10)"
        )

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        """Return ``(DRAM, SHADOW)``: borrowed objects go into the shadow L1."""
        return (MemoryTags(MemorySegment.DRAM, MemoryResidency.SHADOW),)


class GdrPeerL2Adapter(BorrowingPeerL2AdapterBase):
    """Borrow a peer's DRAM objects and read them into GPU memory at retrieve.

    Args:
        config: Peer RPC/transfer endpoints and deadlines.
    """

    _config: GdrPeerL2AdapterConfig

    def __init__(self, config: GdrPeerL2AdapterConfig) -> None:
        # Reads the transport has not stopped yet; guarded by the lock.
        self._reads_in_flight = 0
        self._reads_lock = threading.Lock()
        super().__init__(
            config,
            read_window_s=config.load_timeout_s,
            owner_read_ttl_s=config.read_ttl_s,
        )

    def _register_peer_region(self) -> None:
        """Connect to the peer's transfer channel.

        Raises:
            RuntimeError: If the transfer context is unavailable or setup fails.
        """
        self._tc_context = get_transfer_channel_context()
        self._tc_client = self._tc_context.get_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def _unregister_peer_region(self) -> None:
        """Drain the unlocks, then drop the peer's transfer-channel client.

        Raises:
            RuntimeError: If a lease is still live or a read is in flight;
                dropping the client under a read would not stop it reliably.
        """
        with self._reads_lock:
            if self._reads_in_flight:
                raise RuntimeError("Cannot close GdrPeerL2Adapter with reads in flight")
        self._drain_borrows_for_close()
        self._tc_context.remove_transfer_channel_client(
            self._config.peer_transfer_channel_server_url
        )

    def get_memory_tags(self) -> MemoryTags:
        """Return ``(DRAM, REMOTE)``: a peer's DRAM L1."""
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE)

    def report_status(self) -> dict[str, Any]:
        """Return peer endpoints and the outstanding borrow count."""
        return {
            **super().report_status(),
            "type": "GdrPeerL2Adapter",
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
        }

    def _bind_borrowed_object(
        self, obj: BorrowedMemoryObj, address: TransferChannelAddress, lease: Lease
    ) -> bool:
        """Bind the placeholder to a read of the peer's bytes; the read checks
        the object's current lease."""
        obj.bind(
            lease,
            read_into=partial(self._read_into, address, obj.is_valid),
            address=address.offset,
            source=self._config.peer_transfer_channel_server_url,
        )
        return True

    def _read_into(
        self,
        address: TransferChannelAddress,
        lease_is_valid: Callable[[], bool],
        region_id: int,
        offset: int,
    ) -> PendingRead:
        """Start reading one borrowed object into a registered region."""
        if not lease_is_valid():
            return _FailedRead()
        task_id = self._tc_client.submit_read(
            [TransferChannelAddress(offset=offset, size=address.size)],
            [address],
            local_region=region_id,
        )
        with self._reads_lock:
            self._reads_in_flight += 1
        return _TransferChannelRead(
            self._tc_client,
            task_id,
            time.monotonic() + self._config.load_timeout_s,
            self._read_finished,
        )

    def _read_finished(self) -> None:
        with self._reads_lock:
            self._reads_in_flight -= 1


register_l2_adapter_type("p2p_gdr", GdrPeerL2AdapterConfig)
register_l2_adapter_factory("p2p_gdr", _create_gdr_peer_adapter)
