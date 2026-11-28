# SPDX-License-Identifier: Apache-2.0
"""Borrow a peer's DRAM objects and read them straight into GPU memory.

The owner keeps its read lock while this host uses an object; the bytes move
once, at retrieve, over the transfer channel into the GPU staging buffer. A
shadow L1 indexes the borrowed objects and no local L1 capacity is used.
"""

# Standard
from typing import Any, Callable, Sequence, cast
import threading
import time

# First Party
from lmcache.v1.common.logging import init_logger
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
        self._timed_out = False

    def is_finished(self) -> bool:
        """Return whether the transport stopped the read; latches the result."""
        if self._succeeded is not None:
            return True
        try:
            result = self._client.query_status(self._task_id)
            if result.is_finished():
                # Finishing after the deadline is too late: the owner's lock
                # may lapse before the bytes are used.
                self._succeeded = not self._timed_out and all(result.succeeded_mask)
            elif time.monotonic() > self._deadline:
                self._timed_out = True
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
    read past it is cancelled. ``read_ttl_s`` bounds a lease.
    """

    @classmethod
    def check_timeouts(cls, read_ttl_s: float, load_timeout_s: float) -> None:
        """A lease lasts the read TTL and must leave time for one read.

        Raises:
            ValueError: If the read TTL is not more than 1 s above the load
                timeout: no chunk would ever be borrowed.
        """
        if read_ttl_s <= load_timeout_s + 1:
            raise ValueError(
                f"GPU-direct borrowing needs --l1-read-ttl-seconds ({read_ttl_s}) "
                f"more than 1 s above --p2p-load-timeout ({load_timeout_s}), or "
                "no chunk is ever borrowed"
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
        """Return the borrowing status plus the adapter type and the peer's
        transfer url."""
        return {
            **super().report_status(),
            "type": "GdrPeerL2Adapter",
            "peer_transfer_channel_server_url": (
                self._config.peer_transfer_channel_server_url
            ),
        }

    def read_many(
        self,
        region_id: int,
        targets: Sequence[tuple[BorrowedMemoryObj, int]],
    ) -> PendingRead:
        """Read borrowed objects of this peer into a registered region with
        one transfer-channel submit; see :meth:`BorrowedMemoryObj.read_many`.

        Args:
            region_id: Local region registered with the transfer channel.
            targets: ``(object, offset)`` pairs of objects this adapter bound.

        Returns:
            The read, cancelled past ``load_timeout_s``; a failed read
            without a submit if any object's lease has ended.
        """
        # Counted before the lease check, so close() cannot drop the client
        # between that check and the submit.
        with self._reads_lock:
            self._reads_in_flight += 1
        task_id: int | None = None
        try:
            if all(obj.is_valid() for obj, _ in targets):
                task_id = self._tc_client.submit_read(
                    [
                        TransferChannelAddress(offset=offset, size=obj.get_size())
                        for obj, offset in targets
                    ],
                    [
                        TransferChannelAddress(
                            offset=obj.meta.address, size=obj.get_size()
                        )
                        for obj, _ in targets
                    ],
                    local_region=region_id,
                )
        finally:
            if task_id is None:
                self._read_finished()
        if task_id is None:
            return _FailedRead()
        return _TransferChannelRead(
            self._tc_client,
            task_id,
            time.monotonic() + self._config.load_timeout_s,
            self._read_finished,
        )

    def _bind_borrowed_object(
        self, obj: BorrowedMemoryObj, address: TransferChannelAddress, lease: Lease
    ) -> None:
        """Bind the placeholder to this adapter's reads of the peer's bytes;
        a read checks the object's current lease."""
        obj.bind(
            lease,
            read_many=self.read_many,
            address=address.offset,
            source=self._config.peer_transfer_channel_server_url,
        )

    def _read_finished(self) -> None:
        with self._reads_lock:
            self._reads_in_flight -= 1


register_l2_adapter_type("p2p_gdr", GdrPeerL2AdapterConfig)
register_l2_adapter_factory("p2p_gdr", _create_gdr_peer_adapter)
