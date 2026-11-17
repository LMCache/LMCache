# SPDX-License-Identifier: Apache-2.0
"""Open the Q ring owned by vLLM process, for the SDK's transfers.

vLLM's worker captures each step's query rows into a paged Q ring
(``lmcache.sdk.qringbuffer.QRingBuffer``) and stores full chunks to L1. The SDK
borrows the ring's **free blocks**: the server copies cached L1 objects into them.
This module opens the ring through its own IPC handles, which the server forwards
(``get_q_ring``), and holds what every transfer shares: the reservation of free
pieces, so concurrent transfers use disjoint pieces, and the check that vLLM took
no ring blocks.
"""

# Future
from __future__ import annotations

# Standard
import dataclasses
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_codec import TOKEN_STRIDE
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.transport.qstore import QRingInfo

logger = init_logger(__name__)


class RingStateError(RuntimeError):
    """Raised when the Q ring changed under a retrieve, or cannot serve it."""


# Ring instance ID --> pieces reserved by in-flight transfers, shared by every
# context of the process (the query and KV contexts borrow the same ring).
_RESERVED: dict[int, set[int]] = {}
_RESERVED_LOCK = threading.Lock()

# Wait (in seconds) for free pieces in QRingBuffer to transfer.
_POLL_INTERVAL = 0.1


def check_ring_unchanged(before: QRingInfo, after: QRingInfo, what: str) -> None:
    """Fail if vLLM took ring blocks, or the ring changed, between two reads.

    A restarted vLLM registers its ring under a new instance ID.

    Args:
        before: The ring as read before the transfer.
        after: The ring as read after it.
        what: The tensors transferred, for the message.

    Raises:
        RingStateError: If the allocation count or the instance ID changed.
    """
    if (before.instance_id, before.num_alloc) != (after.instance_id, after.num_alloc):
        raise RingStateError(
            "vLLM took Q ring blocks or replaced the ring during the transfer "
            f"(instance, num_alloc before: {before.instance_id}, "
            f"{before.num_alloc}; after: {after.instance_id}, {after.num_alloc}); "
            f"{what} must be transferred while vLLM schedules no requests"
        )


@dataclasses.dataclass
class MappedRing:
    """One worker's Q ring, opened in this process.

    Attributes:
        info: The ring as the server last reported it.
        layers: The ring's per-layer tensors, each
            ``[num_blocks, block_size, hidden]``.
    """

    info: QRingInfo
    layers: list[torch.Tensor]

    @property
    def block_size(self) -> int:
        """Tokens per ring block."""
        return self.layers[0].shape[1]


@dataclasses.dataclass(frozen=True)
class RingReservation:
    """Free pieces reserved for one transfer batch.

    Attributes:
        ring: The ring as when reserving; compare ``ring.info`` with a later
            read to check vLLM took no blocks.
        pieces: The reserved pieces, for whole chunks.
    """

    ring: MappedRing
    pieces: list[int]


class RingMapping:
    """The engine's Q ring, opened in this process (TP 1).

    Each transfer reserves some free piece_ids from QRingBuffer blocks, and
    release them when done.

    Args:
        req_client: Client of the LMCache server.
        model_name: The query model name (``<model>##query``).
        world_size: World size of the engine's registration.
        timeout: Timeout in seconds for each server call.
    """

    # Transferring all requests' KV cache at once slows down the transfer since
    # they share the host's memory bandwidth and hold the QRingBuffer pieces
    # longer.
    _max_transfers: int = 2
    _transfer_slots: threading.Semaphore = threading.Semaphore(_max_transfers)

    @staticmethod
    def set_max_transfers(max_transfers: int) -> None:
        """Set how many transfers the process runs at once (default 2).

        Every mapping of the process shares the transfer slots, so the last
        call wins. Safe while transfers run: those holding a slot release it
        as usual, and the limit may be exceeded until they finish.

        Args:
            max_transfers: Transfers at once; at least 1 (1 serialises them).

        Raises:
            ValueError: If ``max_transfers`` is less than 1.
        """
        if max_transfers < 1:
            raise ValueError(f"max_transfers must be at least 1, got {max_transfers}")
        if max_transfers != RingMapping._max_transfers:
            RingMapping._max_transfers = max_transfers
            RingMapping._transfer_slots = threading.Semaphore(max_transfers)

    def __init__(
        self,
        req_client: RequestClient,
        model_name: str,
        world_size: int,
        timeout: float,
    ) -> None:
        self._req_client = req_client
        self._model_name = model_name
        self._world_size = world_size
        self._timeout = timeout
        self._ring: MappedRing | None = None

    @property
    def req_client(self) -> RequestClient:
        """The client the ring is read and written through."""
        return self._req_client

    @property
    def transfer_slot(self) -> threading.Semaphore:
        """One of the process's ``max_transfers`` transfer slots: hold it
        (``with``) across a transfer, and a retrieve's lookup."""
        return RingMapping._transfer_slots

    @property
    def ring(self) -> MappedRing:
        """The opened ring."""
        if self._ring is None:
            raise RingStateError("the Q ring is not open; call connect()")
        return self._ring

    @property
    def device(self) -> torch.device:
        """The device the ring lives on."""
        return self.ring.layers[0].device

    @property
    def piece_bytes(self) -> int:
        """Bytes of one piece: one layer's slice of one ring block."""
        layer = self.ring.layers[0]
        return layer[0].numel() * layer.element_size()

    def connect(self) -> None:
        """Open the engine's ring.

        Raises:
            RingStateError: If no ring is registered for the model.
        """
        info = self.get_ring()
        with _RESERVED_LOCK:
            self.map_ring(info)

    def close(self) -> None:
        """Drop the opened ring."""
        with _RESERVED_LOCK:
            self._ring = None
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()

    def get_ring(self) -> QRingInfo:
        """Return the engine's ring as the server last saw it.

        With several engines of the model registered, keeps the open ring
        while it stays registered, and otherwise takes the first.

        Raises:
            RingStateError: If no ring is registered for the model.
        """
        rings = self._req_client.get_q_ring(self._model_name, self._world_size).result(
            timeout=self._timeout
        )
        if not rings:
            raise RingStateError(
                f"no Q ring is registered for {self._model_name!r}; start vLLM "
                "with lmcache.mp.transfer_intermediate_tensors"
            )
        current = self._ring
        if current is not None:
            for info in rings:
                if info.instance_id == current.info.instance_id:
                    return info
        return rings[0]

    def map_ring(self, info: QRingInfo) -> None:
        """Open the ring, unless the same worker's ring is already open.

        A worker's ring keeps its memory, and so its IPC handles, for the
        worker's lifetime, re-registrations included; a restarted worker
        registers under a new instance ID.
        """
        if self._ring is not None and self._ring.info.instance_id == info.instance_id:
            self._ring.info = info
            return
        self._ring = MappedRing(
            info, [wrapper.to_tensor() for wrapper in info.q_caches]
        )
        logger.info(
            "Opened the Q ring of %s: %d layers x %s, instance %s",
            self._model_name,
            len(self._ring.layers),
            list(self._ring.layers[0].shape),
            info.instance_id,
        )

    def reserve(self, pieces_per_chunk: int, max_chunks: int) -> RingReservation:
        """Reserve free pieces for up to ``max_chunks`` chunks.

        Takes as many whole chunks as the ring's unreserved free pieces hold,
        at least one; when not even one chunk fits, polls every
        ``_POLL_INTERVAL`` seconds until other transfers release pieces. The
        reservation holds ``len(pieces) // pieces_per_chunk`` chunks; release
        it with ``release``.

        Args:
            pieces_per_chunk: Pieces one chunk's objects need.
            max_chunks: Chunks wanted; at least one.

        Returns:
            The reservation.

        Raises:
            RingStateError: If no ring is registered, one chunk does not fit
                the free pieces with no other transfer in flight, or none fits
                within the timeout.
        """
        deadline = time.monotonic() + self._timeout
        while True:
            info = self.get_ring()
            with _RESERVED_LOCK:
                self.map_ring(info)
                ring = self.ring
                free = self._free_pieces(ring)
                count = (
                    min(max_chunks, len(free) // pieces_per_chunk) * pieces_per_chunk
                )
                if count:
                    _RESERVED.setdefault(info.instance_id, set()).update(free[:count])
                    return RingReservation(MappedRing(info, ring.layers), free[:count])
                if not _RESERVED.get(info.instance_id):
                    raise RingStateError(
                        f"the Q ring has {len(free)} free pieces; one chunk needs "
                        f"{pieces_per_chunk}. Enlarge lmcache.q.ring_blocks"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RingStateError(
                        f"no free Q ring pieces for one chunk within "
                        f"{self._timeout}s; other transfers hold them"
                    )
            time.sleep(min(_POLL_INTERVAL, remaining))

    def release(self, reservation: RingReservation) -> None:
        """Return a reservation's pieces, once its transfer is done with them."""
        with _RESERVED_LOCK:
            _RESERVED[reservation.ring.info.instance_id].difference_update(
                reservation.pieces
            )

    @staticmethod
    def _free_pieces(ring: MappedRing) -> list[int]:
        """The ring's pieces in blocks vLLM does not hold and no transfer has
        reserved, in block order. Must hold ``_RESERVED_LOCK``."""
        num_layers, num_blocks = len(ring.layers), ring.layers[0].shape[0]
        used = set(ring.info.used_blocks)
        reserved = _RESERVED.get(ring.info.instance_id, set())
        return [
            piece
            for block in range(num_blocks)
            if block not in used
            for piece in range(block * num_layers, (block + 1) * num_layers)
            if piece not in reserved
        ]

    def free_lookup_locks(self, key: IPCCacheServerKey, start: int, end: int) -> None:
        """Release the lookup locks of ``[start, end)`` that go unretrieved."""
        if start >= end:
            return
        lock_key = dataclasses.replace(
            key, token_bytes=key.token_bytes[: end * TOKEN_STRIDE], start=start, end=end
        ).no_worker_id_version()
        try:
            self._req_client.free_lookup_locks(lock_key, key.world_size).result(
                timeout=self._timeout
            )
        except TimeoutError:
            logger.warning(
                "FREE_LOOKUP_LOCKS timed out for request_id=%s at [%d, %d)",
                key.request_id,
                start,
                end,
            )
