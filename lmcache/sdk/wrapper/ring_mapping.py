# SPDX-License-Identifier: Apache-2.0
"""Open the Q ring owned by vLLM process, for the SDK's transfers.

vLLM's worker captures each step's query rows into a paged Q ring
(``lmcache.sdk.qringbuffer.QRingBuffer``) and stores full chunks to L1. The SDK
borrows the ring's **free blocks**: the server copies cached L1 objects into them.
This module opens the ring through its own IPC handles, which the server forwards
(``get_q_ring``), and holds what every transfer shares: the per-ring lock and the
check that vLLM took no ring blocks.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
import dataclasses
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, QRingInfo
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


class RingStateError(RuntimeError):
    """Raised when the Q ring changed under a retrieve, or cannot serve it."""


_RING_LOCKS: dict[tuple[str, int], threading.RLock] = {}
_RING_LOCKS_GUARD = threading.Lock()


def shared_ring_lock(model_name: str, world_size: int) -> threading.RLock:
    """Return the lock that serialises this process's use of a model's rings.

    The query and KV contexts of a process both borrow the ring's free
    blocks, so they share one lock per ring.

    Args:
        model_name: The query model name (``<model>##query``).
        world_size: World size of the engine's registration.

    Returns:
        The same lock for every caller with these arguments.
    """
    with _RING_LOCKS_GUARD:
        return _RING_LOCKS.setdefault((model_name, world_size), threading.RLock())


def check_ring_unchanged(
    before: Sequence[QRingInfo], after: Sequence[QRingInfo], what: str
) -> None:
    """Fail if vLLM took ring blocks, or a ring changed, between two reads.

    A restarted vLLM registers its ring under a new instance ID.

    Args:
        before: The rings as read before the transfer.
        after: The rings as read after it.
        what: The tensors transferred, for the message.

    Raises:
        RingStateError: If an allocation count changed, or the set of rings
            did.
    """
    changed = [
        (b.worker_id, b.num_alloc, a.num_alloc)
        for b, a in zip(before, after, strict=False)
        if (b.instance_id, b.num_alloc) != (a.instance_id, a.num_alloc)
    ]
    if changed or len(after) != len(before):
        raise RingStateError(
            "vLLM took Q ring blocks or replaced the ring during the transfer "
            f"(worker, num_alloc before, num_alloc after: {changed}); {what} "
            "must be transferred while vLLM schedules no requests"
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


class RingMapping:
    """The engine's Q rings, one per rank, opened in this process.

    Transfers through the rings' free blocks are serialised by ``lock``,
    which every context of the process shares for the same rings.

    Args:
        req_client: Client of the LMCache server.
        model_name: The query model name (``<model>##query``).
        world_size: World size of the engine's registration.
        timeout: Timeout in seconds for each server call.
    """

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
        self._rings: list[MappedRing] = []
        self._lock = shared_ring_lock(model_name, world_size)

    @property
    def lock(self) -> threading.RLock:
        """The lock any use of the rings' free blocks must hold."""
        return self._lock

    @property
    def req_client(self) -> RequestClient:
        """The client the rings are read and written through."""
        return self._req_client

    @property
    def rings(self) -> list[MappedRing]:
        """The opened rings, one per rank in rank order."""
        return self._rings

    @property
    def device(self) -> torch.device:
        """The device the rings live on."""
        return self._rings[0].layers[0].device

    def connect(self) -> None:
        """Open the engine's rings.

        Raises:
            RingStateError: If no ring is registered for the model, or the
                ranks disagree on the ring layout.
        """
        with self._lock:
            self.map_rings(self.get_rings())

    def close(self) -> None:
        """Drop the opened rings."""
        with self._lock:
            self._rings = []
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()

    def get_rings(self) -> list[QRingInfo]:
        """Return the model's rings as the server last saw them, by rank.

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
        return list(rings)

    def map_rings(self, rings: Sequence[QRingInfo]) -> None:
        """Open the rings, unless the same workers' rings are already open.

        A worker's ring keeps its memory, and so its IPC handles, for the
        worker's lifetime, re-registrations included; a restarted worker
        registers under a new instance ID.
        """
        current = [r.info.instance_id for r in self._rings]
        if current == [r.instance_id for r in rings]:
            for mapped, info in zip(self._rings, rings, strict=True):
                mapped.info = info
            return
        self._rings = [
            MappedRing(info, [wrapper.to_tensor() for wrapper in info.q_caches])
            for info in rings
        ]
        shapes = {tuple(ring.layers[0].shape[1:]) for ring in self._rings}
        if len(shapes) != 1:
            raise RingStateError(f"the ranks' Q rings disagree on layout: {shapes}")
        logger.info(
            "Opened %d Q ring(s) for %s: %d layers x %s, instances %s",
            len(self._rings),
            self._model_name,
            len(self._rings[0].layers),
            list(self._rings[0].layers[0].shape),
            [ring.info.instance_id for ring in self._rings],
        )

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
