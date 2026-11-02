# SPDX-License-Identifier: Apache-2.0
"""Retrieve query tensors through the Q ring the serving engine owns.

vLLM's worker captures each step's query rows into a paged Q ring
(``lmcache.sdk.qringbuffer.QRingBuffer``) and stores full chunks to L1. Instead
of registering a second pool, the SDK asks the server to copy cached chunks
into **free blocks of that ring** (``retrieve_q``), then gathers them through
the ring's own IPC handles, which the server forwards (``get_q_ring``).

This relies on exclusive phases: while the SDK retrieves, vLLM must run no
steps (the token-dropping workflow pauses every stream during ``modify``).
The worker reports its free list and step counter at the end of every step
that changed it; a retrieve fails if the step moved while it ran.
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
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

logger = init_logger(__name__)


class QRingStateError(RuntimeError):
    """Raised when the Q ring changed under a retrieve, or cannot serve it."""


@dataclasses.dataclass
class _MappedRing:
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


class QRingTransferWrapper:
    """Retrieve query tensors into free blocks of the engine's Q rings.

    Has the retrieve interface of ``PagedPoolTransferWrapper``, so an
    ``LMCacheSDKContext`` uses either without change. Retrieves are
    serialised: one retrieve owns the free blocks it fills until it has
    gathered them.

    Args:
        req_client: Client of the LMCache server.
        model_name: The query model name (``<model>##query``).
        world_size: World size of the engine's registration.
        chunk_size: Tokens per LMCache chunk.
        timeout: Timeout in seconds for each server call.
    """

    def __init__(
        self,
        req_client: RequestClient,
        model_name: str,
        world_size: int,
        chunk_size: int,
        timeout: float,
    ) -> None:
        self._req_client = req_client
        self._model_name = model_name
        self._world_size = world_size
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._rings: list[_MappedRing] = []
        self._lock = threading.Lock()

    @property
    def has_recurrent_state(self) -> bool:
        """Query tensors carry no recurrent state."""
        return False

    @property
    def sw_size_tokens(self) -> int:
        """The window the engine's ring is registered with."""
        return self._rings[0].info.sw_size_tokens

    @property
    def device(self) -> torch.device:
        """The device the rings live on."""
        return self._rings[0].layers[0].device

    def connect(self) -> None:
        """Open the engine's rings.

        Raises:
            QRingStateError: If no ring is registered for the model, or the
                ranks disagree on the ring layout.
        """
        with self._lock:
            self._map(self._get_rings())

    def retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[torch.Tensor, None] | None:
        """Retrieve ``[key.start, key.end)`` through the engine's rings.

        ``[0, key.end)`` must be read-locked by a lookup under
        ``key.request_id``; the prefix ``[0, key.start)`` is released, and on
        failure the rest of the range.

        Args:
            key: The cache server key; ``key.token_ids`` covers ``key.end``.
            instance_id: Unused: the rings belong to the engine's instances.

        Returns:
            ``(query, None)``: a CPU tensor ``[1, L, T, hidden * world_size]``
            of the rows each chunk keeps, the ranks' heads concatenated; or
            None if the range is empty or the server fails the retrieve.

        Raises:
            QRingStateError: If vLLM stepped during the retrieve (the
                exclusive-phase requirement was violated), a ring was
                re-registered, or no free blocks are available.
        """
        del instance_id
        self._free_lookup_locks(key, 0, key.start)
        if key.end <= key.start:
            return None
        with self._lock:
            before = self._get_rings()
            self._map(before)
            parts = self._retrieve_batches(key, before)
            after = self._get_rings()
        changed = [
            (b.worker_id, b.step, a.step)
            for b, a in zip(before, after, strict=False)
            if (b.step, b.generation) != (a.step, a.generation)
        ]
        if changed or len(after) != len(before):
            raise QRingStateError(
                "vLLM changed its Q ring during the retrieve "
                f"(worker, step before, step after: {changed}); query tensors "
                "must be retrieved while vLLM runs no steps"
            )
        if parts is None:
            return None
        return torch.cat(parts, dim=2), None

    def store(self, *_args: object, **_kwargs: object) -> bool:
        """The SDK never writes query tensors.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("query tensors are read-only through the Q ring")

    def close(self) -> None:
        """Drop the opened rings."""
        with self._lock:
            self._rings = []
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()

    def _get_rings(self) -> list[QRingInfo]:
        rings = self._req_client.get_q_ring(self._model_name, self._world_size).result(
            timeout=self._timeout
        )
        if not rings:
            raise QRingStateError(
                f"no Q ring is registered for {self._model_name!r}; start vLLM "
                "with lmcache.mp.transfer_intermediate_tensors"
            )
        return list(rings)

    def _map(self, rings: Sequence[QRingInfo]) -> None:
        """Open the rings, unless the same registrations are already open."""
        current = [(r.info.instance_id, r.info.generation) for r in self._rings]
        if current == [(r.instance_id, r.generation) for r in rings]:
            for mapped, info in zip(self._rings, rings, strict=True):
                mapped.info = info
            return
        self._rings = [
            _MappedRing(info, [wrapper.to_tensor() for wrapper in info.q_caches])
            for info in rings
        ]
        shapes = {tuple(ring.layers[0].shape[1:]) for ring in self._rings}
        if len(shapes) != 1:
            raise QRingStateError(f"the ranks' Q rings disagree on layout: {shapes}")
        logger.info(
            "Opened %d Q ring(s) for %s: %d layers x %s, sw_size_tokens=%d, "
            "generation %s",
            len(self._rings),
            self._model_name,
            len(self._rings[0].layers),
            list(self._rings[0].layers[0].shape),
            self.sw_size_tokens,
            [ring.info.generation for ring in self._rings],
        )

    def _retrieve_batches(
        self, key: IPCCacheServerKey, rings: Sequence[QRingInfo]
    ) -> list[torch.Tensor] | None:
        """Retrieve the range in batches that fit every ring's free blocks."""
        block_size = self._rings[0].block_size
        blocks_per_chunk = self._chunk_size // block_size
        free = [sorted(ring.free_blocks) for ring in rings]
        chunks_per_batch = min(len(f) for f in free) // blocks_per_chunk
        if chunks_per_batch == 0:
            self._free_lookup_locks(key, key.start, key.end)
            raise QRingStateError(
                f"the Q ring has {min(len(f) for f in free)} free blocks, fewer "
                f"than one chunk's {blocks_per_chunk}; enlarge "
                "lmcache.mp.q.ring_blocks"
            )
        step = chunks_per_batch * self._chunk_size
        parts: list[torch.Tensor] = []
        for start in range(key.start, key.end, step):
            end = min(start + step, key.end)
            num_blocks = (end - start) // self._chunk_size * blocks_per_chunk
            heads = []
            for ring, free_blocks in zip(self._rings, free, strict=True):
                block_ids = free_blocks[:num_blocks]
                rows = self._retrieve_into(ring, key, start, end, block_ids)
                if rows is None:
                    self._free_lookup_locks(key, end, key.end)
                    return None
                heads.append(rows)
            parts.append(torch.cat(heads, dim=-1))
        return parts

    def _retrieve_into(
        self,
        ring: _MappedRing,
        key: IPCCacheServerKey,
        start: int,
        end: int,
        block_ids: list[int],
    ) -> torch.Tensor | None:
        """Copy ``[start, end)`` into one ring's blocks and gather them."""
        device = ring.layers[0].device
        event_backend = get_event_ipc_backend(device)
        worker_id = max(ring.info.worker_id, 0)
        sub_key = dataclasses.replace(
            key,
            token_ids=key.token_ids[:end],
            start=start,
            end=end,
            worker_id=worker_id,
        )
        with torch_dev.device(device):
            event = event_backend.create_event(device)
            event_backend.record_event(event, torch_dev.current_stream())
            future = self._req_client.retrieve_q(
                sub_key,
                ring.info.instance_id,
                block_ids,
                event_backend.export_event(event, device),
            ).to_device_future(device=device, event_backend=event_backend)
            if not future.result(timeout=self._timeout):
                logger.warning(
                    "Q ring retrieve failed for request_id=%s at [%d, %d)",
                    key.request_id,
                    start,
                    end,
                )
                return None
            index = torch.tensor(
                self._kept_block_ids(block_ids, ring.block_size), device=device
            )
            # [L, kept blocks, block_size, hidden] -> [1, L, T, hidden]
            rows = torch.stack(
                [layer.index_select(0, index) for layer in ring.layers]
            ).flatten(1, 2)
            rows = rows.unsqueeze(0).to("cpu")
        return rows

    def _kept_block_ids(self, block_ids: list[int], block_size: int) -> list[int]:
        """The trailing blocks of every chunk the ring's window keeps."""
        per_chunk = self._chunk_size // block_size
        sw_size_tokens = self.sw_size_tokens
        tokens = (
            self._chunk_size
            if sw_size_tokens < 0
            else min(sw_size_tokens, self._chunk_size)
        )
        kept = tokens // block_size
        return [
            block_id
            for offset in range(0, len(block_ids), per_chunk)
            for block_id in block_ids[offset : offset + per_chunk][-kept:]
        ]

    def _free_lookup_locks(self, key: IPCCacheServerKey, start: int, end: int) -> None:
        """Release the lookup locks of ``[start, end)`` that go unretrieved."""
        if start >= end:
            return
        lock_key = dataclasses.replace(
            key, token_ids=key.token_ids[:end], start=start, end=end
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
