# SPDX-License-Identifier: Apache-2.0
"""
LMCache-driven store/retrieve of contiguous tensors via a registered
intermediate pool.
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
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)

NULL_BLOCK_ID = 0
"""Block id the server treats as "no data" (vLLM's null block): a chunk whose
block ids are all null is not stored. The pool never puts data in it."""


class PoolCapacityError(ValueError):
    """Raised when a transfer that cannot be batched exceeds the pool."""


@dataclasses.dataclass(frozen=True)
class RecurrentState:
    """Recurrent-state pages at the end of a cached prefix.

    Attributes:
        pages: One CPU tensor per recurrent layer, in registration order,
            each ``[blocks_per_chunk, *page_dims]``: the raw pages of the
            prefix's last chunk. The bytes are opaque (e.g. conv | ssm | pad
            for a GDN layer), only meaningful to the engine that wrote them.
    """

    pages: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class PoolGroup:
    """One kernel group of the pool, mirroring the engine's.

    Attributes:
        layer_names: Pool tensors of the group, in registration order.
        tokens_per_block: Tokens per paged block (the engine's block unit).
        recurrent: Whether the pages hold recurrent state snapshots.
        kernel_block_size: Attention only: tokens per physical kernel page.
            Smaller than ``tokens_per_block`` when the engine sub-pages each
            block (the registered page then interleaves K and V every
            ``kernel_block_size`` tokens); 0 or ``tokens_per_block`` otherwise.
    """

    layer_names: tuple[str, ...]
    tokens_per_block: int
    recurrent: bool = False
    kernel_block_size: int = 0

    @property
    def sub_paged(self) -> bool:
        """Whether each block's K/V are interleaved at kernel-page granularity."""
        return (
            not self.recurrent and 0 < self.kernel_block_size != self.tokens_per_block
        )


def subpaged_to_kv(pages: torch.Tensor, block: int, kernel: int) -> torch.Tensor:
    """Decode sub-paged attention rows into true K and V planes.

    Each registered block is viewed as ``[2, block, W]``, but its bytes are
    ``block // kernel`` kernel pages of ``[K(kernel, W) | V(kernel, W)]``.

    Args:
        pages: ``[2, L, T, W]`` rows as the registered view gathers them,
            ``T`` a multiple of ``block``.
        block: Tokens per registered block.
        kernel: Tokens per kernel page.

    Returns:
        ``[2, L, T, W]`` with plane 0 the true K and plane 1 the true V.
    """
    _, num_layers, num_tokens, width = pages.shape
    num_blocks, per_block = num_tokens // block, block // kernel
    # Each block's bytes, in memory order: plane 0 rows, then plane 1 rows.
    raw = pages.reshape(2, num_layers, num_blocks, block * width).permute(1, 2, 0, 3)
    raw = raw.reshape(num_layers, num_blocks, per_block, 2, kernel, width)
    return raw.permute(3, 0, 1, 2, 4, 5).reshape(2, num_layers, num_tokens, width)


def kv_to_subpaged(kv: torch.Tensor, block: int, kernel: int) -> torch.Tensor:
    """Encode true K/V planes into the sub-paged registered rows.

    The inverse of :func:`subpaged_to_kv`.

    Args:
        kv: ``[2, L, T, W]`` true K and V, ``T`` a multiple of ``block``.
        block: Tokens per registered block.
        kernel: Tokens per kernel page.

    Returns:
        ``[2, L, T, W]`` rows in the registered view's order.
    """
    _, num_layers, num_tokens, width = kv.shape
    num_blocks, per_block = num_tokens // block, block // kernel
    raw = kv.reshape(2, num_layers, num_blocks, per_block, kernel, width)
    raw = raw.permute(1, 2, 3, 0, 4, 5).reshape(num_layers, num_blocks, 2, -1)
    return raw.permute(2, 0, 1, 3).reshape(2, num_layers, num_tokens, width)


class PagedPoolTransferWrapper:
    """Store/retrieve contiguous tensors through a registered paged pool.

    The pool is a fixed set of paged blocks registered once with the server
    in lmcache-driven mode; block 0 of every group is the null block and
    never holds data. The server copies between its storage and the pool
    over device IPC, and this wrapper copies between the pool and the
    caller's contiguous CPU tensor.

    Without recurrent groups, a transfer longer than the pool is split into
    pool-sized batches. With them it cannot be: the server windows every call
    from its own end, so all batches but the last would ask for a recurrent
    state that was never stored. Such a transfer is one call, and must fit.

    Calls may run concurrently (e.g. one thread per request stream): each
    batch holds the pool from filling it to draining it, so batches of
    different calls interleave but never overwrite each other.

    Args:
        context: Registered lmcache-driven transfer context owning the pool.
        instance_id: The instance ID the pool is registered under.
        pool: The registered per-layer pool tensors, in registration order.
        groups: The pool's kernel groups, in the server's kernel-group order.
        layout_hints: Layout hints the pool was registered with.
        chunk_size: Tokens per LMCache chunk.
        num_chunks: Chunks the attention groups' pool holds (besides the
            null block).
        num_planes: Leading dim of the contiguous tensors: 2 for split K/V
            (``[2, L, T, D]``), 1 for single-plane formats such as MLA, fused
            K/V and query (``[1, L, T, D]``).
        tokens_per_chunk: Tokens each stored chunk keeps: ``chunk_size``, or
            a sub-chunk sliding window (a multiple of the block size), whose
            tokens are the chunk's last ones.
        req_client: Client used to release lookup locks the retrieve does
            not consume.
        timeout: Timeout in seconds for each server transfer.
    """

    def __init__(
        self,
        context: LMCacheDrivenTransferContext,
        instance_id: int,
        pool: dict[str, torch.Tensor],
        groups: Sequence[PoolGroup],
        layout_hints: LayoutHints,
        chunk_size: int,
        num_chunks: int,
        num_planes: int,
        tokens_per_chunk: int,
        req_client: RequestClient,
        timeout: float,
    ) -> None:
        self._context = context
        self._instance_id = instance_id
        self._pool = pool
        self._groups = tuple(groups)
        self._layout_hints = layout_hints
        self._chunk_size = chunk_size
        self._num_chunks = num_chunks
        self._num_planes = num_planes
        self._tokens_per_chunk = tokens_per_chunk
        self._req_client = req_client
        self._timeout = timeout
        self._device = next(iter(pool.values())).device
        # One pool serves every caller; a batch owns it end to end.
        self._pool_lock = threading.Lock()

    @property
    def has_recurrent_state(self) -> bool:
        """Whether the pool mirrors recurrent groups (a hybrid model)."""
        return any(group.recurrent for group in self._groups)

    @property
    def capacity_tokens(self) -> int:
        """Most tokens one server call (one batch) can span."""
        return self._num_chunks * self._chunk_size

    def retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[torch.Tensor, RecurrentState | None] | None:
        """Retrieve ``[key.start, key.end)`` as one contiguous CPU tensor.

        ``[0, key.end)`` must be read-locked by a lookup under
        ``key.request_id``. The locks this retrieve does not consume are
        released: the prefix ``[0, key.start)``, and on a failed batch every
        batch after it.

        Args:
            key: The cache server key; ``key.token_ids`` covers ``key.end``.
            instance_id: The instance ID the pool is registered under.

        Returns:
            ``(tensor, state)``: the attention groups' rows as a CPU tensor
            ``[num_planes, L_attn, T, D]`` (true K and V, whatever the
            engine's page layout; ``T`` counts the ``tokens_per_chunk`` each
            chunk keeps), and for a hybrid model the recurrent state at
            ``key.end``, else None. None if the range is empty or any batch
            fails.

        Raises:
            ValueError: If ``instance_id`` is not the pool's.
            PoolCapacityError: If a hybrid range exceeds the pool; its locks
                are released first.
        """
        self._check_instance(instance_id)
        self._free_lookup_locks(key, 0, key.start)
        if self.has_recurrent_state and key.end - key.start > self.capacity_tokens:
            self._free_lookup_locks(key, key.start, key.end)
            raise self._capacity_error(key.end - key.start)
        parts: list[torch.Tensor] = []
        state: RecurrentState | None = None
        for start, end in self._batches(key):
            block_ids = self._block_ids((end - start) // self._chunk_size)
            with self._pool_lock, torch_dev.device(self._device):
                event = self._context.create_recorded_event()
                future = self._context.submit_retrieve(
                    key.request_id,
                    self._sub_key(key, start, end),
                    self._pool,
                    block_ids,
                    event,
                    self._blocks_per_chunk(self._groups[0]),
                )
                if not future.result(timeout=self._timeout):
                    logger.warning(
                        "Retrieve failed for request_id=%s at [%d, %d)",
                        key.request_id,
                        start,
                        end,
                    )
                    self._free_lookup_locks(key, end, key.end)
                    return None
                group_parts: list[torch.Tensor] = []
                pages: list[torch.Tensor] = []
                for group, ids in zip(self._groups, block_ids, strict=True):
                    if group.recurrent:
                        last = self._last_chunk_blocks(group)
                        pages.extend(
                            self._pool[name][last].to("cpu", non_blocking=True)
                            for name in group.layer_names
                        )
                    else:
                        group_parts.append(self._gather(group, ids))
                # The device-to-host copies are asynchronous.
                torch_dev.current_stream().synchronize()
            parts.append(torch.cat(group_parts, dim=1))
            if pages:
                state = RecurrentState(tuple(pages))
        if not parts:
            return None
        return torch.cat(parts, dim=2), state

    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        kv: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> bool:
        """Store a contiguous tensor covering ``[key.start, key.end)``.

        Args:
            key: The cache server key; ``key.token_ids`` covers ``key.end``.
            instance_id: The instance ID the pool is registered under.
            kv: Contiguous CPU tensor ``[num_planes, L_attn, T, D]`` (true K
                and V) with ``T == key.end - key.start``, chunk-aligned.
            state: Hybrid models only: one page tensor per recurrent layer
                (see :class:`RecurrentState`), stored as the state at
                ``key.end``.

        Returns:
            True if every batch was stored, False at the first failed batch.

        Raises:
            ValueError: If ``instance_id`` is not the pool's, or ``kv`` or
                ``state`` does not match the pool.
            PoolCapacityError: If a hybrid range exceeds the pool.
        """
        self._check_instance(instance_id)
        num_tokens = key.end - key.start
        num_attn_layers = sum(
            len(group.layer_names) for group in self._groups if not group.recurrent
        )
        if (
            kv.dim() != 4
            or kv.shape[0] != self._num_planes
            or kv.shape[1] != num_attn_layers
            or kv.shape[2] != num_tokens
        ):
            raise ValueError(
                f"kv shape {tuple(kv.shape)} does not match "
                f"[{self._num_planes}, {num_attn_layers}, {num_tokens}, D]"
            )
        if self.has_recurrent_state:
            num_recurrent = sum(
                len(group.layer_names) for group in self._groups if group.recurrent
            )
            if state is None or len(state.pages) != num_recurrent:
                raise ValueError(
                    f"state must hold one page per recurrent layer ({num_recurrent})"
                )
            if num_tokens > self.capacity_tokens:
                raise self._capacity_error(num_tokens)
        elif state is not None:
            raise ValueError("the pool has no recurrent groups to store state in")
        for start, end in self._batches(key):
            offset = start - key.start
            batch = kv[:, :, offset : offset + end - start, :]
            block_ids = self._block_ids((end - start) // self._chunk_size)
            with self._pool_lock, torch_dev.device(self._device):
                self._fill(batch, block_ids, state)
                event = self._context.create_recorded_event()
                future = self._context.submit_store(
                    key.request_id,
                    self._sub_key(key, start, end),
                    self._pool,
                    block_ids,
                    event,
                    self._blocks_per_chunk(self._groups[0]),
                )
                if not future.result(timeout=self._timeout):
                    logger.warning(
                        "Store failed for request_id=%s at [%d, %d)",
                        key.request_id,
                        start,
                        end,
                    )
                    return False
        return True

    def close(self) -> None:
        """Unregister the pool from the server and release the context."""
        future = self._context.unregister()
        if future is not None:
            future.result(timeout=self._timeout)
        self._context.close()

    def _check_instance(self, instance_id: int) -> None:
        """Reject an instance ID other than the one the pool belongs to."""
        if instance_id != self._instance_id:
            raise ValueError(
                f"instance_id {instance_id} does not own this pool "
                f"(registered under {self._instance_id})"
            )

    def _capacity_error(self, num_tokens: int) -> PoolCapacityError:
        return PoolCapacityError(
            f"transfer of {num_tokens} tokens exceeds the pool's "
            f"{self.capacity_tokens} tokens; a hybrid model's recurrent state "
            "cannot be split into batches, so connect with a larger pool_chunks"
        )

    def _batches(self, key: IPCCacheServerKey) -> list[tuple[int, int]]:
        """Split ``[key.start, key.end)`` into token ranges of one call each."""
        if self.has_recurrent_state:
            return [(key.start, key.end)] if key.end > key.start else []
        step = self.capacity_tokens
        return [
            (start, min(start + step, key.end))
            for start in range(key.start, key.end, step)
        ]

    def _blocks_per_chunk(self, group: PoolGroup) -> int:
        return self._chunk_size // group.tokens_per_block

    def _last_chunk_blocks(self, group: PoolGroup) -> slice:
        """Pool blocks holding a recurrent group's last-chunk state."""
        first = NULL_BLOCK_ID + 1
        return slice(first, first + self._blocks_per_chunk(group))

    def _block_ids(self, num_chunks: int) -> list[list[int]]:
        """Block ids of each kernel group for a call of ``num_chunks``.

        Attention groups use blocks ``1..n``; a recurrent group uses the null
        block for every chunk but the last, whose state the server keeps.
        """
        block_ids = []
        first = NULL_BLOCK_ID + 1
        for group in self._groups:
            bpc = self._blocks_per_chunk(group)
            if group.recurrent:
                ids = [NULL_BLOCK_ID] * ((num_chunks - 1) * bpc)
                ids += list(range(first, first + bpc))
            else:
                ids = list(range(first, first + num_chunks * bpc))
            block_ids.append(ids)
        return block_ids

    def _kept_block_ids(self, group: PoolGroup, block_ids: list[int]) -> list[int]:
        """The trailing blocks of every chunk a sub-chunk window keeps."""
        per_chunk = self._blocks_per_chunk(group)
        kept = self._tokens_per_chunk // group.tokens_per_block
        return [
            block_id
            for offset in range(0, len(block_ids), per_chunk)
            for block_id in block_ids[offset : offset + per_chunk][-kept:]
        ]

    def _gather(self, group: PoolGroup, block_ids: list[int]) -> torch.Tensor:
        """Copy an attention group's kept rows out of the pool."""
        chunks = gather_paged_kv_to_cpu(
            {name: self._pool[name] for name in group.layer_names},
            self._kept_block_ids(group, block_ids),
            self._tokens_per_chunk // group.tokens_per_block,
            layout_hints=self._layout_hints,
        )
        # Chunks are [2, L, T, D] (split K/V) or [L, T, D] (single-plane).
        rows = torch.cat(chunks, dim=chunks[0].dim() - 2)
        if self._num_planes == 1:
            return rows.unsqueeze(0)
        if group.sub_paged:
            return subpaged_to_kv(rows, group.tokens_per_block, group.kernel_block_size)
        return rows

    def _fill(
        self,
        kv: torch.Tensor,
        block_ids: list[list[int]],
        state: RecurrentState | None,
    ) -> None:
        """Copy a batch's rows (and the recurrent state) into the pool."""
        layer_offset = 0
        pages = iter(state.pages if state is not None else ())
        for group, ids in zip(self._groups, block_ids, strict=True):
            if group.recurrent:
                last = self._last_chunk_blocks(group)
                for name in group.layer_names:
                    page = next(pages)
                    target = self._pool[name][last]
                    if page.shape != target.shape:
                        raise ValueError(
                            f"state page {tuple(page.shape)} does not match "
                            f"{name}'s {tuple(target.shape)}"
                        )
                    target.copy_(page, non_blocking=True)
                continue
            num_layers = len(group.layer_names)
            rows = kv[:, layer_offset : layer_offset + num_layers]
            layer_offset += num_layers
            if group.sub_paged:
                rows = kv_to_subpaged(
                    rows, group.tokens_per_block, group.kernel_block_size
                )
            scatter_cpu_to_paged_kv(
                {name: self._pool[name] for name in group.layer_names},
                ids,
                self._to_chunks(rows),
                self._blocks_per_chunk(group),
                layout_hints=self._layout_hints,
            )

    def _to_chunks(self, rows: torch.Tensor) -> list[torch.Tensor]:
        """Split ``[num_planes, L, T, D]`` rows into per-chunk tensors."""
        chunks = []
        for offset in range(0, rows.shape[2], self._chunk_size):
            chunk = rows[:, :, offset : offset + self._chunk_size, :]
            if self._num_planes == 1:
                chunk = chunk.squeeze(0)
            chunks.append(chunk.contiguous())
        return chunks

    @staticmethod
    def _sub_key(key: IPCCacheServerKey, start: int, end: int) -> IPCCacheServerKey:
        """The key for ``[start, end)`` within ``key``'s token chain."""
        return dataclasses.replace(
            key, token_ids=key.token_ids[:end], start=start, end=end
        )

    def _free_lookup_locks(self, key: IPCCacheServerKey, start: int, end: int) -> None:
        """Release the lookup locks of ``[start, end)`` that go unretrieved."""
        if start >= end:
            return
        lock_key = self._sub_key(key, start, end).no_worker_id_version()
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
