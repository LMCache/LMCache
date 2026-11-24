# SPDX-License-Identifier: Apache-2.0
"""Inline EngineDrivenContext implementation for multiprocess mode."""

# Standard
from typing import Any, cast
import math

# Third Party
import torch

# First Party
from lmcache.v1.client.transfer_context.base import (
    EngineDrivenContext,
)
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.transfer_context import EngineDrivenContextMetadata
from lmcache.v1.transport.request_client import RequestClient
from lmcache.v1.transport.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)

# Tensor bytes per inline message. gRPC rejects messages over 2 GiB and ZMQ's
# msgpack over 4 GiB; 64-256 MiB pages measured as fast as one message.
INLINE_PAGE_BYTES = 128 * 1024 * 1024


def _pages(sizes: list[int]) -> list[tuple[int, int]]:
    """Cut chunk positions ``[0, len(sizes))`` into consecutive pages.

    Each page holds at most :data:`INLINE_PAGE_BYTES`, or one chunk if that
    chunk alone is larger.

    Args:
        sizes: Tensor bytes at each chunk position (``0`` for none).

    Returns:
        ``[start, end)`` per page, covering every position in order.
    """
    pages: list[tuple[int, int]] = []
    start = used = 0
    for pos, size in enumerate(sizes):
        if pos > start and used + size > INLINE_PAGE_BYTES:
            pages.append((start, pos))
            start, used = pos, 0
        used += size
    pages.append((start, len(sizes)))
    return pages


def _page_count(pages: list[tuple[int, int]], i: int) -> int:
    """Chunk count to send for page ``i``; the last page sends ``0`` (to the end)."""
    start, end = pages[i]
    return 0 if i == len(pages) - 1 else end - start


def _run_pages(
    positions: list[int], sizes: list[int], num_chunks: int
) -> list[tuple[int, int, int, int]]:
    """Cut the chunks at increasing ``positions`` into flat-envelope pages.

    A flat envelope carries exactly the chunks of its range, so a page ends
    where the positions skip a chunk, or where it would exceed
    :data:`INLINE_PAGE_BYTES` (it holds one chunk at least).

    Args:
        positions: Increasing chunk positions within the key.
        sizes: Tensor bytes of each chunk, parallel to ``positions``.
        num_chunks: Chunk positions in the key.

    Returns:
        ``(lo, hi, start, count)`` per page: the page carries the chunks at
        ``positions[lo:hi]`` and is sent as chunk ``start`` and ``count``.
        The last page sends count ``0`` (through the last chunk); when the
        last chunk is not sent, that page is empty and starts at
        ``num_chunks``.
    """
    pages: list[tuple[int, int, int, int]] = []
    lo = used = 0
    for i, (pos, size) in enumerate(zip(positions, sizes, strict=True)):
        if i > lo and (pos != positions[i - 1] + 1 or used + size > INLINE_PAGE_BYTES):
            pages.append((lo, i, positions[lo], i - lo))
            lo, used = i, 0
        used += size
    end = len(positions)
    if positions and positions[-1] == num_chunks - 1:
        pages.append((lo, end, positions[lo], 0))
        return pages
    if lo < end:
        pages.append((lo, end, positions[lo], end - lo))
    pages.append((end, end, num_chunks, 0))
    return pages


class EngineDrivenContextInline(EngineDrivenContext):
    """Safe inline implementation of :class:`EngineDrivenContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE``, which names the
      chunks to send; ``commit_store`` serialises those chunks and sends
      ``COMMIT_STORE``.
    - **Retrieve**: ``prepare_retrieve`` sends ``PREPARE_RETRIEVE`` and
      deserialises the returned bytes; ``commit_retrieve`` sends
      ``COMMIT_RETRIEVE`` (no-op for inline transfer).
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        req_client: RequestClient,
        mq_timeout: float,
    ) -> None:
        super().__init__(metadata, req_client, mq_timeout)

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor] | None, list[int]] | None:
        """Send PREPARE_STORE and return the chunks the server names.

        Returns:
            ``(None, chunk_indices)``: no pre-allocated buffers, and the
            chunk positions to send. ``None`` on a timeout, or when the
            server names no chunks, as one that predates chunk selection
            does: the caller then sends every chunk.
        """
        context = self._prepare_store_context(key, instance_id, ())
        chunk_indices = context.get("chunk_indices") if context else None
        if not isinstance(chunk_indices, list):
            return None
        return None, chunk_indices

    def prepare_store_multigroup(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> tuple[list[int], list[int]] | None:
        """Send a multi-group PREPARE_STORE; return the chunks the server names.

        Named apart from the SHM ``prepare_store_grouped`` because inline
        mode returns chunk positions, not buffers.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            skipped_chunks: Per group, chunk positions the worker will not
                store (all-null engine blocks); the server never names them.

        Returns:
            Parallel ``(chunk_indices, group_ids)`` lists naming each chunk
            to send. ``None`` on a timeout, or when the server names no
            chunks: the caller then sends every chunk it stores.
        """
        context = self._prepare_store_context(key, instance_id, skipped_chunks)
        chunk_indices = context.get("chunk_indices") if context else None
        group_ids = context.get("group_ids") if context else None
        if not isinstance(chunk_indices, list) or not isinstance(group_ids, list):
            return None
        return chunk_indices, group_ids

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: "list[torch.Tensor] | list[tuple[list[int], list[torch.Tensor]]]",
        chunk_indices: list[int] | None = None,
        num_chunks: int = 0,
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE, one page at a time.

        Single-group callers pass a flat chunk list, sent as tensor
        envelopes; ``chunk_indices`` and ``num_chunks`` place the chunks in
        the key when the server named a subset of it (see
        :func:`_run_pages`). Multi-group callers pass, per group,
        ``(chunk_indices, tensors)`` -- the chunks that group stores, which
        omit all-null chunks and the ones the server did not name -- sent as
        a grouped envelope that the server validates against each group's
        registered layout. Pages are committed in order and the first failed
        page stops the store; earlier pages stay stored.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        if chunks and isinstance(chunks[0], tuple):
            groups = cast("list[tuple[list[int], list[torch.Tensor]]]", chunks)
            sizes = [0] * (1 + max((ix[-1] for ix, _ in groups if ix), default=-1))
            for indices, tensors in groups:
                for idx, tensor in zip(indices, tensors, strict=True):
                    sizes[idx] += tensor.nbytes
            pages = _pages(sizes)
            for i, (start, end) in enumerate(pages):
                payload = encode_grouped_tensors(
                    [
                        (
                            [idx for idx in indices if start <= idx < end],
                            [
                                tensor
                                for idx, tensor in zip(indices, tensors, strict=True)
                                if start <= idx < end
                            ],
                        )
                        for indices, tensors in groups
                    ]
                )
                if not self._commit_page(
                    key, instance_id, payload, start, _page_count(pages, i)
                ):
                    return False
            return True
        flat = cast("list[torch.Tensor]", chunks)
        if chunk_indices is None:
            chunk_indices, num_chunks = list(range(len(flat))), len(flat)
        for lo, hi, start, count in _run_pages(
            chunk_indices, [tensor.nbytes for tensor in flat], num_chunks
        ):
            if not self._commit_page(
                key, instance_id, encode_tensors(flat[lo:hi]), start, count
            ):
                return False
        return True

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int, expected_count: int
    ) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE and deserialize the response data.

        Args:
            key: Cache key to retrieve.
            instance_id: Worker instance identifier.
            expected_count: Number of LMCache chunks derived from the worker's
                block IDs and blocks-per-chunk geometry.

        Returns:
            Chunks on hit, or None on miss/timeout.
        """
        shape = self.metadata.layout_desc.shapes[0]
        dtype = self.metadata.layout_desc.dtypes[0]
        pages = _pages([math.prod(shape) * dtype.itemsize] * expected_count)
        chunks: list[torch.Tensor] = []
        for i, (start, end) in enumerate(pages):
            data = self._retrieve_page(key, instance_id, start, _page_count(pages, i))
            if data is None:
                return None
            chunks += decode_tensors(
                data,
                expected_shape=shape,
                expected_dtype=dtype,
                expected_count=end - start,
            )
        return chunks

    def _retrieve_page(
        self, key: IPCCacheServerKey, instance_id: int, start: int, count: int
    ) -> bytes | None:
        """Send one PREPARE_RETRIEVE page; return its payload, or None on miss."""
        future = self.req_client.prepare_retrieve(key, instance_id, start, count)
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        return response.data

    def prepare_retrieve_multigroup(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        expected: list[tuple[torch.Size, torch.dtype, int]],
        first_chunks: "list[int] | None" = None,
    ) -> list[tuple[list[int], list[torch.Tensor]]] | None:
        """Send PREPARE_RETRIEVE and decode a group-major payload.

        The server responds with each group's in-window chunks and their
        chunk positions (all-or-nothing), or a miss. Named apart from the SHM
        ``prepare_retrieve_grouped`` because the result is the chunk data
        itself, not slot views.

        Args:
            key: Cache key to retrieve.
            instance_id: Worker instance identifier.
            expected: One ``(shape, dtype, num_chunks)`` per group, from the
                worker's registered group layouts and block geometry.
            first_chunks: Per group, the first chunk of its window, used to
                size pages; ``None`` sizes them as if every window were full.

        Returns:
            Per group ``(chunk_indices, tensors)`` on hit, or None on
            miss/timeout.
        """
        firsts = first_chunks or [0] * len(expected)
        num_chunks = max((count for _, _, count in expected), default=0)
        sizes = [
            sum(
                math.prod(shape) * dtype.itemsize
                for (shape, dtype, _), first in zip(expected, firsts, strict=True)
                if pos >= first
            )
            for pos in range(num_chunks)
        ]
        pages = _pages(sizes)
        merged: list[tuple[list[int], list[torch.Tensor]]] = [
            ([], []) for _ in expected
        ]
        for i, (start, _) in enumerate(pages):
            data = self._retrieve_page(key, instance_id, start, _page_count(pages, i))
            if data is None:
                return None
            for (indices, tensors), (page_indices, page_tensors) in zip(
                merged, decode_grouped_tensors(data, expected=expected), strict=True
            ):
                indices += page_indices
                tensors += page_tensors
        return merged

    def commit_retrieve(
        self, key: IPCCacheServerKey, instance_id: int, failed: bool = False
    ) -> bool:
        """Send COMMIT_RETRIEVE (no-op for inline path)."""
        future = self.req_client.commit_retrieve(key, instance_id, failed)
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return True

    def close(self) -> None:
        """No-op: the inline path holds no persistent resources."""

    def _prepare_store_context(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...],
    ) -> dict[str, Any] | None:
        """Send PREPARE_STORE; return the response context, None on timeout."""
        future = (
            self.req_client.prepare_store(key, instance_id, skipped_chunks)
            if skipped_chunks
            else self.req_client.prepare_store(key, instance_id)
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        return response.context if isinstance(response.context, dict) else None

    def _commit_page(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        payload: bytes,
        start: int,
        count: int,
    ) -> bool:
        """Send one COMMIT_STORE page; return whether the server stored it."""
        future = self.req_client.commit_store(key, instance_id, payload, start, count)
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False
