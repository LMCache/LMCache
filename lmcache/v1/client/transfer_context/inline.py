# SPDX-License-Identifier: Apache-2.0
"""Inline EngineDrivenContext implementation for multiprocess mode."""

# Standard
from typing import cast
import math

# Third Party
import torch

# First Party
from lmcache.v1.client.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)
from lmcache.v1.client.transfer_context.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.transport.request_client import RequestClient

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


class EngineDrivenContextInline(EngineDrivenContext):
    """Safe inline implementation of :class:`EngineDrivenContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE`` (returns empty slots
      for inline mode); ``commit_store`` serialises chunks and sends
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
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Send PREPARE_STORE RPC and return no pre-allocated buffers."""
        future = self.req_client.prepare_store(key, instance_id)
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return None

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: "list[torch.Tensor] | list[tuple[list[int], list[torch.Tensor]]]",
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE, one page at a time.

        Single-group callers pass a flat chunk list, sent as a tensor
        envelope. Multi-group callers pass, per group, ``(chunk_indices,
        tensors)`` -- the chunks that group stores, which omit all-null
        chunks -- sent as a grouped envelope that the server validates
        against each group's registered layout. Pages are committed in order
        and the first failed page stops the store; earlier pages stay stored.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        grouped = bool(chunks) and isinstance(chunks[0], tuple)
        groups = cast("list[tuple[list[int], list[torch.Tensor]]]", chunks)
        flat = cast("list[torch.Tensor]", chunks)
        if grouped:
            sizes = [0] * (1 + max((ix[-1] for ix, _ in groups if ix), default=-1))
            for indices, tensors in groups:
                for idx, tensor in zip(indices, tensors, strict=True):
                    sizes[idx] += tensor.nbytes
        else:
            sizes = [tensor.nbytes for tensor in flat]
        pages = _pages(sizes)
        for i, (start, end) in enumerate(pages):
            if grouped:
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
            else:
                payload = encode_tensors(flat[start:end])
            future = self.req_client.commit_store(
                key, instance_id, payload, start, _page_count(pages, i)
            )
            try:
                if not future.result(timeout=self.mq_timeout):
                    return False
            except TimeoutError:
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
