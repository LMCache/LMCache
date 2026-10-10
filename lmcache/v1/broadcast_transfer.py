# SPDX-License-Identifier: Apache-2.0
"""Bounded windows for first-rank KV broadcast retrieval.

With ``save_only_first_rank`` the first rank owns every KV chunk and each rank
must write those chunks into its own KV pages. This module packs consecutive
chunks into a fixed device staging buffer so that the device memory a single
retrieval needs is bounded by that buffer rather than by the size of the cache
hit. The collective choreography (which rank sends what, and when) lives in
``LMCacheEngine``; this module only describes windows and moves bytes.
"""

# Standard
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# Every chunk starts on an allocator-sized boundary inside the staging buffer so
# transfer kernels see the same alignment as a standalone device allocation.
CHUNK_ALIGNMENT = 256

StagedChunk = tuple[MemoryObj, int, int]
"""A source memory object with the token range ``[start, end)`` it covers."""


def _align(num_bytes: int) -> int:
    return (num_bytes + CHUNK_ALIGNMENT - 1) // CHUNK_ALIGNMENT * CHUNK_ALIGNMENT


@dataclass(frozen=True)
class Window:
    """Layout of one broadcast: consecutive chunks packed into the staging buffer.

    The source rank builds a window from its chunks and sends it to every rank
    as a host object, so it carries plain data only. :meth:`views` rebuilds
    identical memory objects over each rank's own staging buffer.

    Attributes:
        starts: Token start of each chunk, in token order.
        ends: Token end (exclusive) of each chunk.
        offsets: Byte offset of each chunk inside the staging buffer.
        metadata: ``MemoryObjMetadata.to_dict()`` of each chunk, with
            ``address`` and ``phy_size`` describing the packed slice.
        num_bytes: Bytes of the staging buffer this window occupies.
    """

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    offsets: tuple[int, ...]
    metadata: tuple[dict, ...]
    num_bytes: int

    @classmethod
    def from_chunks(cls, chunks: Sequence[StagedChunk]) -> "Window":
        """Lay out ``chunks`` back to back, each on an aligned offset.

        Args:
            chunks: Non-empty, token-ordered chunks to pack together.

        Returns:
            The window describing where each chunk lands in the buffer.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("A window must contain at least one chunk")
        starts: list[int] = []
        ends: list[int] = []
        offsets: list[int] = []
        metadata: list[dict] = []
        num_bytes = 0
        for memory_obj, start, end in chunks:
            offset = _align(num_bytes)
            size = memory_obj.get_size()
            starts.append(start)
            ends.append(end)
            offsets.append(offset)
            metadata.append(
                {
                    **memory_obj.metadata.to_dict(),
                    "address": offset,
                    "phy_size": size,
                    "ref_count": 1,
                }
            )
            num_bytes = offset + size
        return cls(
            tuple(starts), tuple(ends), tuple(offsets), tuple(metadata), num_bytes
        )

    def views(self, buffer: torch.Tensor) -> list[MemoryObj]:
        """Memory objects over slices of ``buffer``; they own no memory.

        Args:
            buffer: Flat ``uint8`` staging tensor of at least ``num_bytes``.

        Returns:
            One memory object per chunk, sharing storage with ``buffer``.

        Raises:
            ValueError: If ``buffer`` is smaller than ``num_bytes``.
        """
        if buffer.numel() < self.num_bytes:
            raise ValueError(
                f"Staging buffer holds {buffer.numel()} bytes; "
                f"the window needs {self.num_bytes}"
            )
        return [
            TensorMemoryObj(
                buffer[offset : offset + description["phy_size"]],
                MemoryObjMetadata.from_dict(description),
                None,
            )
            for offset, description in zip(self.offsets, self.metadata, strict=True)
        ]


def plan_windows(
    chunks: Sequence[StagedChunk], capacity: int
) -> Iterator[Sequence[StagedChunk]]:
    """Split token-ordered chunks into runs whose packed layout fits ``capacity``.

    A chunk larger than ``capacity`` forms a window of its own; the caller must
    then grow its staging buffer to that window's ``num_bytes``.

    Args:
        chunks: Chunks in token order.
        capacity: Staging buffer size in bytes.

    Yields:
        Consecutive, non-empty slices of ``chunks``.
    """
    begin = 0
    used = 0
    for index, (memory_obj, _, _) in enumerate(chunks):
        size = memory_obj.get_size()
        if index > begin and _align(used) + size > capacity:
            yield chunks[begin:index]
            begin = index
            used = 0
        used = _align(used) + size
    if begin < len(chunks):
        yield chunks[begin:]


def pack(chunks: Sequence[StagedChunk], views: Sequence[MemoryObj]) -> None:
    """Copy each source chunk into its staging view on the current stream.

    Copies are enqueued without blocking the host; the caller must fence the
    stream before reusing the staging buffer or releasing the sources.

    Args:
        chunks: Source chunks, in the order used to build the window.
        views: The window's views over the staging buffer.

    Raises:
        ValueError: If a source or view no longer exposes its tensor.
    """
    for (memory_obj, _, _), view in zip(chunks, views, strict=True):
        source = memory_obj.raw_tensor
        target = view.raw_tensor
        if source is None or target is None:
            raise ValueError("KV retrieval requires valid tensor storage")
        target.copy_(source[: target.numel()], non_blocking=True)
