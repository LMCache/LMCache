# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for hardware-accelerated compression backends."""

# Standard
from collections.abc import Sequence
import abc


class AccelCompressBackend(abc.ABC):
    """ABC for accelerated compression backends.

    Implementations are thread-safe.
    """

    @abc.abstractmethod
    def compress(self, src: memoryview, dst: memoryview) -> int:
        """Compress src into dst.

        Args:
            src: Source data (read-only).
            dst: Pre-allocated destination buffer.

        Returns:
            Number of bytes written to dst.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, src: memoryview, dst: memoryview) -> int:
        """Decompress src into dst.

        Args:
            src: Compressed data (read-only).
            dst: Pre-allocated destination buffer.

        Returns:
            Number of bytes written to dst.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def max_compressed_length(self, src_size: int) -> int:
        """Return the worst-case compressed size for a given input size.

        Used to pre-allocate the destination buffer before compression.
        """
        raise NotImplementedError

    @property
    def max_block_size(self) -> int | None:
        """Largest input a single compress call accepts, or ``None`` if unlimited.

        The blocked container splits every group into blocks no larger than
        this, so that each block is exactly one codec call.
        """
        return None

    def compress_blocks(
        self, srcs: Sequence[memoryview], dsts: Sequence[memoryview]
    ) -> list[int]:
        """Compress a batch of blocks.

        Backends with a batch API should override this to submit all blocks in
        a single call; the default compresses them one at a time.

        Args:
            srcs: Source block views.
            dsts: Destination views, one per block, non-overlapping.

        Returns:
            The compressed size of each block, in order.
        """
        return [self.compress(s, d) for s, d in zip(srcs, dsts, strict=True)]

    def decompress_blocks(
        self, srcs: Sequence[memoryview], dsts: Sequence[memoryview]
    ) -> list[int]:
        """Decompress a batch of blocks.

        Args:
            srcs: Compressed block views.
            dsts: Destination views, one per block, non-overlapping.

        Returns:
            The decompressed size of each block, in order.
        """
        return [self.decompress(s, d) for s, d in zip(srcs, dsts, strict=True)]

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        raise NotImplementedError
