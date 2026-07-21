# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for hardware-accelerated compression backends."""

import abc


class AccelCompressBackend(abc.ABC):
    """ABC for accelerated compression backends (QAT, IAA, etc.).

    Implementations are thread-safe: the underlying C library manages
    per-thread sessions internally.
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

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        raise NotImplementedError
