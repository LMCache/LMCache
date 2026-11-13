# SPDX-License-Identifier: Apache-2.0
"""The chunk hasher LMCache hands to ATOM's scheduler.

The external-connector API keeps the chunk-hash chain on the scheduler: ATOM
asks the backend for a hasher, computes each chunk once, caches it on the
sequence and sends the slice it is transferring. The hasher therefore has to
name its scheme and its chunk size, because the LMCache server checks the scheme
and rejects a mismatch instead of storing keys no reader can find.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

# First Party
from lmcache.v1.common.token_hasher import TokenHasher

DEFAULT_HASH_ALGORITHM = "blake3"


@runtime_checkable
class ChunkHasher(Protocol):
    """Chunk-hash chain the backend provides to the scheduler."""

    scheme: str
    """Hash scheme identifier the server compares against its own."""

    chunk_tokens: int
    """Tokens per chunk this hasher expects."""

    def chain(
        self,
        tokens: Sequence[int],
        start_chunk: int,
        prev: bytes | None,
    ) -> bytes:
        """Return the hash of one chunk, chained on the previous chunk's hash."""
        ...


class TokenChainHasher:
    """LMCache's chunk hasher, over the same hashing the server uses.

    Args:
        chunk_size: Tokens per chunk; must match the server's.
        hash_algorithm: Hash function name; must match the server's.
    """

    def __init__(
        self, chunk_size: int = 256, hash_algorithm: str = DEFAULT_HASH_ALGORITHM
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._hasher = TokenHasher(chunk_size=chunk_size, hash_algorithm=hash_algorithm)
        self.chunk_tokens = chunk_size
        self.scheme = f"lmcache-{hash_algorithm}-{chunk_size}-v1"

    def chain(
        self,
        tokens: Sequence[int],
        start_chunk: int = 0,
        prev: bytes | None = None,
    ) -> bytes:
        """Return the hash of one chunk, chained on ``prev``.

        Args:
            tokens: The chunk's token ids, in order.
            start_chunk: Chunk index this hash belongs to; informational, the
                roll comes from ``prev``.
            prev: Previous chunk's hash, or ``None`` for the first chunk.

        Returns:
            The 32-byte chunk hash.

        Raises:
            ValueError: If the chunk does not hold exactly ``chunk_tokens``
                tokens, which would make the chain disagree with the server.
        """
        if len(tokens) != self.chunk_tokens:
            raise ValueError(
                f"chunk has {len(tokens)} tokens, expected {self.chunk_tokens}"
            )
        del start_chunk
        return TokenHasher.hash_to_bytes(self._hasher.hash_tokens(list(tokens), prev))

    def stats(self) -> dict[str, Any]:
        """Return the hasher's identity, for the backend's stats report."""
        return {
            "scheme": self.scheme,
            "chunk_tokens": self.chunk_tokens,
            "hash_algorithm": self._hasher.hash_algorithm_name,
        }


__all__ = ["ChunkHasher", "TokenChainHasher"]
