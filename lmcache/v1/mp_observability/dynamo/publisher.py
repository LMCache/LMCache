# SPDX-License-Identifier: Apache-2.0

"""Coordinator that turns LMCache store/evict activity into Dynamo KV events.

This class glues the three single-purpose pieces together:

* :func:`lmcache.v1.mp_observability.dynamo.block_builder.build_blocks` derives
  per-block Dynamo hashes from a token sequence.
* :class:`lmcache.v1.mp_observability.dynamo.chunk_block_registry.ChunkBlockRegistry`
  remembers, per ``chunk_hash``, which block hashes were emitted so they can be
  removed on eviction (eviction only carries ``chunk_hash``, never tokens).
* :class:`lmcache.v1.mp_observability.dynamo.emitter.DynamoKvEventEmitter`
  publishes the resulting ``BlockStored`` / ``BlockRemoved`` events over ZMQ.

It performs no ZMQ, hashing, or wire-format work of its own -- only the
store->blocks->emit and evict->lookup->emit orchestration. Wiring it into the
storage path and the EventBus evict callback lives in
:mod:`lmcache.v1.mp_observability.dynamo.setup`.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.dynamo.block_builder import build_blocks
from lmcache.v1.mp_observability.dynamo.chunk_block_registry import (
    ChunkBlockRegistry,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.dynamo.emitter import DynamoKvEventEmitter
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class DynamoKvPublisher:
    """Translates store/evict events into Dynamo KV cache events.

    On store: recompute the stored range's block hashes, record them under
    their chunk hashes, emit one ``BlockStored``. On evict: look up the chunk
    hashes, remove them, emit one ``BlockRemoved``.

    The registry and emitter are each thread-safe, but this class adds no lock
    across steps, so a concurrent ``on_store`` / ``on_evict`` for the same
    chunk could interleave. The cache orders store before evict for a given
    chunk, so this is acceptable for v1.
    """

    def __init__(
        self,
        emitter: DynamoKvEventEmitter,
        hasher: TokenHasher,
        kv_block_size: int,
        chunk_size: int,
    ) -> None:
        """Create the coordinator.

        Args:
            emitter: Publisher for the resulting KV events.
            hasher: Hasher used to derive block hashes (must be the same
                algorithm the storage path uses, so hashes line up).
            kv_block_size: Tokens per Dynamo block.
            chunk_size: Tokens per LMCache chunk. Must be a positive multiple
                of ``kv_block_size`` so each chunk maps to a whole number of
                blocks; otherwise chunk-to-block grouping would be misaligned.

        Raises:
            ValueError: If ``kv_block_size`` or ``chunk_size`` is not positive,
                or ``chunk_size`` is not a multiple of ``kv_block_size``.
        """
        if kv_block_size <= 0:
            raise ValueError(f"kv_block_size must be positive (got {kv_block_size})")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive (got {chunk_size})")
        if chunk_size % kv_block_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be a multiple of "
                f"kv_block_size ({kv_block_size}) so each chunk maps to a "
                "whole number of blocks"
            )
        self._emitter = emitter
        self._hasher = hasher
        self._kv_block_size = kv_block_size
        self._chunk_size = chunk_size
        self._registry = ChunkBlockRegistry()

    def on_store(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        chunk_hashes: list[bytes],
    ) -> None:
        """Record and publish the blocks newly stored in ``[start, end)``.

        Args:
            token_ids: Tokens from the sequence start through ``end`` (the full
                prefix). Only ``token_ids[:end]`` is used.
            start: Start offset (inclusive) of the newly stored range. Assumed
                to fall on a ``kv_block_size`` boundary (chunk alignment
                guarantees this).
            end: End offset (exclusive) of the newly stored range.
            chunk_hashes: ``chunk_hash`` of each chunk covering ``[start, end)``
                in order. Used as the registry keys for later eviction.

        Note:
             The full prefix is rebuilt on every call (O(sequence length) per
            store); a future optimization could carry per-sequence rolling
            state. A ``start`` not aligned to ``kv_block_size`` is skipped with
            a warning rather than raised, so it never disturbs the store path.
        """
        if start % self._kv_block_size != 0:
            logger.warning(
                "Dynamo KV store skipped: start (%d) is not a multiple of "
                "kv_block_size (%d)",
                start,
                self._kv_block_size,
            )
            return

        _, all_blocks = build_blocks(
            token_ids[:end],
            self._kv_block_size,
            self._hasher.none_hash,
            self._hasher,
        )

        first = start // self._kv_block_size
        new_blocks = all_blocks[first:]
        if not new_blocks:
            return

        parent = None if first == 0 else all_blocks[first - 1][1]

        # Record each chunk's block-hash segment for later eviction.
        blocks_per_chunk = self._chunk_size // self._kv_block_size
        for j, chunk_hash in enumerate(chunk_hashes):
            seg = new_blocks[j * blocks_per_chunk : (j + 1) * blocks_per_chunk]
            if seg:
                self._registry.record(chunk_hash, [h for _, h in seg])

        flat_tokens = [t for toks, _ in new_blocks for t in toks]
        hashes = [h for _, h in new_blocks]
        self._emitter.publish_stored(flat_tokens, hashes, parent, self._kv_block_size)

    def on_evict(self, chunk_hashes: list[bytes]) -> None:
        """Publish a ``BlockRemoved`` for the blocks of evicted chunks.

        Chunk hashes with no recorded blocks are skipped silently (an evict
        for a chunk that was never published is a no-op).

        Args:
            chunk_hashes: ``chunk_hash`` of each evicted chunk.
        """
        removed: list[int] = []
        for chunk_hash in chunk_hashes:
            block_hashes = self._registry.pop(chunk_hash)
            if block_hashes:
                removed.extend(block_hashes)
        if removed:
            self._emitter.publish_removed(removed)

    def close(self) -> None:
        """Close the underlying emitter."""
        self._emitter.close()
