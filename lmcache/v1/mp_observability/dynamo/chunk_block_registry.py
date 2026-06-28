# SPDX-License-Identifier: Apache-2.0

"""Thread-safe chunk_hash -> block_hashes registry for Dynamo KV events.

At store time the publisher knows a chunk's ``token_ids`` and can derive its
Dynamo block hashes; at evict time only the ``chunk_hash`` is available and
the block hashes can no longer be recomputed. This registry bridges the gap:
``record`` stores the mapping on store, and ``pop`` retrieves-and-removes it
on evict so a ``BlockRemoved`` event can be emitted.

The store path and the evict path run on different threads (eviction is
dispatched from the EventBus drain thread), so all access is guarded by a
single lock.
"""

# Future
from __future__ import annotations

# Standard
import threading


class ChunkBlockRegistry:
    """Thread-safe map from ``chunk_hash`` to its Dynamo block hashes.

    All reads and writes are serialized by an internal lock, making the
    registry safe to share between the store path and the EventBus-driven
    evict path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mapping: dict[bytes, list[int]] = {}

    def record(self, chunk_hash: bytes, block_hashes: list[int]) -> None:
        """Store the block hashes for a chunk, overwriting any prior entry.

        Args:
            chunk_hash: Content hash identifying the chunk.
            block_hashes: Dynamo block hashes derived for this chunk.
        """
        with self._lock:
            self._mapping[chunk_hash] = block_hashes

    def pop(self, chunk_hash: bytes) -> list[int] | None:
        """Remove and return the block hashes recorded for a chunk.

        Args:
            chunk_hash: Content hash identifying the chunk.

        Returns:
            The recorded block hashes, or ``None`` if the chunk was never
            recorded (evicting an unrecorded chunk is a no-op upstream).
        """
        with self._lock:
            return self._mapping.pop(chunk_hash, None)

    def __len__(self) -> int:
        """Return the number of currently recorded chunks."""
        with self._lock:
            return len(self._mapping)
