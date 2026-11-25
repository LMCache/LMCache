# SPDX-License-Identifier: Apache-2.0
"""Client half of the chunk-hash key form.

A client may name a prompt by the rolling blake3 hashes of its complete
chunks instead of its tokens (``IPCCacheServerKey.from_chunk_hashes``). The
hashes are computed with the server's own ``TokenHasher``, so they are
byte-identical to what a blake3 server would derive from the tokens.
"""

# Standard
from collections import OrderedDict
from collections.abc import Sequence
import functools
import threading

# First Party
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher


@functools.lru_cache(maxsize=8)
def _blake3_hasher(chunk_size: int) -> TokenHasher:
    return TokenHasher(chunk_size=chunk_size, hash_algorithm="blake3")


def lmcache_chunk_hashes(
    token_ids: Sequence[int],
    chunk_size: int,
    prev: bytes | None = None,
) -> bytes:
    """Chunk hashes of the complete chunks of ``token_ids``, concatenated.

    The rolling blake3 chain of a blake3 server's ``TokenHasher``, computed
    with that same code, so the result is byte-identical to what the server
    would derive from the tokens. A caller that keeps a prompt's chain (for
    example across conversation turns) can extend it chunk by chunk and pass
    it to an adapter that accepts caller hashes, instead of the tokens.

    Args:
        token_ids: Tokens starting at a chunk boundary; a trailing partial
            chunk is ignored.
        chunk_size: The server's chunk size.
        prev: Hash of the chunk before ``token_ids[0]`` when continuing a
            chain, or ``None`` when ``token_ids`` starts at chunk 0.

    Returns:
        ``CHUNK_HASH_BYTES`` bytes per complete chunk.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk size must be positive, got {chunk_size}")
    num_chunks = len(token_ids) // chunk_size
    if num_chunks == 0:
        return b""
    packed = pack_token_ids(token_ids[: num_chunks * chunk_size])
    return b"".join(
        _blake3_hasher(chunk_size).compute_packed_chunk_hashes(packed, prev)
    )


class ChunkHashCache:
    """Per-request rolling chunk hashes, extended as the prompt grows.

    Keys built from these hashes spare the server decoding and hashing the
    token prefix. A request's prompt only grows, so only the new chunks are
    hashed; if the last hashed chunk no longer matches, the chain restarts.
    Requests are evicted least recently used.
    """

    def __init__(self, chunk_size: int, max_requests: int = 1024) -> None:
        self._hasher = TokenHasher(chunk_size=chunk_size, hash_algorithm="blake3")
        self._chunk_size = chunk_size
        self._max_requests = max_requests
        # request_id -> (chunk hashes from chunk 0, packed last hashed chunk)
        self._entries: OrderedDict[str, tuple[list[bytes], bytes]] = OrderedDict()
        self._lock = threading.Lock()

    def hashes(
        self, request_id: str, token_ids: Sequence[int], end: int
    ) -> list[bytes]:
        """Hashes of the complete chunks in ``[0, min(end, len(token_ids)))``.

        Args:
            request_id: Request the tokens belong to.
            token_ids: The request's prompt from position 0.
            end: End token index of the key being built.

        Returns:
            One hash per complete chunk, from chunk 0.
        """
        chunk_size = self._chunk_size
        num_chunks = min(end, len(token_ids)) // chunk_size
        with self._lock:
            hashes, tail = self._entries.pop(request_id, ([], b""))
        have = len(hashes)
        if have and (
            have * chunk_size > len(token_ids)
            or pack_token_ids(token_ids[(have - 1) * chunk_size : have * chunk_size])
            != tail
        ):
            hashes, have = [], 0
        if have < num_chunks:
            packed = pack_token_ids(
                token_ids[have * chunk_size : num_chunks * chunk_size]
            )
            prefix = hashes[-1] if hashes else None
            hashes.extend(self._hasher.compute_packed_chunk_hashes(packed, prefix))
            tail = packed[-(len(packed) // (num_chunks - have)) :]
        with self._lock:
            self._entries[request_id] = (hashes, tail)
            while len(self._entries) > self._max_requests:
                self._entries.popitem(last=False)
        return hashes[:num_chunks]

    def drop(self, request_id: str) -> None:
        """Forget ``request_id``'s hashes."""
        with self._lock:
            self._entries.pop(request_id, None)
