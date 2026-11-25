# SPDX-License-Identifier: Apache-2.0
"""Request id → token sequence and chunk keys, for the MP coordinator.

Built from the ``request_ids`` and ``parent_hash_hex`` the MP servers
stamp on ``STORE`` entries. Keeps each chunk's tokens after its KV is
evicted, so a request's tokens stay recoverable; where the KV lives now
is the key directory's answer, asked with the keys this index returns.

See ``docs/design/v1/mp_coordinator/views/request_index.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.api import (
    BlendNamespace,
    CacheEventBatch,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.views.base import View

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry

DEFAULT_MAX_CHUNKS = 65536
"""Default bound on retained chunks: about 64 MiB of token content at 256
tokens per chunk."""

MAX_CHUNKS_CONFIG_KEY = "request_index_max_chunks"
"""``extra_config`` key overriding :data:`DEFAULT_MAX_CHUNKS`."""

DEFAULT_MAX_REQUESTS = 1048576
"""Default bound on tracked requests. A request costs a few hundred bytes,
far less than a chunk, and every request with a hit takes a slot, so this
bound is larger: about 17 minutes of requests at 1000 per second."""

MAX_REQUESTS_CONFIG_KEY = "request_index_max_requests"
"""``extra_config`` key overriding :data:`DEFAULT_MAX_REQUESTS`."""


@dataclass(frozen=True)
class RequestChunk:
    """One recovered chunk of a request.

    Attributes:
        chunk_hash: The chunk's ``ObjectKey.chunk_hash``.
        token_offset: Position of the chunk's first token in the request.
        token_ids: The chunk's token ids.
        keys: The keys the chunk was stored under in the request's own
            namespace, sorted by ``(kv_rank, object_group_id)``. Empty when
            only another namespace stored it.
    """

    chunk_hash: bytes
    token_offset: int
    token_ids: tuple[int, ...]
    keys: tuple[ObjectKey, ...]


@dataclass(frozen=True)
class RequestTrace:
    """A request's recovered chunks.

    Attributes:
        chunks: The chunks in sequence order, ending at the request's
            highest-offset chunk, stored or hit.
        complete: ``True`` when ``chunks`` reaches back to position 0. It
            says the chain is unbroken, not that the whole prompt came
            back: a request that hit part of its prompt and stored nothing
            returns only the hit prefix.
    """

    chunks: tuple[RequestChunk, ...]
    complete: bool

    @property
    def start_offset(self) -> int:
        """Position of the first recovered token."""
        return self.chunks[0].token_offset

    @property
    def token_ids(self) -> list[int]:
        """The recovered tokens, concatenated in sequence order."""
        return [token for chunk in self.chunks for token in chunk.token_ids]


@dataclass(frozen=True)
class RequestRef:
    """A request whose sequence contains a given chunk.

    Attributes:
        instance_id: The MP server that reported the request.
        request_id: The request id on that server.
        namespace: The model, salt, world size and tags it ran under.
    """

    instance_id: str
    request_id: str
    namespace: BlendNamespace


@dataclass
class _ChunkRecord:
    """Retained content of one chunk hash, outliving its placements, and
    the retained chunks stored after it in some sequence."""

    token_ids: tuple[int, ...]
    token_offset: int
    parent_hash: bytes | None
    keys: set[ObjectKey] = field(default_factory=set)
    children: set[bytes] = field(default_factory=set)


@dataclass(frozen=True)
class _Tail:
    """A request's highest-offset chunk, stored or hit, and the namespace whose
    keys belong to the request: chunk hashes depend on tokens alone, so
    one chunk can hold several tenants' keys."""

    chunk_hash: bytes
    token_offset: int
    namespace: BlendNamespace


class RequestIndex(View):
    """Thread-safe, bounded index from ``(instance_id, request_id)`` to
    the request's chunks.

    A request is recorded when it stores a chunk or when its lookup hits
    one, so a request served entirely from cache is recorded too. The
    chunk table is LRU-bounded to ``max_chunks`` entries and the request
    table to ``max_requests``, by recency of store or hit. Storing or
    hitting a chunk also refreshes its ancestors, so a prefix many requests
    reach outlives
    unrelated chunks. Evicting a chunk's KV does not remove it; only the
    bound does.

    The reverse question, which requests contain a chunk, is answered by
    walking child links down from the chunk to the tails below it, so it
    costs no memory beyond the two bounded tables.
    """

    def __init__(
        self,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ) -> None:
        """Create an empty index.

        Args:
            max_chunks: Bound on retained chunks.
            max_requests: Bound on tracked requests.

        Raises:
            ValueError: If ``max_chunks`` or ``max_requests`` < 1.
        """
        if max_chunks < 1:
            raise ValueError(f"max_chunks must be >= 1 (got {max_chunks})")
        if max_requests < 1:
            raise ValueError(f"max_requests must be >= 1 (got {max_requests})")
        self._max_chunks = max_chunks
        self._max_requests = max_requests
        self._lock = threading.Lock()
        self._chunks: OrderedDict[bytes, _ChunkRecord] = OrderedDict()
        self._tails: OrderedDict[tuple[str, str], _Tail] = OrderedDict()
        # tail chunk hash -> requests whose tail it is; the inverse of
        # ``_tails``, for walking down from a chunk to its requests.
        self._requests_by_tail: dict[bytes, set[tuple[str, str]]] = {}

    @classmethod
    def from_config(
        cls, config: "MPCoordinatorConfig", views: "Registry[View]"
    ) -> "RequestIndex":
        """Build the index, bounded by ``extra_config`` keys
        :data:`MAX_CHUNKS_CONFIG_KEY` and :data:`MAX_REQUESTS_CONFIG_KEY`
        when set.

        Args:
            config: The coordinator configuration.
            views: Unused; the index depends on no other view.

        Raises:
            ValueError: If a configured bound is not a positive integer.
        """
        bounds = {}
        for config_key, default in (
            (MAX_CHUNKS_CONFIG_KEY, DEFAULT_MAX_CHUNKS),
            (MAX_REQUESTS_CONFIG_KEY, DEFAULT_MAX_REQUESTS),
        ):
            bound = config.extra_config.get(config_key, default)
            if not isinstance(bound, int) or isinstance(bound, bool):
                raise ValueError(f"{config_key} must be an integer (got {bound!r})")
            bounds[config_key] = bound
        return cls(
            max_chunks=bounds[MAX_CHUNKS_CONFIG_KEY],
            max_requests=bounds[MAX_REQUESTS_CONFIG_KEY],
        )

    def consume(self, batch: CacheEventBatch) -> None:
        """Record the chunks and requests a ``STORE`` batch reports, and
        the requests any batch's ``lookups`` report.

        Store entries without token ids carry no content to retain and are
        skipped.

        Args:
            batch: The admitted batch.
        """
        is_store = batch.event_type == CacheEventType.STORE
        if not is_store and not batch.lookups:
            return
        with self._lock:
            # One chunk arrives once per rank and object group; refresh its
            # chain once per batch.
            refreshed: set[bytes] = set()
            for entry in batch.entries if is_store else ():
                if not entry.token_ids:
                    continue
                key = entry.key.to_object_key()
                record = self._chunks.get(key.chunk_hash)
                if record is None:
                    record = _ChunkRecord(
                        token_ids=tuple(entry.token_ids),
                        token_offset=entry.token_offset,
                        parent_hash=bytes.fromhex(entry.parent_hash_hex)
                        if entry.parent_hash_hex
                        else None,
                    )
                    self._chunks[key.chunk_hash] = record
                    if record.parent_hash is not None:
                        parent = self._chunks.get(record.parent_hash)
                        if parent is not None:
                            parent.children.add(key.chunk_hash)
                record.keys.add(key)
                if key.chunk_hash not in refreshed:
                    self._refresh_chain(key.chunk_hash, record, refreshed)
                if entry.request_ids:
                    namespace = BlendNamespace.from_object_key(key)
                    for request_id in entry.request_ids:
                        self._record_tail(
                            (batch.instance_id, request_id),
                            _Tail(key.chunk_hash, entry.token_offset, namespace),
                        )
            for lookup in batch.lookups:
                key = lookup.key.to_object_key()
                # A hit on a chunk the index lacks (engine-covered, or aged
                # out) still records the request; it resolves once the
                # chunk is stored and answers 404 until then.
                record = self._chunks.get(key.chunk_hash)
                if record is not None and key.chunk_hash not in refreshed:
                    self._refresh_chain(key.chunk_hash, record, refreshed)
                self._record_tail(
                    (batch.instance_id, lookup.request_id),
                    _Tail(
                        key.chunk_hash,
                        lookup.token_offset,
                        BlendNamespace.from_object_key(key),
                    ),
                )
            self._enforce_bounds()

    def fence_instance(self, instance_id: str) -> None:
        """Keep everything: tokens outlive the placements a restart voids.

        Args:
            instance_id: The restarted or departed instance.
        """

    def get_request(self, instance_id: str, request_id: str) -> RequestTrace | None:
        """Recover a request's chunks by walking parent links back from
        its highest-offset chunk.

        Args:
            instance_id: The MP server that reported the request.
            request_id: The request id on that server.

        Returns:
            The request's chunks, or ``None`` when the index holds no
            chunk of it (its tail chunk was never stored or has aged out).
        """
        with self._lock:
            tail = self._tails.get((instance_id, request_id))
            if tail is None:
                return None
            chunks: list[RequestChunk] = []
            seen: set[bytes] = set()
            chunk_hash: bytes | None = tail.chunk_hash
            complete = False
            # ``seen`` stops a corrupt parent cycle from looping forever.
            while chunk_hash is not None and chunk_hash not in seen:
                record = self._chunks.get(chunk_hash)
                if record is None:
                    break
                seen.add(chunk_hash)
                chunks.append(
                    RequestChunk(
                        chunk_hash=chunk_hash,
                        token_offset=record.token_offset,
                        token_ids=record.token_ids,
                        keys=tuple(
                            sorted(
                                (
                                    k
                                    for k in record.keys
                                    if BlendNamespace.from_object_key(k)
                                    == tail.namespace
                                ),
                                key=lambda k: (k.kv_rank, k.object_group_id),
                            )
                        ),
                    )
                )
                if record.token_offset == 0:
                    complete = True
                    break
                chunk_hash = record.parent_hash
            if not chunks:
                return None
            chunks.reverse()
            return RequestTrace(chunks=tuple(chunks), complete=complete)

    def get_chunk_requests(self, chunk_hash: bytes) -> list[RequestRef] | None:
        """List the tracked requests whose sequence contains ``chunk_hash``.

        A request's sequence contains every chunk on the parent chain of
        its highest-offset chunk, whether it stored the chunk or hit it.

        Args:
            chunk_hash: The chunk's ``ObjectKey.chunk_hash``.

        Returns:
            The requests sorted by ``(instance_id, request_id)``, or
            ``None`` when the index does not hold the chunk.
        """
        with self._lock:
            if chunk_hash not in self._chunks:
                return None
            requests: set[tuple[str, str]] = set()
            pending = [chunk_hash]
            seen = {chunk_hash}
            while pending:
                current = pending.pop()
                requests.update(self._requests_by_tail.get(current, ()))
                record = self._chunks.get(current)
                if record is None:
                    continue
                for child in record.children - seen:
                    seen.add(child)
                    pending.append(child)
            return [
                RequestRef(
                    instance_id=instance_id,
                    request_id=request_id,
                    namespace=self._tails[(instance_id, request_id)].namespace,
                )
                for instance_id, request_id in sorted(requests)
            ]

    def _refresh_chain(
        self, chunk_hash: bytes, record: _ChunkRecord, refreshed: set[bytes]
    ) -> None:
        """Mark ``chunk_hash`` and its retained ancestors most recent, root
        first, so the bound ages out a prefix only after the chunks built
        on it. Stops at an ancestor already refreshed in this batch, which
        also ends a corrupt parent cycle. Call under the lock."""
        chain = [chunk_hash]
        refreshed.add(chunk_hash)
        parent_hash = record.parent_hash
        while parent_hash is not None and parent_hash not in refreshed:
            parent = self._chunks.get(parent_hash)
            if parent is None:
                break
            chain.append(parent_hash)
            refreshed.add(parent_hash)
            parent_hash = parent.parent_hash
        for hash_ in reversed(chain):
            self._chunks.move_to_end(hash_)

    def _record_tail(self, request: tuple[str, str], tail: _Tail) -> None:
        """Keep ``tail`` as ``request``'s tail unless a higher-offset chunk
        already is. Call under the lock."""
        current = self._tails.get(request)
        if current is None or tail.token_offset >= current.token_offset:
            if current is not None:
                self._forget_tail(request, current.chunk_hash)
            self._tails[request] = tail
            self._requests_by_tail.setdefault(tail.chunk_hash, set()).add(request)
        self._tails.move_to_end(request)

    def _forget_tail(self, request: tuple[str, str], chunk_hash: bytes) -> None:
        """Drop ``request`` from ``chunk_hash``'s tail set. Call under the
        lock."""
        requests = self._requests_by_tail[chunk_hash]
        requests.discard(request)
        if not requests:
            del self._requests_by_tail[chunk_hash]

    def _enforce_bounds(self) -> None:
        """Age out the least recent chunks and requests past the bound,
        unlinking them from the reverse indexes. Call under the lock."""
        while len(self._chunks) > self._max_chunks:
            chunk_hash, record = self._chunks.popitem(last=False)
            if record.parent_hash is not None:
                parent = self._chunks.get(record.parent_hash)
                if parent is not None:
                    parent.children.discard(chunk_hash)
        while len(self._tails) > self._max_requests:
            request, tail = self._tails.popitem(last=False)
            self._forget_tail(request, tail.chunk_hash)
