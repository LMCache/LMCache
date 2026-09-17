# SPDX-License-Identifier: Apache-2.0
"""A simulated LMCache server + worker used as the correctness oracle.

Two pieces:

``FakeSchedulerAdapter`` stands in for ``LMCacheMPSchedulerAdapter``.  It
answers lookups from the ``CacheModel`` contents and keeps a per-request
read-lock ledger that mirrors the server's prefetch locks, so the tests can
assert every lock is released exactly once.

``CacheModel`` is the "server + worker + GPU memory" model.  It tracks what
token content each GPU block holds, applies STORE / RETRIEVE ops emitted by
the connector, and marks a stored chunk POISONED when one of its blocks was
rewritten with different tokens before the store completed under the
configured completion rule.  A RETRIEVE of a poisoned or missing chunk is a
test failure.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)

ChunkKey = tuple[str, tuple[int, ...]]


class CompletionPolicy(str, Enum):
    """When an in-flight STORE has finished reading its GPU blocks."""

    # Store completes inside the step that submitted it (engine-driven sync).
    SYNC = "sync"
    # Store completes ``latency`` steps later, but any step whose metadata
    # carries ``need_flush_before_forward`` completes all in-flight stores
    # before that step's forward pass (engine-driven async with flush, or
    # lmcache-driven once handle_preemptions waits on outstanding stores).
    FLUSH_ON_FLAG = "flush_on_flag"
    # Store completes ``latency`` steps later and nothing on the vLLM side
    # ever waits for it (lmcache-driven path before the H1 fix).
    SERVER_ASYNC = "server_async"


@dataclass
class InflightStore:
    request_id: str
    submitted_step: int
    complete_step: int
    keys: list[ChunkKey]
    # block_id -> expected token content of that block at store time.
    expected: dict[int, tuple[int, ...]]
    poisoned: bool = False


@dataclass
class InflightRetrieve:
    request_id: str
    submitted_step: int
    complete_step: int
    keys: list[ChunkKey]
    block_ids: list[int]
    failed: bool
    written: dict[int, tuple[int, ...]]


@dataclass
class LockLedger:
    """Read locks per chunk key, and which request holds which key."""

    counts: dict[ChunkKey, int] = field(default_factory=dict)
    held: dict[str, set[ChunkKey]] = field(default_factory=dict)
    history: list[tuple[str, str, ChunkKey]] = field(default_factory=list)

    def acquire(self, request_id: str, key: ChunkKey) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        self.held.setdefault(request_id, set()).add(key)
        self.history.append(("acquire", request_id, key))

    def release(self, request_id: str, key: ChunkKey) -> None:
        held = self.held.get(request_id, set())
        assert key in held, (
            f"request {request_id} released lock it does not hold: {key[1][-4:]}"
        )
        held.discard(key)
        self.counts[key] -= 1
        assert self.counts[key] >= 0
        self.history.append(("release", request_id, key))

    def outstanding(self) -> dict[str, set[ChunkKey]]:
        return {r: k for r, k in self.held.items() if k}


def chunk_keys(
    token_ids: list[int], start: int, end: int, chunk: int, salt: str
) -> list[ChunkKey]:
    """Prefix keys for chunks covering tokens ``[start, end)``.

    LMCache keys a chunk by the hash of every token up to the chunk's end, so
    the model uses the token prefix itself as the key.
    """
    assert start % chunk == 0 and end % chunk == 0, (start, end, chunk)
    return [
        (salt, tuple(token_ids[: (c + 1) * chunk]))
        for c in range(start // chunk, end // chunk)
    ]


class CacheModel:
    def __init__(
        self,
        *,
        chunk_size: int,
        block_size: int,
        completion_policy: CompletionPolicy,
        store_latency: int = 1,
        load_latency: int = 1,
        lookup_defers: int = 0,
        evict_before_retrieve: Callable[[ChunkKey], bool] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.completion_policy = completion_policy
        self.store_latency = store_latency
        self.load_latency = load_latency
        self.lookup_defers = lookup_defers
        self.evict_before_retrieve = evict_before_retrieve

        # key -> token content of the chunk, or None when poisoned.
        self.chunks: dict[ChunkKey, tuple[int, ...] | None] = {}
        self.poisoned_keys: set[ChunkKey] = set()
        # GPU block_id -> token content currently written in it.
        self.block_content: dict[int, tuple[int, ...]] = {}
        self.inflight_stores: list[InflightStore] = []
        self.inflight_retrieves: list[InflightRetrieve] = []
        self.locks = LockLedger()
        self.step = 0

        # Observability for assertions.
        self.store_ops: list[tuple[int, str, int, int]] = []
        self.retrieve_ops: list[tuple[int, str, int, int]] = []
        self.poisoned_stores: list[InflightStore] = []
        self.poisoned_retrieves: list[InflightRetrieve] = []
        self.missing_retrieves: list[InflightRetrieve] = []
        self.num_stored_chunks = 0

    # ------------------------------------------------------------------
    # Lookup (server side)
    # ------------------------------------------------------------------
    def lookup_hit_tokens(self, token_ids: list[int], salt: str) -> int:
        """Longest prefix of full chunks present (and not poisoned)."""
        hit = 0
        for c in range(len(token_ids) // self.chunk_size):
            key = (salt, tuple(token_ids[: (c + 1) * self.chunk_size]))
            if key not in self.chunks or self.chunks[key] is None:
                break
            hit += 1
        return hit * self.chunk_size

    # ------------------------------------------------------------------
    # Step bookkeeping (worker + GPU side)
    # ------------------------------------------------------------------
    def begin_step(self, step: int) -> None:
        self.step = step

    def apply_forward_writes(
        self,
        writes: dict[int, tuple[int, ...]],
        need_flush_before_forward: bool,
    ) -> None:
        """Model the forward pass of the current step writing ``writes``.

        ``writes`` maps block_id -> the token content the forward pass puts in
        that block this step.  Under ``FLUSH_ON_FLAG`` with the flag set, all
        in-flight stores complete before the writes land.
        """
        if (
            self.completion_policy == CompletionPolicy.FLUSH_ON_FLAG
            and need_flush_before_forward
        ):
            for s in self.inflight_stores:
                s.complete_step = min(s.complete_step, self.step)
            self._complete_stores(before_forward=True)

        for block_id, content in writes.items():
            old = self.block_content.get(block_id)
            self.block_content[block_id] = content
            if old == content:
                continue
            for s in self.inflight_stores:
                if block_id in s.expected and s.expected[block_id] != content:
                    s.poisoned = True

    def submit_ops(
        self,
        metadata: LMCacheMPConnectorMetadata,
        request_tokens: dict[str, list[int]],
    ) -> None:
        """Register this step's STORE / RETRIEVE ops (emitted after forward)."""
        for r in metadata.requests:
            assert isinstance(r, LMCacheMPRequestMetadata)
            op = r.op
            token_ids = list(op.token_ids)
            salt = r.cache_salt or ""
            if r.direction == "STORE":
                self.store_ops.append((self.step, r.request_id, op.start, op.end))
                keys = chunk_keys(token_ids, op.start, op.end, self.chunk_size, salt)
                expected: dict[int, tuple[int, ...]] = {}
                group0 = op.block_ids[0] if op.block_ids else []
                for i, block_id in enumerate(group0):
                    lo = op.start + i * self.block_size
                    expected[block_id] = tuple(token_ids[lo : lo + self.block_size])
                    # The store reads whatever the block holds *now*.
                    assert self.block_content.get(block_id) == expected[block_id], (
                        f"STORE {r.request_id} [{op.start},{op.end}) reads block "
                        f"{block_id} expecting tokens {expected[block_id][:3]}... but "
                        f"the block holds {self.block_content.get(block_id, ())[:3]}..."
                    )
                if self.completion_policy == CompletionPolicy.SYNC:
                    complete = self.step
                else:
                    complete = self.step + self.store_latency
                self.inflight_stores.append(
                    InflightStore(
                        request_id=r.request_id,
                        submitted_step=self.step,
                        complete_step=complete,
                        keys=keys,
                        expected=expected,
                    )
                )
            else:
                self.retrieve_ops.append((self.step, r.request_id, op.start, op.end))
                keys = chunk_keys(token_ids, op.start, op.end, self.chunk_size, salt)
                failed = False
                for key in keys:
                    if self.evict_before_retrieve is not None and (
                        self.evict_before_retrieve(key)
                    ):
                        self.chunks.pop(key, None)
                    if key not in self.chunks:
                        failed = True
                    elif self.chunks[key] is None:
                        failed = True
                group0 = op.block_ids[0] if op.block_ids else []
                written: dict[int, tuple[int, ...]] = {}
                skip = op.skip_first_n_tokens
                for i, block_id in enumerate(group0):
                    lo = op.start + i * self.block_size
                    if lo + self.block_size <= op.start + skip:
                        continue  # APC-shared block, server must not write it
                    written[block_id] = tuple(token_ids[lo : lo + self.block_size])
                rec = InflightRetrieve(
                    request_id=r.request_id,
                    submitted_step=self.step,
                    complete_step=self.step + self.load_latency,
                    keys=keys,
                    block_ids=list(group0),
                    failed=failed,
                    written=written,
                )
                if failed:
                    if any(k in self.poisoned_keys for k in keys):
                        self.poisoned_retrieves.append(rec)
                    else:
                        self.missing_retrieves.append(rec)
                self.inflight_retrieves.append(rec)
                # The server releases the prefetch read locks as it serves
                # the retrieve (or fails it).
                for key in keys:
                    if key in self.locks.held.get(r.request_id, set()):
                        self.locks.release(r.request_id, key)

    def end_step(self) -> tuple[set[str], set[int]]:
        """Complete stores/retrieves due this step.

        Returns (finished_recving request ids, invalid block ids).
        """
        self._complete_stores(before_forward=False)
        finished_recving: set[str] = set()
        invalid: set[int] = set()
        still = []
        for rr in self.inflight_retrieves:
            if rr.complete_step > self.step:
                still.append(rr)
                continue
            finished_recving.add(rr.request_id)
            if rr.failed:
                invalid.update(rr.block_ids)
            else:
                for block_id, content in rr.written.items():
                    self.block_content[block_id] = content
        self.inflight_retrieves = still
        return finished_recving, invalid

    def _complete_stores(self, *, before_forward: bool) -> None:
        still = []
        for s in self.inflight_stores:
            if s.complete_step > self.step:
                still.append(s)
                continue
            if s.poisoned:
                self.poisoned_stores.append(s)
                for key in s.keys:
                    self.chunks[key] = None
                    self.poisoned_keys.add(key)
            else:
                for key in s.keys:
                    if key not in self.chunks:
                        self.num_stored_chunks += 1
                    self.chunks[key] = key[1][-self.chunk_size :]
        self.inflight_stores = still

    def has_inflight_stores_for(self, request_id: str) -> bool:
        return any(s.request_id == request_id for s in self.inflight_stores)

    def drain(self) -> None:
        """Force-complete everything (end of simulation)."""
        for s in self.inflight_stores:
            s.complete_step = self.step
        self._complete_stores(before_forward=False)


class FakeSchedulerAdapter:
    """Stand-in for ``LMCacheMPSchedulerAdapter`` backed by a ``CacheModel``."""

    def __init__(self, model: CacheModel) -> None:
        self.model = model
        self.lmcache_tokens_per_chunk = model.chunk_size
        self.is_healthy = True
        self._pending: dict[str, tuple[list[int], str, int]] = {}
        self._finished: dict[str, int] = {}
        self.lookups_submitted: list[tuple[str, int]] = []
        self.end_session_calls: dict[str, int] = {}
        self.free_lock_calls: list[tuple[str, int, int]] = []
        self.allocation_reports: int = 0
        self.pending_store_counts: dict[str, int] = {}

    # -- constructor shim: the connector calls the class with server args ----
    def __call__(self, *args: Any, **kwargs: Any) -> "FakeSchedulerAdapter":
        return self

    # -- lookup ---------------------------------------------------------------
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        if request_id in self._pending or request_id in self._finished:
            return
        self._pending[request_id] = (list(token_ids), cache_salt or "", 0)
        self.lookups_submitted.append((request_id, len(token_ids)))

    def check_lookup_result(self, request_id: str) -> int | None:
        if request_id in self._finished:
            return self._finished[request_id]
        token_ids, salt, polls = self._pending[request_id]
        if polls < self.model.lookup_defers:
            self._pending[request_id] = (token_ids, salt, polls + 1)
            return None
        hit = self.model.lookup_hit_tokens(token_ids, salt)
        # The server read-locks every hit chunk until it is freed or served.
        for key in chunk_keys(token_ids, 0, hit, self.model.chunk_size, salt):
            self.model.locks.acquire(request_id, key)
        self._finished[request_id] = hit
        del self._pending[request_id]
        return hit

    def cleanup_lookup_result(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        self._finished.pop(request_id, None)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        self.free_lock_calls.append((request_id, start, end))
        chunk = self.model.chunk_size
        # Chunk containing ``start`` is freed; the one containing ``end`` is
        # not (floor), matching the real adapter's documented behaviour.
        for c in range(start // chunk, end // chunk):
            key = (cache_salt or "", tuple(token_ids[: (c + 1) * chunk]))
            if key in self.model.locks.held.get(request_id, set()):
                self.model.locks.release(request_id, key)

    # -- session --------------------------------------------------------------
    def end_session(self, request_id: str) -> None:
        self.end_session_calls[request_id] = (
            self.end_session_calls.get(request_id, 0) + 1
        )

    def report_block_allocations(self, records: list[Any]) -> None:
        self.allocation_reports += len(records)

    def update_pending_store_count(self, request_id: str, count: int) -> bool:
        self.pending_store_counts[request_id] = count
        return True

    def shutdown(self) -> None:
        pass
