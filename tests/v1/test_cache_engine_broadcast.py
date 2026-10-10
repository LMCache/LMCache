# SPDX-License-Identifier: Apache-2.0
"""Exercise first-rank broadcast retrieval through LMCacheEngine on two ranks.

Both ranks run real engines in one process. A fake tensor-parallel group stands
in for the serving engine's collectives: rank 0 publishes, rank 1 mirrors, votes
are combined with MIN, and a barrier turns any rank that stops participating
into a test failure instead of a hang. The connector completes its page writes
at the fence, not at enqueue, so premature buffer reuse shows up as wrong data.
"""

# Standard
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any
from unittest.mock import MagicMock
import asyncio
import logging
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.broadcast_transfer import CHUNK_ALIGNMENT
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import TokenDatabase

# Local
from .test_broadcast_transfer import make_object

pytestmark = pytest.mark.no_shared_allocator

TIMEOUT = 20.0
CHUNK_TOKENS = 4
CHUNK_BYTES = 1 * 2 * CHUNK_TOKENS * 3 * 4
NUM_CHUNKS = 5
NUM_TOKENS = NUM_CHUNKS * CHUNK_TOKENS
TOKENS = list(range(NUM_TOKENS))
# Chunks start on aligned offsets, so two chunks need one alignment step plus one.
TWO_CHUNK_BUDGET = CHUNK_ALIGNMENT + CHUNK_BYTES

TwoRanks = Callable[..., tuple[list[LMCacheEngine], list["RecordingConnector"]]]


class FakeTensorParallelGroup:
    """In-process stand-in for a TP group: the source publishes, peers mirror."""

    def __init__(self, world_size: int) -> None:
        self._barrier = threading.Barrier(world_size)
        self._lock = threading.Lock()
        self._slot: Any = None
        self._votes: list[bool] = []
        self.payload_sizes: list[int] = []
        self.object_count = 0

    def broadcast_object(self, rank: int, obj: Any, src: int) -> Any:
        if rank == src:
            self._slot = obj
            self.object_count += 1
        self._barrier.wait(TIMEOUT)
        received = self._slot
        self._barrier.wait(TIMEOUT)
        return received

    def broadcast(self, rank: int, tensor: torch.Tensor, src: int) -> None:
        if rank == src:
            self._slot = tensor.clone()
            self.payload_sizes.append(tensor.numel())
        self._barrier.wait(TIMEOUT)
        if rank != src:
            tensor.copy_(self._slot)
        self._barrier.wait(TIMEOUT)

    def agree(self, rank: int, ready: bool) -> bool:
        with self._lock:
            self._votes.append(ready)
        self._barrier.wait(TIMEOUT)
        result = all(self._votes)
        self._barrier.wait(TIMEOUT)
        with self._lock:
            self._votes.clear()
        self._barrier.wait(TIMEOUT)
        return result


class RecordingConnector(MockGPUConnector):
    """Complete page writes at the fence, like a native writer with its own queue.

    ``batched_to_gpu`` only queues the views it was given; ``synchronize_load``
    performs the reads into ``destination``. An injected failure raises after
    queuing, as a native writer fails once part of its work is already in
    flight. Reads that run after the staging buffer was reused land wrong data.
    """

    def __init__(self, fail_on_write: int | None = None) -> None:
        super().__init__((2, 1, CHUNK_TOKENS, 1, 3))
        self.destination = torch.full((1, 2, NUM_TOKENS, 3), -1.0)
        self.fail_on_write = fail_on_write
        self.write_count = 0
        self.pending: list[tuple[MemoryObj, int, int]] = []

    def batched_to_gpu(
        self,
        memory_objs: Any = None,
        starts: Any = None,
        ends: Any = None,
        **kwargs: Any,
    ) -> None:
        """Queue the views, then raise if this is the injected failure."""
        self.write_count += 1
        self.pending.extend(zip(memory_objs, starts, ends, strict=True))
        if self.write_count == self.fail_on_write:
            raise torch.OutOfMemoryError("injected connector allocation failure")

    def synchronize_load(self) -> None:
        """Perform every queued read."""
        for obj, start, end in self.pending:
            assert obj.tensor is not None
            self.destination[:, :, start:end].copy_(obj.tensor)
        self.pending.clear()


def expected_kv() -> torch.Tensor:
    return torch.arange(2 * NUM_TOKENS * 3, dtype=torch.float32).reshape(
        1, 2, NUM_TOKENS, 3
    )


def make_sources(
    tokens_per_chunk: int = CHUNK_TOKENS,
) -> tuple[list[CacheEngineKey], list[MemoryObj]]:
    starts = range(0, NUM_TOKENS, tokens_per_chunk)
    keys = [
        CacheEngineKey("test", 1, 0, index, torch.float32)
        for index in range(len(starts))
    ]
    kv = expected_kv()
    sources = [
        make_object([kv[:, :, start : start + tokens_per_chunk]]) for start in starts
    ]
    return keys, sources


def make_engine(
    rank: int,
    group: FakeTensorParallelGroup,
    connector: MockGPUConnector,
    *,
    world_size: int = 2,
    budget: int = TWO_CHUNK_BUDGET,
    async_loading: bool = False,
    use_mla: bool = True,
) -> LMCacheEngine:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_TOKENS,
        retrieve_buffer_size=budget,
        enable_async_loading=async_loading,
        py_enable_gc=True,
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=rank,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 1, CHUNK_TOKENS, 1, 3),
        use_mla=use_mla,
    )
    return LMCacheEngine(
        config,
        metadata,
        MagicMock(spec=TokenDatabase),
        connector,
        partial(group.broadcast, rank),
        partial(group.broadcast_object, rank),
        partial(group.agree, rank),
    )


def attach_storage(
    engine: LMCacheEngine,
    keys: list[CacheEngineKey],
    sources: list[MemoryObj],
    *,
    missing: frozenset[int] = frozenset(),
    tokens_per_chunk: int = CHUNK_TOKENS,
) -> MagicMock:
    """Serve ``sources`` from a mocked storage layer on the source rank."""
    chunk_infos = [
        (key, index * tokens_per_chunk, min((index + 1) * tokens_per_chunk, NUM_TOKENS))
        for index, key in enumerate(keys)
    ]
    engine.token_database.process_tokens.side_effect = lambda **kwargs: iter(  # type: ignore[attr-defined]
        [(start, end, key) for key, start, end in chunk_infos]
    )
    storage = MagicMock(spec=StorageManager)
    storage.get_block_mapping.return_value = {"LocalCPUBackend": chunk_infos}
    storage.batched_get.side_effect = lambda keys, location: [
        None if key.chunk_hash in missing else sources[key.chunk_hash] for key in keys
    ]
    engine.storage_manager = storage
    return storage


def retrieve_on_all(
    engines: list[LMCacheEngine], req_ids: tuple[str, ...] = ("req", "req")
) -> list[Future[torch.Tensor]]:
    """Run one retrieve per engine concurrently, as the serving engine would."""
    with ThreadPoolExecutor(len(engines)) as pool:
        futures = [
            pool.submit(engine.retrieve, TOKENS, req_id=req_id)
            for engine, req_id in zip(engines, req_ids, strict=True)
        ]
        for future in futures:
            future.exception(timeout=TIMEOUT)
    return futures


def prefetch(
    engine: LMCacheEngine,
    keys: list[CacheEngineKey],
    sources: list[MemoryObj],
    req_id: str,
) -> None:
    """Register a completed asynchronous prefetch for ``req_id``."""
    future: asyncio.Future = asyncio.new_event_loop().create_future()
    future.set_result([list(zip(keys, sources, strict=True))])
    engine.event_manager.add_event(EventType.LOADING, req_id, future)
    engine.event_manager.update_event_status(
        EventType.LOADING, req_id, EventStatus.DONE
    )


@pytest.fixture
def two_ranks() -> Generator[TwoRanks, None, None]:
    engines: list[LMCacheEngine] = []

    def build(
        group: FakeTensorParallelGroup, **options: Any
    ) -> tuple[list[LMCacheEngine], list[RecordingConnector]]:
        fail_follower_write = options.pop("fail_follower_write", None)
        budgets = (
            options.pop("budgets", None)
            or (options.pop("budget", TWO_CHUNK_BUDGET),) * 2
        )
        connectors = [
            RecordingConnector(fail_follower_write if rank else None)
            for rank in range(2)
        ]
        engines.extend(
            make_engine(rank, group, connectors[rank], budget=budgets[rank], **options)
            for rank in range(2)
        )
        return engines, connectors

    yield build
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("async_loading", [False, True])
@pytest.mark.parametrize(
    ("budget", "expected_windows"),
    [(CHUNK_BYTES, NUM_CHUNKS), (TWO_CHUNK_BUDGET, 3), (64 * 1024, 1)],
)
def test_every_rank_receives_the_hit_in_bounded_windows(
    two_ranks: TwoRanks, async_loading: bool, budget: int, expected_windows: int
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budget=budget, async_loading=async_loading)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)
    if async_loading:
        prefetch(engines[0], keys, sources, "req")

    masks = [future.result() for future in retrieve_on_all(engines)]

    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        assert not connector.pending
        torch.testing.assert_close(connector.destination, expected_kv())
    assert max(group.payload_sizes) <= budget
    assert len(group.payload_sizes) == expected_windows
    # Admission, one header per window, and the terminating header.
    assert group.object_count == expected_windows + 2
    assert all(source.get_ref_count() == 0 for source in sources)
    if async_loading:
        # The prefetch result is consumed exactly once; a later unpin finds nothing.
        assert (
            engines[0].event_manager.get_event_status(EventType.LOADING, "req")
            == EventStatus.NOT_FOUND
        )
        engines[0].lookup_unpin("req")
        assert all(source.get_ref_count() == 0 for source in sources)
    else:
        storage.batched_get.assert_called_once()


def test_missing_chunk_truncates_to_the_same_prefix_on_every_rank(
    two_ranks: TwoRanks,
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources, missing=frozenset({2}))

    masks = [future.result() for future in retrieve_on_all(engines)]

    prefix = 2 * CHUNK_TOKENS
    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.tolist() == [True] * prefix + [False] * (NUM_TOKENS - prefix)
        torch.testing.assert_close(
            connector.destination[:, :, :prefix], expected_kv()[:, :, :prefix]
        )
        assert (connector.destination[:, :, prefix:] == -1).all()
    # Every chunk that storage handed out was released; the missing one never was.
    assert [source.get_ref_count() for source in sources] == [0, 0, 1, 0, 0]


def test_failed_write_is_fenced_before_the_buffer_is_reused(
    two_ranks: TwoRanks, caplog: pytest.LogCaptureFixture
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budget=CHUNK_BYTES, fail_follower_write=2)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    with caplog.at_level(logging.ERROR, logger="lmcache.v1.cache_engine"):
        leader, follower = (future.result() for future in retrieve_on_all(engines))

    assert leader.all()
    torch.testing.assert_close(connectors[0].destination, expected_kv())
    # The failed rank reports only the window it committed...
    assert follower.tolist() == [True] * CHUNK_TOKENS + [False] * (
        NUM_TOKENS - CHUNK_TOKENS
    )
    # ...but the read it had already queued still completed, with the right
    # bytes, before the next window overwrote the staging buffer.
    assert not connectors[1].pending
    written = 2 * CHUNK_TOKENS
    torch.testing.assert_close(
        connectors[1].destination[:, :, :written], expected_kv()[:, :, :written]
    )
    assert (connectors[1].destination[:, :, written:] == -1).all()
    # The failed rank kept receiving, so the exchange ran to completion.
    assert len(group.payload_sizes) == NUM_CHUNKS
    assert "could not write a KV window" in caplog.text
    assert all(source.get_ref_count() == 0 for source in sources)

    # The very next request succeeds on every rank.
    connectors[1].fail_on_write = None
    attach_storage(engines[0], keys, sources)
    for future in retrieve_on_all(engines):
        assert future.result().all()
    torch.testing.assert_close(connectors[1].destination, expected_kv())


@pytest.mark.parametrize("unhealthy_rank", [0, 1])
def test_unhealthy_rank_is_refused_at_admission(
    two_ranks: TwoRanks, unhealthy_rank: int
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)
    engines[unhealthy_rank].mark_init_failed("test")

    masks = [future.result() for future in retrieve_on_all(engines)]

    assert not any(mask.any() for mask in masks)
    assert all((connector.destination == -1).all() for connector in connectors)
    storage.batched_get.assert_not_called()
    assert group.object_count == 1
    assert not group.payload_sizes


def test_rank_without_a_staging_buffer_is_refused_at_admission(
    two_ranks: TwoRanks, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = FakeTensorParallelGroup(2)
    real_empty = torch.empty

    def empty_without_staging(*args: Any, **kwargs: Any) -> torch.Tensor:
        if args and args[0] == CHUNK_BYTES and kwargs.get("dtype") is torch.uint8:
            raise torch.OutOfMemoryError("injected staging reservation failure")
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", empty_without_staging)
        engines, _ = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines)]

    assert not any(mask.any() for mask in masks)
    storage.batched_get.assert_not_called()
    assert not group.payload_sizes


@pytest.mark.parametrize(
    ("follower_budget", "admitted"),
    [(CHUNK_BYTES, False), (TWO_CHUNK_BUDGET + CHUNK_ALIGNMENT, True)],
)
def test_follower_must_hold_at_least_the_source_capacity(
    two_ranks: TwoRanks, follower_budget: int, admitted: bool
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budgets=(TWO_CHUNK_BUDGET, follower_budget))
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines)]

    assert all(bool(mask.all()) == admitted for mask in masks)
    if admitted:
        torch.testing.assert_close(connectors[1].destination, expected_kv())
        assert max(group.payload_sizes) <= TWO_CHUNK_BUDGET
    else:
        assert not group.payload_sizes


def test_ranks_serving_different_requests_are_refused_at_admission(
    two_ranks: TwoRanks,
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines, req_ids=("a", "b"))]

    assert not any(mask.any() for mask in masks)
    assert not group.payload_sizes


def test_source_storage_failure_releases_peers_before_propagating(
    two_ranks: TwoRanks,
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)
    storage.batched_get.side_effect = OSError("injected storage failure")

    leader, follower = retrieve_on_all(engines)

    with pytest.raises(OSError, match="injected storage failure"):
        leader.result()
    assert not follower.result().any()
    # Admission plus the terminating header.
    assert group.object_count == 2
    # A storage failure is local: the engine stays usable.
    assert engines[0].is_healthy()
    storage.batched_get.side_effect = lambda keys, location: [
        sources[key.chunk_hash] for key in keys
    ]
    for future in retrieve_on_all(engines):
        assert future.result().all()


def test_budget_smaller_than_one_chunk_is_rejected_at_init() -> None:
    group = FakeTensorParallelGroup(1)
    with pytest.raises(ValueError, match="at least one KV chunk"):
        make_engine(
            0, group, RecordingConnector(), world_size=1, budget=CHUNK_BYTES - 1
        )


def test_chunk_larger_than_the_buffer_fails_before_any_window(
    two_ranks: TwoRanks,
) -> None:
    """A layout change after init cannot grow the buffer; it fails loudly."""
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, budget=CHUNK_BYTES)
    keys, sources = make_sources(tokens_per_chunk=2 * CHUNK_TOKENS)
    attach_storage(engines[0], keys, sources, tokens_per_chunk=2 * CHUNK_TOKENS)

    leader, follower = retrieve_on_all(engines)

    with pytest.raises(ValueError, match="exceeds retrieve_buffer_size"):
        leader.result()
    assert not follower.result().any()
    assert not group.payload_sizes
    assert all(source.get_ref_count() == 0 for source in sources)
    assert engines[0].is_healthy()


def test_single_rank_stages_through_the_buffer_without_peers() -> None:
    group = FakeTensorParallelGroup(1)
    connector = RecordingConnector()
    engine = make_engine(0, group, connector, world_size=1, budget=TWO_CHUNK_BUDGET)
    keys, sources = make_sources()
    attach_storage(engine, keys, sources)
    try:
        mask = engine.retrieve(TOKENS, req_id="req")
    finally:
        engine.close()

    assert mask.all()
    torch.testing.assert_close(connector.destination, expected_kv())
    assert len(group.payload_sizes) == 3
    assert all(source.get_ref_count() == 0 for source in sources)


def test_single_rank_async_retrieval_releases_prefetched_objects_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Plain (non-broadcast) retrieval must not let lookup_unpin free twice."""
    group = FakeTensorParallelGroup(1)
    connector = RecordingConnector()
    engine = make_engine(
        0, group, connector, world_size=1, async_loading=True, use_mla=False
    )
    keys, sources = make_sources()
    attach_storage(engine, keys, sources)
    prefetch(engine, keys, sources, "req")
    try:
        with caplog.at_level(logging.WARNING, logger="lmcache.v1.memory_management"):
            assert engine.retrieve(TOKENS, req_id="req").all()
            engine.lookup_unpin("req")
    finally:
        engine.close()

    assert all(source.get_ref_count() == 0 for source in sources)
    assert "Double free" not in caplog.text
    assert not connector.pending
    torch.testing.assert_close(connector.destination, expected_kv())


def test_single_rank_write_failure_is_fenced_before_sources_are_released() -> None:
    """Plain retrieval fences queued reads even when the connector raised."""
    group = FakeTensorParallelGroup(1)
    connector = RecordingConnector(fail_on_write=1)
    engine = make_engine(0, group, connector, world_size=1, use_mla=False)
    keys, sources = make_sources()
    attach_storage(engine, keys, sources)
    try:
        with pytest.raises(torch.OutOfMemoryError, match="injected"):
            engine.retrieve(TOKENS, req_id="req")
    finally:
        engine.close()

    assert not connector.pending
    torch.testing.assert_close(connector.destination, expected_kv())
    assert all(source.get_ref_count() == 0 for source in sources)


def test_fatal_transfer_failure_disables_the_engine() -> None:
    """After a collective failure the engine refuses retrievals at admission."""
    group = FakeTensorParallelGroup(1)
    connector = RecordingConnector()
    keys, sources = make_sources()

    def failing_broadcast(tensor: torch.Tensor, src: int) -> None:
        raise RuntimeError("injected communicator failure")

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_TOKENS,
        retrieve_buffer_size=TWO_CHUNK_BUDGET,
        py_enable_gc=True,
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 1, CHUNK_TOKENS, 1, 3),
        use_mla=True,
    )
    engine = LMCacheEngine(
        config,
        metadata,
        MagicMock(spec=TokenDatabase),
        connector,
        failing_broadcast,
        partial(group.broadcast_object, 0),
    )
    storage = attach_storage(engine, keys, sources)
    try:
        with pytest.raises(RuntimeError, match="communicator failure"):
            engine.retrieve(TOKENS, req_id="first")
        assert not engine.is_healthy()
        assert all(source.get_ref_count() == 0 for source in sources)
        assert not engine.retrieve(TOKENS, req_id="second").any()
    finally:
        engine.close()
    storage.batched_get.assert_called_once()


@pytest.mark.parametrize("budget", [0, -1])
def test_retrieval_budget_must_be_positive(budget: int) -> None:
    config = LMCacheEngineConfig.from_defaults(retrieve_buffer_size=budget)
    with pytest.raises(ValueError, match="retrieve_buffer_size"):
        config.validate()


def test_retrieval_budget_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMCACHE_RETRIEVE_BUFFER_SIZE", "4096")
    assert LMCacheEngineConfig.from_env().retrieve_buffer_size == 4096


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_staging_does_not_grow_with_the_hit(two_ranks: TwoRanks) -> None:
    """Device memory used by a retrieval stays within the fixed staging buffer."""
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, budget=TWO_CHUNK_BUDGET)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    masks = [future.result() for future in retrieve_on_all(engines)]

    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        torch.testing.assert_close(connector.destination, expected_kv())
    # The fake group clones one window per broadcast; nothing else may appear.
    assert torch.cuda.max_memory_allocated() - baseline <= 2 * TWO_CHUNK_BUDGET
