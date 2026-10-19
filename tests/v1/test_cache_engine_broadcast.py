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
from contextlib import AbstractContextManager
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
from lmcache.utils import (
    CacheEngineKey,
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)
from lmcache.v1.broadcast_transfer import staging_bytes
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import TokenDatabase
import lmcache.lmcache_native as lmcache_native

# Local
from .test_broadcast_transfer import make_object

pytestmark = pytest.mark.no_shared_allocator

TIMEOUT = 20.0
CHUNK_TOKENS = 4
HIDDEN = 3
CHUNK_BYTES = 1 * 2 * CHUNK_TOKENS * HIDDEN * 4
NUM_CHUNKS = 5
NUM_TOKENS = NUM_CHUNKS * CHUNK_TOKENS
TOKENS = list(range(NUM_TOKENS))

TwoRanks = Callable[..., tuple[list[LMCacheEngine], list["RecordingConnector"]]]
Collectives = tuple[
    Callable[[torch.Tensor, int], None],
    Callable[[Any, int], Any],
    Callable[[bool], bool] | None,
]
LogCapture = Callable[[str, int], AbstractContextManager[None]]


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

    def bind(self, rank: int) -> Collectives:
        """The broadcast, object broadcast, and agreement callbacks of ``rank``."""
        return (
            partial(self.broadcast, rank),
            partial(self.broadcast_object, rank),
            partial(self.agree, rank),
        )


class BrokenCommunicatorGroup(FakeTensorParallelGroup):
    """Every rank's tensor broadcast fails, as a dead communicator does."""

    def broadcast(self, rank: int, tensor: torch.Tensor, src: int) -> None:
        raise RuntimeError("injected communicator failure")


class RecordingConnector(MockGPUConnector):
    """Complete page writes at the fence, like a native writer with its own queue.

    ``batched_to_gpu`` only queues the views it was given; ``synchronize_load``
    performs the reads into ``destination``. An injected failure raises after
    queuing, as a native writer fails once part of its work is already in
    flight. Reads that run after the staging buffer was reused land wrong data.
    """

    def __init__(
        self, fail_on_write: int | None = None, *, hidden: int = HIDDEN
    ) -> None:
        super().__init__((2, 1, CHUNK_TOKENS, 1, hidden))
        self.destination = torch.full((1, 2, NUM_TOKENS, hidden), -1.0)
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


def expected_kv(hidden: int = HIDDEN) -> torch.Tensor:
    return torch.arange(2 * NUM_TOKENS * hidden, dtype=torch.float32).reshape(
        1, 2, NUM_TOKENS, hidden
    )


def make_sources(
    tokens_per_chunk: int = CHUNK_TOKENS, hidden: int = HIDDEN
) -> tuple[list[CacheEngineKey], list[MemoryObj]]:
    starts = range(0, NUM_TOKENS, tokens_per_chunk)
    keys = [
        CacheEngineKey("test", 1, 0, index, torch.float32)
        for index in range(len(starts))
    ]
    kv = expected_kv(hidden)
    sources = [
        make_object([kv[:, :, start : start + tokens_per_chunk]]) for start in starts
    ]
    return keys, sources


def make_engine(
    rank: int,
    collectives: Collectives,
    connector: MockGPUConnector,
    *,
    world_size: int = 2,
    window_chunks: int = 2,
    window_bytes: int | None = None,
    async_loading: bool = False,
    use_mla: bool = True,
) -> LMCacheEngine:
    """Build one rank's engine; ``window_bytes`` overrides the chunk-count window.

    The metadata carries the layout a serving engine derives from its model
    configuration, ``HIDDEN`` wide; what it registers later may differ.
    """
    if window_bytes is None:
        window_bytes = staging_bytes(CHUNK_BYTES, window_chunks)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_TOKENS,
        retrieve_window_bytes=window_bytes,
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
        config, metadata, MagicMock(spec=TokenDatabase), connector, *collectives
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


def register_kv_layout(engine: LMCacheEngine, hidden: int) -> None:
    """Install the layout a serving engine registers after the engine exists.

    vLLM's V3 connector builds it at the first KV transfer, so only a rank
    that has transferred carries it. The layout has ``hidden`` columns per
    token, in the MLA format, over the two layers the configured one describes.
    """
    kv_caches: DiscoverableKVCache = [
        torch.zeros(1, CHUNK_TOKENS, hidden) for _ in range(2)
    ]
    engine.metadata.kv_layer_groups_manager = KVLayerGroupsManager(
        kv_caches,
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_NB_BS_HS] * 2,
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
    )


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
        hidden = options.pop("hidden", HIDDEN)
        window_chunks = (
            options.pop("window_chunks_per_rank", None)
            or (options.pop("window_chunks", 2),) * 2
        )
        connectors = [
            RecordingConnector(fail_follower_write if rank else None, hidden=hidden)
            for rank in range(2)
        ]
        engines.extend(
            make_engine(
                rank,
                group.bind(rank),
                connectors[rank],
                window_chunks=window_chunks[rank],
                **options,
            )
            for rank in range(2)
        )
        return engines, connectors

    yield build
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("async_loading", [False, True])
@pytest.mark.parametrize(
    ("window_chunks", "expected_windows"),
    [(1, NUM_CHUNKS), (2, 3), (8, 1)],
)
def test_every_rank_receives_the_hit_in_bounded_windows(
    two_ranks: TwoRanks, async_loading: bool, window_chunks: int, expected_windows: int
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(
        group, window_chunks=window_chunks, async_loading=async_loading
    )
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)
    if async_loading:
        prefetch(engines[0], keys, sources, "req")

    masks = [future.result() for future in retrieve_on_all(engines)]

    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        assert not connector.pending
        torch.testing.assert_close(connector.destination, expected_kv())
    assert max(group.payload_sizes) <= staging_bytes(CHUNK_BYTES, window_chunks)
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
    engines, connectors = two_ranks(group, window_chunks=1)
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
    two_ranks: TwoRanks, caplog: pytest.LogCaptureFixture, lmcache_logs: LogCapture
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, window_chunks=1, fail_follower_write=2)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    with lmcache_logs("lmcache.v1.cache_engine", logging.ERROR):
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
    engines, connectors = two_ranks(group, window_chunks=1)
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
    """A reservation that failed at creation is retried at admission; when
    that fails as well, every rank refuses the retrieval."""
    group = FakeTensorParallelGroup(2)
    real_empty = torch.empty

    def empty_without_staging(*args: Any, **kwargs: Any) -> torch.Tensor:
        if args and args[0] == CHUNK_BYTES and kwargs.get("dtype") is torch.uint8:
            raise torch.OutOfMemoryError("injected staging reservation failure")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_without_staging)
    engines, _ = two_ranks(group, window_chunks=1)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines)]

    assert not any(mask.any() for mask in masks)
    storage.batched_get.assert_not_called()
    assert not group.payload_sizes


@pytest.mark.parametrize("follower_chunks", [1, 3])
def test_follower_adopts_the_source_plan_at_admission(
    two_ranks: TwoRanks,
    follower_chunks: int,
    caplog: pytest.LogCaptureFixture,
    lmcache_logs: LogCapture,
) -> None:
    """A rank whose buffer differs from the announced plan replaces it first."""
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, window_chunks_per_rank=(2, follower_chunks))
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    with lmcache_logs("lmcache.v1.cache_engine", logging.WARNING):
        masks = [future.result() for future in retrieve_on_all(engines)]

    before = staging_bytes(CHUNK_BYTES, follower_chunks)
    after = staging_bytes(CHUNK_BYTES, 2)
    assert caplog.text.count(f"staging changes from {before} to {after} bytes") == 1
    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        torch.testing.assert_close(connector.destination, expected_kv())
    assert max(group.payload_sizes) <= after


def test_ranks_serving_different_requests_are_refused_at_admission(
    two_ranks: TwoRanks,
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, window_chunks=1)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines, req_ids=("a", "b"))]

    assert not any(mask.any() for mask in masks)
    assert not group.payload_sizes


def test_source_storage_failure_releases_peers_before_propagating(
    two_ranks: TwoRanks,
) -> None:
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, window_chunks=1)
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


def test_multi_rank_engine_requires_an_agreement_callback() -> None:
    broadcast, broadcast_object, _ = FakeTensorParallelGroup(2).bind(0)
    with pytest.raises(ValueError, match="all_ranks_agree_fn"):
        make_engine(0, (broadcast, broadcast_object, None), RecordingConnector())


def test_chunk_larger_than_the_buffer_fails_before_any_window(
    two_ranks: TwoRanks,
) -> None:
    """A chunk no window can carry fails loudly, before any peer is committed."""
    group = FakeTensorParallelGroup(2)
    engines, _ = two_ranks(group, window_chunks=1)
    keys, sources = make_sources(tokens_per_chunk=2 * CHUNK_TOKENS)
    attach_storage(engines[0], keys, sources, tokens_per_chunk=2 * CHUNK_TOKENS)

    leader, follower = retrieve_on_all(engines)

    with pytest.raises(ValueError, match="exceeds"):
        leader.result()
    assert not follower.result().any()
    assert not group.payload_sizes
    assert all(source.get_ref_count() == 0 for source in sources)
    assert engines[0].is_healthy()


def test_layout_registered_on_the_source_resizes_every_rank_at_admission(
    two_ranks: TwoRanks, caplog: pytest.LogCaptureFixture, lmcache_logs: LogCapture
) -> None:
    """The first rank re-plans from its registered layout; peers follow it.

    The engines were sized for two configured chunks. Only the first rank has
    transferred KV, so only its metadata carries the registered layout, which
    doubles the chunk: each window now carries one chunk of the new size, which
    the original buffer could not have held, on both ranks. A second retrieval
    finds the buffers already right.
    """
    wide = 2 * HIDDEN
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, window_chunks=2, hidden=wide)
    register_kv_layout(engines[0], wide)
    keys, sources = make_sources(hidden=wide)
    attach_storage(engines[0], keys, sources)

    with lmcache_logs("lmcache.v1.cache_engine", logging.WARNING):
        masks = [future.result() for future in retrieve_on_all(engines)]
        attach_storage(engines[0], keys, sources)
        again = [future.result() for future in retrieve_on_all(engines)]

    before, after = staging_bytes(CHUNK_BYTES, 2), 2 * CHUNK_BYTES
    assert caplog.text.count(f"staging changes from {before} to {after} bytes") == 2
    for mask, connector in zip(masks + again, connectors * 2, strict=True):
        assert mask.all()
        torch.testing.assert_close(connector.destination, expected_kv(wide))
    assert group.payload_sizes == [2 * CHUNK_BYTES] * (2 * NUM_CHUNKS)
    assert all(source.get_ref_count() == 0 for source in sources)


def test_registering_the_configured_layout_keeps_the_staging(
    two_ranks: TwoRanks, caplog: pytest.LogCaptureFixture, lmcache_logs: LogCapture
) -> None:
    """A registered layout equal to the configured one is not reallocated."""
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, window_chunks=2)
    register_kv_layout(engines[0], HIDDEN)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    caplog.clear()
    with lmcache_logs("lmcache.v1.cache_engine", logging.INFO):
        masks = [future.result() for future in retrieve_on_all(engines)]

    assert "staging" not in caplog.text
    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        torch.testing.assert_close(connector.destination, expected_kv())
    assert max(group.payload_sizes) == staging_bytes(CHUNK_BYTES, 2)


@pytest.mark.parametrize("async_loading", [False, True])
def test_single_rank_writes_its_chunks_without_staging_or_collectives(
    async_loading: bool,
) -> None:
    """One rank has nobody to broadcast to; it retrieves like any other model.

    The callbacks are the placeholders SGLang and TensorRT-LLM pass, which raise
    if called, and the window setting is irrelevant because nothing is reserved.
    """
    connector = RecordingConnector()
    engine = make_engine(
        0,
        (mock_up_broadcast_fn, mock_up_broadcast_object_fn, None),
        connector,
        world_size=1,
        window_chunks=1,
        async_loading=async_loading,
    )
    keys, sources = make_sources()
    attach_storage(engine, keys, sources)
    if async_loading:
        prefetch(engine, keys, sources, "req")
    try:
        mask = engine.retrieve(TOKENS, req_id="req")
    finally:
        engine.close()

    assert mask.all()
    assert not connector.pending
    torch.testing.assert_close(connector.destination, expected_kv())
    assert all(source.get_ref_count() == 0 for source in sources)
    assert all(source.metadata.pin_count == 0 for source in sources)


def test_single_rank_async_retrieval_releases_prefetched_objects_once(
    caplog: pytest.LogCaptureFixture, lmcache_logs: LogCapture
) -> None:
    """Plain (non-broadcast) retrieval must not let lookup_unpin free twice."""
    connector = RecordingConnector()
    engine = make_engine(
        0,
        FakeTensorParallelGroup(1).bind(0),
        connector,
        world_size=1,
        async_loading=True,
        use_mla=False,
    )
    keys, sources = make_sources()
    attach_storage(engine, keys, sources)
    prefetch(engine, keys, sources, "req")
    try:
        with lmcache_logs("lmcache.v1.memory_management", logging.WARNING):
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
    connector = RecordingConnector(fail_on_write=1)
    engine = make_engine(
        0, FakeTensorParallelGroup(1).bind(0), connector, world_size=1, use_mla=False
    )
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


def test_fatal_transfer_failure_disables_the_engine(two_ranks: TwoRanks) -> None:
    """After a collective failure every rank refuses retrievals at admission."""
    group = BrokenCommunicatorGroup(2)
    engines, _ = two_ranks(group)
    keys, sources = make_sources()
    storage = attach_storage(engines[0], keys, sources)

    for future in retrieve_on_all(engines):
        with pytest.raises(RuntimeError, match="communicator failure"):
            future.result()
    assert not any(engine.is_healthy() for engine in engines)
    assert all(source.get_ref_count() == 0 for source in sources)

    for future in retrieve_on_all(engines):
        assert not future.result().any()
    storage.batched_get.assert_called_once()


@pytest.mark.parametrize("window_bytes", [0, -1])
def test_window_bytes_must_be_positive(window_bytes: int) -> None:
    config = LMCacheEngineConfig.from_defaults(retrieve_window_bytes=window_bytes)
    with pytest.raises(ValueError, match="retrieve_window_bytes"):
        config.validate()


def test_window_bytes_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMCACHE_RETRIEVE_WINDOW_BYTES", "4096")
    assert LMCacheEngineConfig.from_env().retrieve_window_bytes == 4096


def test_window_below_one_chunk_is_raised_to_one_chunk(
    two_ranks: TwoRanks, caplog: pytest.LogCaptureFixture, lmcache_logs: LogCapture
) -> None:
    """A target smaller than a chunk still moves whole chunks, one per window."""
    group = FakeTensorParallelGroup(2)
    with lmcache_logs("lmcache.v1.cache_engine", logging.WARNING):
        engines, connectors = two_ranks(group, window_bytes=1)
    keys, sources = make_sources()
    attach_storage(engines[0], keys, sources)

    masks = [future.result() for future in retrieve_on_all(engines)]

    assert "below one KV chunk" in caplog.text
    for mask, connector in zip(masks, connectors, strict=True):
        assert mask.all()
        torch.testing.assert_close(connector.destination, expected_kv())
    assert len(group.payload_sizes) == NUM_CHUNKS
    assert max(group.payload_sizes) == CHUNK_BYTES


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_staging_does_not_grow_with_the_hit(two_ranks: TwoRanks) -> None:
    """Device memory used by a retrieval stays within the fixed staging buffer."""
    group = FakeTensorParallelGroup(2)
    engines, connectors = two_ranks(group, window_chunks=2)
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
    assert torch.cuda.max_memory_allocated() - baseline <= 2 * staging_bytes(
        CHUNK_BYTES, 2
    )
