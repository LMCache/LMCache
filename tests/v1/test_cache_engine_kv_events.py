# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the parent chain exposed by in-process store events."""

# Standard
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.cache_engine import LMCacheEngine
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.inprocess.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

pytestmark = pytest.mark.no_shared_allocator
CHUNK_SIZE = 4
TOKENS = list(range(100, 112))


def _store(
    engine: LMCacheEngine,
    tokens: list[int] | torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Finish a public store operation, including every layerwise yield."""
    if engine.use_layerwise:
        list(engine.store_layer(tokens, mask=mask))
    else:
        engine.store(tokens, mask=mask)


def _hashes(engine: LMCacheEngine, tokens: list[int] | torch.Tensor) -> list[int]:
    """Read the real database hashes through its public processing interface."""
    return [
        cast(int, value)
        for _, _, value in engine.token_database.process_tokens(tokens, make_key=False)
    ]


@pytest.fixture(params=[False, True], ids=["store", "store_layer"])
def engine(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> LMCacheEngine:
    """Build a real engine and database with CPU-only storage/transfer doubles."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE,
        save_unfull_chunk=True,
        enable_kv_events=True,
        use_layerwise=request.param,
        py_enable_gc=True,
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, CHUNK_SIZE, 1, 8),
        chunk_size=CHUNK_SIZE,
    )
    monkeypatch.setattr(
        "lmcache.inprocess.v1.cache_engine.InitializeUsageContext", MagicMock()
    )
    monkeypatch.setattr(
        "lmcache.inprocess.v1.cache_engine.PinMonitor.GetOrCreate", MagicMock()
    )
    stats = MagicMock()
    stats.on_store_request.return_value.time_to_store.return_value = 1.0
    monkeypatch.setattr(
        "lmcache.inprocess.v1.cache_engine.LMCStatsMonitor.GetOrCreate", lambda: stats
    )
    connector = MagicMock(spec=VLLMPagedMemLayerwiseGPUConnector)
    connector.get_shape.return_value = torch.Size([2, CHUNK_SIZE, 8])
    if request.param:
        connector.batched_from_gpu.side_effect = lambda *args, **kwargs: iter(
            [None, None, None]
        )
    result = LMCacheEngine(
        config,
        metadata,
        ChunkedTokenDatabase(config, metadata),
        connector,
        MagicMock(),
        MagicMock(),
    )
    storage = MagicMock()
    storage.is_frozen.return_value = False
    storage.contains.return_value = False
    memory = MagicMock()
    memory.get_size.return_value = 64
    storage.allocate.return_value = memory
    storage.batched_allocate.return_value = [memory, memory]
    result.storage_manager = storage
    return result


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_full_store_parent_chain(
    engine: LMCacheEngine, as_tensor: bool, with_mask: bool
) -> None:
    """An unmasked sequence starts at a root and links each following chunk."""
    tokens = torch.tensor(TOKENS) if as_tensor else TOKENS
    mask = torch.ones(len(tokens), dtype=torch.bool) if with_mask else None
    hashes = _hashes(engine, tokens)
    _store(engine, tokens, mask)
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [[value] for value in hashes]
    assert [event.parent_block_hash for event in events] == [None, *hashes[:-1]]
    assert list(engine.get_kv_events()) == []


@pytest.mark.parametrize("skip_chunks", [1, 2])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_masked_store_parent_chain(
    engine: LMCacheEngine, skip_chunks: int, as_tensor: bool
) -> None:
    """A masked prefix remains the parent of the first newly stored chunk."""
    tokens = torch.tensor(TOKENS) if as_tensor else TOKENS
    hashes = _hashes(engine, tokens)
    mask = torch.arange(len(tokens)) >= skip_chunks * CHUNK_SIZE
    _store(engine, tokens, mask)
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [
        [value] for value in hashes[skip_chunks:]
    ]
    assert [event.parent_block_hash for event in events] == hashes[skip_chunks - 1 : -1]


def test_second_turn_parent_chain(engine: LMCacheEngine) -> None:
    """A later turn attaches to the preceding turn's last stored chunk."""
    _store(engine, TOKENS[:8])
    first_turn = list(engine.get_kv_events())
    _store(engine, TOKENS, torch.arange(len(TOKENS)) >= 8)
    second_turn = list(engine.get_kv_events())
    assert len(second_turn) == 1
    assert second_turn[0].parent_block_hash == first_turn[-1].block_hashes[0]


@pytest.mark.parametrize("skip_chunks", [0, 2])
def test_partial_tail_parent_chain(engine: LMCacheEngine, skip_chunks: int) -> None:
    """A saved partial tail has the same predecessor as a full final chunk."""
    tokens = TOKENS[:10]
    hashes = _hashes(engine, tokens)
    _store(engine, tokens, torch.arange(len(tokens)) >= skip_chunks * CHUNK_SIZE)
    events = list(engine.get_kv_events())
    assert [event.parent_block_hash for event in events] == [None, *hashes[:-1]][
        skip_chunks:
    ]
    assert [event.block_size for event in events] == [4, 4, 2][skip_chunks:]


def test_partial_tail_not_stored(engine: LMCacheEngine) -> None:
    """Disabling partial chunks preserves the full-chunk event chain."""
    engine.config.save_unfull_chunk = False
    _store(engine, TOKENS[:10])
    events = list(engine.get_kv_events())
    assert [event.block_size for event in events] == [4, 4]
    assert [event.parent_block_hash for event in events] == [
        None,
        _hashes(engine, TOKENS)[0],
    ]


@pytest.mark.parametrize("case", ["empty", "all_masked", "unsaved_tail"])
def test_nothing_to_store(engine: LMCacheEngine, case: str) -> None:
    """An empty effective store produces no events or allocations."""
    tokens = [] if case == "empty" else TOKENS
    mask = torch.zeros(len(tokens), dtype=torch.bool) if case == "all_masked" else None
    if case == "unsaved_tail":
        engine.config.save_unfull_chunk = False
        tokens = TOKENS[:2]
    _store(engine, tokens, mask)
    assert list(engine.get_kv_events()) == []
    cast(MagicMock, engine.storage_manager).allocate.assert_not_called()
    cast(MagicMock, engine.storage_manager).batched_allocate.assert_not_called()


@pytest.mark.parametrize("skip_chunks", [0, 1])
@pytest.mark.parametrize("successful_chunks", [0, 1])
def test_allocation_failure(
    engine: LMCacheEngine, skip_chunks: int, successful_chunks: int
) -> None:
    """Only successfully allocated chunks produce events, with correct parents."""
    storage = cast(MagicMock, engine.storage_manager)
    allocator = storage.batched_allocate if engine.use_layerwise else storage.allocate
    allocator.side_effect = [allocator.return_value] * successful_chunks + [None]
    hashes = _hashes(engine, TOKENS)
    _store(engine, TOKENS, torch.arange(len(TOKENS)) >= skip_chunks * CHUNK_SIZE)
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [
        [value] for value in hashes[skip_chunks : skip_chunks + successful_chunks]
    ]
    assert [event.parent_block_hash for event in events] == [None, *hashes[:-1]][
        skip_chunks : skip_chunks + successful_chunks
    ]


@pytest.mark.parametrize("skip_chunks", [0, 1])
def test_events_disabled(engine: LMCacheEngine, skip_chunks: int) -> None:
    """Disabling events still stores the requested chunks."""
    engine.kv_events_enabled = False
    _store(engine, TOKENS, torch.arange(len(TOKENS)) >= skip_chunks * CHUNK_SIZE)
    assert list(engine.get_kv_events()) == []
    calls = cast(MagicMock, engine.storage_manager).batched_put.call_args_list
    assert len(calls) == (engine.num_layers if engine.use_layerwise else 1)
    assert all(len(call.args[0]) == 3 - skip_chunks for call in calls)


@pytest.mark.parametrize("engine", [True], indirect=True)
@pytest.mark.parametrize("skip_chunk", [0, 1])
def test_layer_contains_skip(engine: LMCacheEngine, skip_chunk: int) -> None:
    """An already stored chunk is the parent of its immediate successor."""
    hashes = _hashes(engine, TOKENS)
    cast(MagicMock, engine.storage_manager).contains.side_effect = (
        lambda key, locations: key.chunk_hash == hashes[skip_chunk]
    )
    _store(engine, TOKENS)
    events = list(engine.get_kv_events())
    indices = [index for index in range(3) if index != skip_chunk]
    assert [event.block_hashes for event in events] == [[hashes[i]] for i in indices]
    assert [event.parent_block_hash for event in events] == [
        None if i == 0 else hashes[i - 1] for i in indices
    ]


@pytest.mark.parametrize("engine", [True], indirect=True)
def test_layer_contains_all(engine: LMCacheEngine) -> None:
    """Layerwise stores emit nothing when all chunks already exist."""
    cast(MagicMock, engine.storage_manager).contains.return_value = True
    _store(engine, TOKENS)
    assert list(engine.get_kv_events()) == []
    cast(MagicMock, engine.storage_manager).batched_allocate.assert_not_called()


@pytest.mark.parametrize("engine", [False], indirect=True)
@pytest.mark.parametrize("mask_prefix", [None, 0, 4])
@pytest.mark.parametrize("segmented", [False, True])
def test_prehashed_parent_chain(
    engine: LMCacheEngine,
    monkeypatch: pytest.MonkeyPatch,
    mask_prefix: int | None,
    segmented: bool,
) -> None:
    """Supplied hashes and variable offsets preserve their existing parent chain."""
    if segmented:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 90, 91]
        monkeypatch.setattr(
            "lmcache.inprocess.v1.token_database.AutoTokenizer.from_pretrained",
            lambda _: tokenizer,
        )
        engine.token_database = SegmentTokenDatabase(engine.config, engine.metadata)
    hashes = [10, 0, 30]
    offsets = [4, 4, 2]
    mask = None if mask_prefix is None else torch.arange(sum(offsets)) >= mask_prefix
    engine.store(
        hashes=hashes, offsets=offsets, mask=mask, slot_mapping=list(range(10))
    )
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [[value] for value in hashes]
    assert [event.parent_block_hash for event in events] == [None, 10, 0]


@pytest.mark.parametrize("skip_segments", [0, 1, 2])
@pytest.mark.parametrize("trailing_separator", [False, True])
def test_segment_parent_chain(
    engine: LMCacheEngine,
    monkeypatch: pytest.MonkeyPatch,
    skip_segments: int,
    trailing_separator: bool,
) -> None:
    """Separator boundaries must not replace the preceding nonempty segment hash."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 90, 91]
    monkeypatch.setattr(
        "lmcache.inprocess.v1.token_database.AutoTokenizer.from_pretrained",
        lambda _: tokenizer,
    )
    engine.token_database = SegmentTokenDatabase(engine.config, engine.metadata)
    tokens = [10, 11, 12, 13, 90, 91, 20, 21, 22, 23, 90, 91, 30, 31, 32, 33]
    if trailing_separator:
        tokens += [90, 91]
    hashes = _hashes(engine, tokens)[:3]
    _store(engine, tokens, torch.arange(len(tokens)) >= skip_segments * 6)
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [
        [value] for value in hashes[skip_segments:]
    ]
    assert [event.parent_block_hash for event in events] == [None, *hashes[:-1]][
        skip_segments:
    ]
    assert [event.block_size for event in events] == [4] * (3 - skip_segments)


@pytest.mark.parametrize("skip_segments", [0, 1, 2])
def test_segment_empty_ranges(
    engine: LMCacheEngine,
    monkeypatch: pytest.MonkeyPatch,
    skip_segments: int,
) -> None:
    """Leading and consecutive separators do not become stored parent blocks."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 90, 91]
    monkeypatch.setattr(
        "lmcache.inprocess.v1.token_database.AutoTokenizer.from_pretrained",
        lambda _: tokenizer,
    )
    engine.token_database = SegmentTokenDatabase(engine.config, engine.metadata)
    tokens = [
        90,
        91,
        10,
        11,
        12,
        13,
        90,
        91,
        90,
        91,
        20,
        21,
        22,
        23,
        90,
        91,
        30,
        31,
        32,
        33,
    ]
    hashes = [
        cast(int, value)
        for start, end, value in engine.token_database.process_tokens(
            tokens, make_key=False
        )
        if end > start
    ]
    mask_start = [0, 10, 16][skip_segments]
    _store(engine, tokens, torch.arange(len(tokens)) >= mask_start)
    events = list(engine.get_kv_events())
    assert [event.block_hashes for event in events] == [
        [value] for value in hashes[skip_segments:]
    ]
    assert [event.parent_block_hash for event in events] == [None, *hashes[:-1]][
        skip_segments:
    ]


def test_invalid_mask(engine: LMCacheEngine) -> None:
    """A nonaligned chunk mask retains the public ValueError contract."""
    with pytest.raises(ValueError, match="multiple of the chunk size"):
        _store(engine, TOKENS, torch.arange(len(TOKENS)) >= 1)
    assert list(engine.get_kv_events()) == []
    cast(MagicMock, engine.storage_manager).allocate.assert_not_called()
    cast(MagicMock, engine.storage_manager).batched_allocate.assert_not_called()


@pytest.mark.parametrize("state", ["frozen", "unhealthy"])
def test_store_disabled_by_engine_state(engine: LMCacheEngine, state: str) -> None:
    """Frozen and unhealthy stores produce no events or allocations."""
    if state == "frozen":
        engine.freeze(True)
        cast(MagicMock, engine.storage_manager).is_frozen.return_value = True
    else:
        engine.mark_init_failed("test")
    _store(engine, TOKENS)
    assert list(engine.get_kv_events()) == []
    cast(MagicMock, engine.storage_manager).allocate.assert_not_called()
    cast(MagicMock, engine.storage_manager).batched_allocate.assert_not_called()


@pytest.mark.parametrize("engine", [False], indirect=True)
def test_passive_store(engine: LMCacheEngine) -> None:
    """A passive non-layerwise worker produces no events or allocations."""
    engine.save_only_first_rank = True
    engine.metadata.worker_id = 1
    engine.store(TOKENS)
    assert list(engine.get_kv_events()) == []
    cast(MagicMock, engine.storage_manager).allocate.assert_not_called()
