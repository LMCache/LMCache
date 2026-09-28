# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable

# Third Party
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.metadata import LMCacheMetadata


def _create_engine(
    autorelease_v1: Callable[[LMCacheEngine], LMCacheEngine],
) -> LMCacheEngine:
    chunk_size = 4
    kv_shape = (1, 2, chunk_size, 1, 1)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=0.01,
        save_unfull_chunk=True,
        enable_kv_events=True,
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )
    return autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            config,
            metadata,
            MockGPUConnector(kv_shape),
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )


def test_kv_events_use_half_open_chunk_bounds(
    autorelease_v1: Callable[[LMCacheEngine], LMCacheEngine],
) -> None:
    engine = _create_engine(autorelease_v1)
    tokens = list(range(10))

    engine.store(tokens=tokens)

    events = list(engine.get_kv_events())
    assert [event.token_ids for event in events] == [
        tokens[0:4],
        tokens[4:8],
        tokens[8:10],
    ]
    assert [len(event.token_ids) for event in events] == [
        event.block_size for event in events
    ]


def test_kv_events_use_half_open_chunk_bounds_for_tensor_tokens(
    autorelease_v1: Callable[[LMCacheEngine], LMCacheEngine],
) -> None:
    engine = _create_engine(autorelease_v1)
    tokens = torch.arange(10, dtype=torch.long)

    engine.store(tokens=tokens)

    events = list(engine.get_kv_events())
    assert [event.token_ids for event in events] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert [len(event.token_ids) for event in events] == [
        event.block_size for event in events
    ]


def test_kv_events_map_hashes_by_chunk_index(
    autorelease_v1: Callable[[LMCacheEngine], LMCacheEngine],
) -> None:
    engine = _create_engine(autorelease_v1)
    hashes = [101, 202]
    offsets = [4, 4]

    engine.store(hashes=hashes, offsets=offsets, slot_mapping=list(range(8)))

    events = list(engine.get_kv_events())
    assert [event.token_ids for event in events] == [[101], [202]]
