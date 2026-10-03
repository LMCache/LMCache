# SPDX-License-Identifier: Apache-2.0
"""Generator-contract tests for ``LMCacheEngine.retrieve_layer``.

``retrieve_layer`` returns a generator that its callers advance a fixed number
of times: once per model layer, once more to finalize the GPU connector, and a
final time to receive the boolean mask of retrieved tokens. That is
``num_layers + 2`` advances in total, and it is the same on both the vLLM path
(``LMCacheConnectorV1Impl.wait_for_layer_load``) and the layerwise benchmark
path (``_layerwise_retrieve_vllm_contract``).

Callers cannot know in advance whether the lookup will hit -- with async
loading the decision is made by the scheduler and the retrieve happens later on
the worker -- so every exit path of the generator has to yield the same number
of values.
"""

# Standard
from typing import Optional
import shutil
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)

# Local
from .utils import (
    check_paged_kv_cache_equal,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]

_NUM_LAYERS = 4
_CHUNK_SIZE = 256
_NUM_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 16
_NUM_BLOCKS = 256
_DTYPE = torch.bfloat16
_DISK_PATH = "local/disk_test/local_disk/"


def _create_engine(autorelease_v1, backend: str = "cpu") -> LMCacheEngine:
    """Build a layerwise engine on top of ``backend``.

    :param autorelease_v1: The fixture that destroys the engine afterwards.

    :param str backend: A ``from_legacy`` backend name. ``"local_cpu_disk"``
        adds a disk tier, which is what makes a retrieve load through CPU
        staging objects that have to be unpinned afterwards.
    """
    if "disk" in backend:
        shutil.rmtree(_DISK_PATH, ignore_errors=True)
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=_CHUNK_SIZE,
        backend=backend,
        use_layerwise=True,
    )
    connector = VLLMPagedMemLayerwiseGPUConnector(
        _NUM_HEADS * _HEAD_DIM,
        _NUM_LAYERS,
        use_gpu=True,
        chunk_size=_CHUNK_SIZE,
        dtype=_DTYPE,
        device=torch.device(torch_device_type),
    )
    kv_shape = (_NUM_LAYERS, 2, _CHUNK_SIZE, _NUM_HEADS, _HEAD_DIM)
    return autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )


def _pump(retriever, num_layers: int = _NUM_LAYERS) -> list[Optional[torch.Tensor]]:
    """Advance a ``retrieve_layer`` generator the way its callers do.

    :param retriever: The generator returned by ``retrieve_layer``.

    :param int num_layers: The number of model layers.

    return: The ``num_layers + 2`` yielded values, in order.
    """
    return [next(retriever) for _ in range(num_layers + 2)]


def _wait_for_store(engine: LMCacheEngine, tokens: torch.Tensor, expected: int) -> None:
    """Block until an asynchronous store is visible to ``lookup``."""
    deadline = time.time() + 30
    hit = engine.lookup(tokens)
    while hit != expected:
        if time.time() > deadline:
            raise TimeoutError(f"store did not finish: expected {expected}, got {hit}")
        time.sleep(0.01)
        hit = engine.lookup(tokens)


def _drop_cpu_tier(engine: LMCacheEngine, tokens: torch.Tensor, expected: int) -> None:
    """Leave ``tokens`` on disk only, so a retrieve has to load through staging.

    ``clear`` skips objects that are still pinned by the store that just
    finished, so it is retried until the CPU tier really holds none of these
    tokens. A partially cleared tier would put the chunks in two different
    locations, which layerwise retrieval rejects outright.
    """
    storage_manager = engine.storage_manager
    assert storage_manager is not None
    deadline = time.time() + 30
    while True:
        storage_manager.clear(locations=["LocalCPUBackend"])
        on_disk = engine.lookup(tokens, search_range=["LocalDiskBackend"])
        in_cpu = engine.lookup(tokens, search_range=["LocalCPUBackend"])
        if on_disk == expected and in_cpu == 0:
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"tokens are not disk-only: on disk {on_disk}/{expected}, "
                f"still in CPU {in_cpu}"
            )
        time.sleep(0.01)


def _paged_kv_cache() -> list[torch.Tensor]:
    return generate_kv_cache_paged_list_tensors(
        _NUM_BLOCKS,
        torch.device(torch_device_type),
        _BLOCK_SIZE,
        _DTYPE,
        num_layers=_NUM_LAYERS,
        head_size=_HEAD_DIM,
    )


def _slot_mapping(num_tokens: int) -> torch.Tensor:
    """Pick ``num_tokens`` distinct slots out of the paged KV cache."""
    return torch.randperm(
        _NUM_BLOCKS * _BLOCK_SIZE, device=torch.device(torch_device_type)
    )[:num_tokens]


def _store_layerwise(
    engine: LMCacheEngine,
    tokens: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> None:
    """Store ``tokens`` through the layerwise contract, then wait for it."""
    storer = engine.store_layer(
        tokens.tolist(),
        mask=torch.ones(len(tokens), dtype=torch.bool, device=slot_mapping.device),
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        offset=0,
        sync=True,
    )
    for _ in range(_NUM_LAYERS + 1):
        next(storer)
    _wait_for_store(engine, tokens, len(tokens))


@pytest.mark.parametrize("backend", ["cpu", "local_cpu_disk"])
def test_retrieve_layer_total_miss_completes_with_empty_mask(autorelease_v1, backend):
    """A retrieve for tokens that are not cached must still yield in full.

    Before the fix the last advance raised ``UnboundLocalError`` because the
    variables the tail of the generator touches are only bound when at least
    one chunk was found.
    """
    engine = _create_engine(autorelease_v1, backend)
    tokens = generate_tokens(_CHUNK_SIZE * 2, torch_device_type)
    assert engine.lookup(tokens) == 0

    retriever = engine.retrieve_layer(tokens)
    yielded = _pump(retriever)

    assert all(value is None for value in yielded[:-1])
    ret_mask = yielded[-1]
    assert isinstance(ret_mask, torch.Tensor)
    assert ret_mask.dtype == torch.bool
    assert ret_mask.numel() == len(tokens)
    assert not bool(ret_mask.any())
    with pytest.raises(StopIteration):
        next(retriever)


def test_retrieve_layer_reports_miss_when_first_chunk_disappears(autorelease_v1):
    """The lookup-then-retrieve race, which is how this happens in production.

    The scheduler decides to load based on ``lookup``; the worker calls
    ``retrieve_layer`` later. Anything evicted in between turns a hit into a
    miss. ``retrieve_layer`` stops at the first chunk it cannot find, so
    losing the first chunk alone is enough to make the whole retrieve empty,
    even though the remaining chunks are still cached.
    """
    engine = _create_engine(autorelease_v1)
    tokens = generate_tokens(_CHUNK_SIZE * 3, torch_device_type)
    src_kv = _paged_kv_cache()
    dst_kv = _paged_kv_cache()
    slot_mapping = _slot_mapping(len(tokens))

    _store_layerwise(engine, tokens, src_kv, slot_mapping)
    assert engine.lookup(tokens) == len(tokens)

    chunk_keys = [
        key for _, _, key in engine.token_database.process_tokens(tokens=tokens)
    ]
    assert len(chunk_keys) == 3
    cpu_backend = engine.storage_manager.storage_backends["LocalCPUBackend"]
    for layer_key in chunk_keys[0].split_layers(_NUM_LAYERS):
        assert cpu_backend.remove(layer_key)

    retriever = engine.retrieve_layer(
        tokens, kvcaches=dst_kv, slot_mapping=slot_mapping, sync=True
    )
    yielded = _pump(retriever)

    ret_mask = yielded[-1]
    assert isinstance(ret_mask, torch.Tensor)
    assert not bool(ret_mask.any())
    with pytest.raises(StopIteration):
        next(retriever)


def test_retrieve_layer_partial_hit_stops_at_the_missing_chunk(autorelease_v1):
    """Losing a later chunk keeps the earlier ones, and the mask says so.

    The counterpart of the test above: the retrieve walks chunks in order and
    stops at the first one it cannot find, so a hit prefix is still loaded and
    the returned mask covers exactly that prefix.
    """
    engine = _create_engine(autorelease_v1)
    tokens = generate_tokens(_CHUNK_SIZE * 3, torch_device_type)
    src_kv = _paged_kv_cache()
    dst_kv = _paged_kv_cache()
    slot_mapping = _slot_mapping(len(tokens))

    _store_layerwise(engine, tokens, src_kv, slot_mapping)

    chunk_keys = [
        key for _, _, key in engine.token_database.process_tokens(tokens=tokens)
    ]
    cpu_backend = engine.storage_manager.storage_backends["LocalCPUBackend"]
    for layer_key in chunk_keys[-1].split_layers(_NUM_LAYERS):
        assert cpu_backend.remove(layer_key)

    retriever = engine.retrieve_layer(
        tokens, kvcaches=dst_kv, slot_mapping=slot_mapping, sync=True
    )
    yielded = _pump(retriever)

    hit_tokens = _CHUNK_SIZE * 2
    assert int(yielded[0]) == hit_tokens
    ret_mask = yielded[-1]
    assert isinstance(ret_mask, torch.Tensor)
    assert bool(ret_mask[:hit_tokens].all())
    assert not bool(ret_mask[hit_tokens:].any())
    with pytest.raises(StopIteration):
        next(retriever)

    check_paged_kv_cache_equal(
        src_kv,
        dst_kv,
        slot_mapping[:hit_tokens],
        num_heads=_NUM_HEADS,
        head_size=_HEAD_DIM,
    )


@pytest.mark.parametrize("backend", ["cpu", "local_cpu_disk"])
def test_retrieve_layer_hit_restores_kv_cache(autorelease_v1, backend):
    """The hit path: same number of yields, and the KV cache is restored.

    This also pins the first yield, which is the retrieved token count that
    the SGLang integration reads instead of calling ``lookup``.

    The disk tier matters because those loads go through pinned CPU staging
    objects: the unpin loop that moves with this change is what keeps the
    staging pool from starving, and the last assertion is what notices.
    """
    engine = _create_engine(autorelease_v1, backend)
    tokens = generate_tokens(_CHUNK_SIZE * 2, torch_device_type)
    src_kv = _paged_kv_cache()
    dst_kv = _paged_kv_cache()
    slot_mapping = _slot_mapping(len(tokens))

    _store_layerwise(engine, tokens, src_kv, slot_mapping)
    if "disk" in backend:
        _drop_cpu_tier(engine, tokens, len(tokens))

    retriever = engine.retrieve_layer(
        tokens, kvcaches=dst_kv, slot_mapping=slot_mapping, sync=True
    )
    yielded = _pump(retriever)

    assert int(yielded[0]) == len(tokens)
    ret_mask = yielded[-1]
    assert isinstance(ret_mask, torch.Tensor)
    assert bool(ret_mask.all())
    with pytest.raises(StopIteration):
        next(retriever)

    check_paged_kv_cache_equal(
        src_kv,
        dst_kv,
        slot_mapping,
        num_heads=_NUM_HEADS,
        head_size=_HEAD_DIM,
    )

    # Nothing may stay pinned: on the disk tier every object the retrieve
    # loaded arrives pinned, and the tail of the generator is what releases
    # them. A leaked pin exhausts the staging pool on a later allocate().
    cpu_backend = engine.storage_manager.storage_backends["LocalCPUBackend"]
    assert not any(
        memory_obj.is_pinned for memory_obj in list(cpu_backend.hot_cache.values())
    )
