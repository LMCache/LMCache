# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for lookup-pin ownership during non-layerwise retrieval.

Blocking LocalCPUBackend reads acquire references, not pins. Only lookup_unpin
may release the pins recorded by synchronous lookup, even when another request
retrieves the same chunk. Async prefetch and passive-rank cleanup retain their
separate ownership contracts.
"""

# Standard
from collections import OrderedDict
from collections.abc import Generator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, cast
from unittest.mock import MagicMock, call
import asyncio
import logging
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import ChunkedTokenDatabase
import lmcache.v1.cache_engine as cache_engine_module

# Local
from .utils import create_test_memory_obj

pytestmark = pytest.mark.no_shared_allocator

CHUNK_SIZE = 4
LOCATION = "LocalCPUBackend"


# Helpers


def _assert_counts(
    memory_objs: Sequence[MemoryObj], pins: Sequence[int], refs: int = 1
) -> None:
    """Check actual object ownership, rather than warning suppression.

    Args:
        memory_objs: Objects in token order.
        pins: Expected pin counts in the same order.
        refs: Expected reference count for every object.

    Raises:
        AssertionError: If either ownership count differs.
    """
    assert [obj.metadata.pin_count for obj in memory_objs] == list(pins)
    assert [obj.get_ref_count() for obj in memory_objs] == [refs] * len(memory_objs)


def _complete_prefetch(
    engine: LMCacheEngine, pairs: list[tuple[CacheEngineKey, MemoryObj]]
) -> None:
    """Publish a completed backend result through the real EventManager.

    Args:
        engine: Engine whose async retrieve or cancellation will consume A.
        pairs: Key/object pairs with their prefetch references already acquired.
            Event results contain one such list per backend.
    """
    loop = asyncio.new_event_loop()
    try:
        future: asyncio.Future[list[list[tuple[CacheEngineKey, MemoryObj]]]] = (
            loop.create_future()
        )
        future.set_result([pairs])
        engine.event_manager.add_event(EventType.LOADING, "A", future)
        engine.event_manager.update_event_status(
            EventType.LOADING, "A", EventStatus.DONE
        )
    finally:
        loop.close()


class _SynchronousStorageManager(StorageManager):
    """Keep production routing/get/unpin methods without starting storage threads."""

    def __init__(self, backend: LocalCPUBackend) -> None:
        """Initialize only state needed by the synchronous storage interfaces.

        Args:
            backend: Real CPU backend used by all normal test requests.
        """
        self.storage_backends = OrderedDict([(LOCATION, backend)])
        self._freeze = False
        self._freeze_lock = threading.RLock()
        self._bypassed_backends: set[str] = set()
        self._bypass_lock = threading.RLock()


@dataclass
class _CacheCase:
    """Real engine/cache objects and the recording-only GPU connector."""

    engine: LMCacheEngine
    backend: LocalCPUBackend
    connector: MagicMock
    tokens: list[int]
    keys: list[CacheEngineKey]
    memory_objs: list[MemoryObj]


# Fixtures


@pytest.fixture
def pin_monitor() -> Generator[None, None, None]:
    """Create the real pin monitor and always destroy its singleton at teardown.

    Yields:
        None while pin/unpin calls can register real test objects.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE, lmcache_instance_id="test_retrieve_pin_ownership"
    )
    PinMonitor.GetOrCreate(config)
    try:
        yield
    finally:
        PinMonitor.DestroyInstance()


@pytest.fixture
def cache_case(
    monkeypatch: pytest.MonkeyPatch, pin_monitor: None
) -> Generator[_CacheCase, None, None]:
    """Build three tiny CPU chunks with one cache reference apiece.

    GPU copies, telemetry and stats are mocked; token hashing, lookup, blocking
    retrieval, backend pinning and request cleanup are production code. No
    allocator pool, storage worker, GPU stream or engine post_init is created.

    Args:
        monkeypatch: Restores the isolated dependency replacements after the test.
        pin_monitor: Keeps the real monitor alive through object cleanup.

    Yields:
        The engine, backend, copy recorder, tokens, keys and corresponding objects.
    """
    stats = MagicMock(spec=LMCStatsMonitor)
    stats.on_retrieve_request.return_value.time_to_retrieve.return_value = 0.0
    monkeypatch.setattr(LMCStatsMonitor, "GetOrCreate", MagicMock(return_value=stats))
    monkeypatch.setattr(PrometheusLogger, "GetOrCreate", MagicMock())
    monkeypatch.setattr(TensorMemoryObj, "monitor", stats)
    monkeypatch.setattr(cache_engine_module, "InitializeUsageContext", MagicMock())
    monkeypatch.setattr(
        cache_engine_module.multiprocessing, "set_start_method", MagicMock()
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE,
        local_cpu=True,
        enable_async_loading=False,
        use_layerwise=False,
        py_enable_gc=True,
        lmcache_instance_id="test_retrieve_pin_ownership",
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(1, 2, CHUNK_SIZE, 1, 1),
        chunk_size=CHUNK_SIZE,
    )
    token_database = ChunkedTokenDatabase(config, metadata)
    connector = MagicMock(spec=GPUConnectorInterface)
    engine = LMCacheEngine(
        config, metadata, token_database, connector, MagicMock(), MagicMock()
    )
    backend = LocalCPUBackend(
        config, dst_device="cpu", memory_allocator=AdHocMemoryAllocator(device="cpu")
    )
    engine.storage_manager = _SynchronousStorageManager(backend)
    tokens = list(range(3 * CHUNK_SIZE))
    keys: list[CacheEngineKey] = []
    memory_objs: list[MemoryObj] = []
    try:
        for _, _, key in token_database.process_tokens(tokens=tokens):
            assert isinstance(key, CacheEngineKey)
            memory_obj = create_test_memory_obj(shape=torch.Size([2, CHUNK_SIZE, 1, 1]))
            keys.append(key)
            memory_objs.append(memory_obj)
            backend.submit_put_task(key, memory_obj)
            memory_obj.ref_count_down()  # Transfer allocation ownership to the cache.
        _assert_counts(memory_objs, [0, 0, 0])
        yield _CacheCase(engine, backend, connector, tokens, keys, memory_objs)
    finally:
        # Failed baseline assertions must not leak pins into the next test. Tests
        # assert counts before this final safety cleanup, which is not the oracle.
        for memory_obj in memory_objs:
            while memory_obj.metadata.pin_count > 0:
                memory_obj.unpin()
            while memory_obj.get_ref_count() > 0:
                memory_obj.ref_count_down()
        backend.hot_cache.clear()


# Active synchronous retrieval


def test_single_request_retains_lookup_pins(cache_case: _CacheCase) -> None:
    """Retrieve releases get references; request cleanup alone releases lookup pins.

    Args:
        cache_case: Real CPU cache with three chunks.
    """
    engine = cache_case.engine
    assert engine.lookup(cache_case.tokens, lookup_id="A", pin=True) == 12
    _assert_counts(cache_case.memory_objs, [1, 1, 1])

    retrieved = engine.retrieve(cache_case.tokens, req_id="A")

    assert retrieved.tolist() == [True] * 12
    cache_case.connector.batched_to_gpu.assert_called_once_with(
        cache_case.memory_objs, [0, 4, 8], [4, 8, 12], req_id="A"
    )
    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    assert engine.lookup_pins == {"A": {LOCATION: cache_case.keys}}
    engine.lookup_unpin("A")
    _assert_counts(cache_case.memory_objs, [0, 0, 0])
    assert not engine.lookup_pins


def test_shared_chunk_keeps_other_request_pinned(cache_case: _CacheCase) -> None:
    """A's retrieve and cleanup must not steal B's pin on their shared chunk.

    Args:
        cache_case: Real CPU cache; A and B both use its first chunk.
    """
    engine = cache_case.engine
    tokens = cache_case.tokens[:CHUNK_SIZE]
    memory_objs = cache_case.memory_objs[:1]
    assert engine.lookup(tokens, lookup_id="A", pin=True) == CHUNK_SIZE
    assert engine.lookup(tokens, lookup_id="B", pin=True) == CHUNK_SIZE
    _assert_counts(memory_objs, [2])

    assert engine.retrieve(tokens, req_id="A").all()
    _assert_counts(memory_objs, [2])
    engine.lookup_unpin("A")
    _assert_counts(memory_objs, [1])
    assert engine.lookup_pins == {"B": {LOCATION: cache_case.keys[:1]}}

    assert engine.retrieve(tokens, req_id="B").all()
    _assert_counts(memory_objs, [1])
    engine.lookup_unpin("B")
    _assert_counts(memory_objs, [0])
    assert not engine.lookup_pins


@pytest.mark.parametrize("request_id", [None, "no-own-lookup"])
@pytest.mark.parametrize("pin_owner", ["lookup", "controller"])
def test_retrieve_without_own_lookup_preserves_foreign_pins(
    cache_case: _CacheCase,
    request_id: str | None,
    pin_owner: Literal["lookup", "controller"],
) -> None:
    """Neither missing req_id nor missing lookup records confer pin ownership.

    Args:
        cache_case: Real CPU cache with three chunks.
        request_id: Omitted from retrieve when None; otherwise an unrelated ID.
        pin_owner: Whether another lookup or a backend/controller owns the pins.
    """
    engine = cache_case.engine
    if pin_owner == "lookup":
        assert engine.lookup(cache_case.tokens, lookup_id="owner", pin=True) == 12
    else:
        for key in cache_case.keys:
            assert cache_case.backend.pin(key)
    kwargs = {} if request_id is None else {"req_id": request_id}

    assert engine.retrieve(cache_case.tokens, **kwargs).all()

    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    if pin_owner == "lookup":
        assert engine.lookup_pins == {"owner": {LOCATION: cache_case.keys}}
        engine.lookup_unpin("owner")
    else:
        assert not engine.lookup_pins
        for key in cache_case.keys:
            assert cache_case.backend.unpin(key)
    _assert_counts(cache_case.memory_objs, [0, 0, 0])


@pytest.mark.parametrize("skipped_chunks", [1, 2, 3])
def test_masked_prefix_remains_owned_until_lookup_cleanup(
    cache_case: _CacheCase, skipped_chunks: int
) -> None:
    """Cleanup releases all lookup pins, including masked chunks never retrieved.

    Args:
        cache_case: Real CPU cache with three chunks shared by A and B.
        skipped_chunks: Number of whole prefix chunks already present on the GPU.
    """
    engine = cache_case.engine
    for request_id in ("A", "B"):
        assert engine.lookup(cache_case.tokens, lookup_id=request_id, pin=True) == 12
    mask = torch.arange(12) >= skipped_chunks * CHUNK_SIZE

    assert torch.equal(engine.retrieve(cache_case.tokens, mask, req_id="A"), mask)

    if skipped_chunks == 3:
        cache_case.connector.batched_to_gpu.assert_not_called()
    else:
        cache_case.connector.batched_to_gpu.assert_called_once_with(
            cache_case.memory_objs[skipped_chunks:],
            [0, 4, 8][skipped_chunks:],
            [4, 8, 12][skipped_chunks:],
            req_id="A",
        )
    _assert_counts(cache_case.memory_objs, [2, 2, 2])
    assert engine.lookup_pins == {
        "A": {LOCATION: cache_case.keys},
        "B": {LOCATION: cache_case.keys},
    }
    engine.lookup_unpin("A")
    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    engine.lookup_unpin("B")
    _assert_counts(cache_case.memory_objs, [0, 0, 0])
    assert not engine.lookup_pins


def test_middle_get_miss_releases_unused_refs_not_lookup_pins(
    cache_case: _CacheCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A middle read failure also returns the acquired, unused suffix reference.

    Args:
        cache_case: Three cached chunks whose lookup succeeds for A and B.
        monkeypatch: Injects one transient get failure without removing lookup pins.
    """
    engine = cache_case.engine
    for request_id in ("A", "B"):
        assert engine.lookup(cache_case.tokens, lookup_id=request_id, pin=True) == 12
    get_blocking = cache_case.backend.get_blocking

    def fail_middle_get(key: CacheEngineKey) -> MemoryObj | None:
        """Leave the cached middle chunk pinned but fail to read it once."""
        return None if key == cache_case.keys[1] else get_blocking(key)

    get_spy = MagicMock(side_effect=fail_middle_get)
    monkeypatch.setattr(cache_case.backend, "get_blocking", get_spy)

    retrieved = engine.retrieve(cache_case.tokens, req_id="A")

    assert retrieved.tolist() == [True] * 4 + [False] * 8
    assert get_spy.call_args_list == [call(key) for key in cache_case.keys]
    cache_case.connector.batched_to_gpu.assert_called_once_with(
        cache_case.memory_objs[:1], [0], [4], req_id="A"
    )
    _assert_counts(cache_case.memory_objs, [2, 2, 2])
    engine.lookup_unpin("A")
    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    assert engine.lookup_pins == {"B": {LOCATION: cache_case.keys}}
    engine.lookup_unpin("B")
    _assert_counts(cache_case.memory_objs, [0, 0, 0])
    assert not engine.lookup_pins


def test_cancel_and_repeated_unpin_preserve_other_request(
    cache_case: _CacheCase, caplog: pytest.LogCaptureFixture
) -> None:
    """Cancelled lookups clean once, without consuming another request's pins.

    Args:
        cache_case: Real CPU cache shared by A and B.
        caplog: Captures double-unpin warnings in addition to the count assertions.
    """
    engine = cache_case.engine
    caplog.set_level(logging.WARNING, logger="lmcache")
    for request_id in ("A", "B"):
        assert engine.lookup(cache_case.tokens, lookup_id=request_id, pin=True) == 12
    _assert_counts(cache_case.memory_objs, [2, 2, 2])

    engine.lookup_unpin("A")
    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    engine.lookup_unpin("A")
    engine.lookup_unpin("unknown")
    _assert_counts(cache_case.memory_objs, [1, 1, 1])
    assert engine.lookup_pins == {"B": {LOCATION: cache_case.keys}}
    engine.lookup_unpin("B")
    engine.lookup_unpin("B")
    _assert_counts(cache_case.memory_objs, [0, 0, 0])
    assert not engine.lookup_pins
    assert "Double unpin" not in caplog.text


def test_targeted_clear_during_retrieve_releases_lookup_pin(
    cache_case: _CacheCase,
) -> None:
    """Force-deleting a key during retrieval releases the lookup pins it holds.

    The lookup ledger stores keys, so a later ``lookup_unpin`` cannot reach an
    object whose key has already been dropped from the cache. Forced removal
    therefore releases those pins itself. This deterministic transfer-boundary
    interleaving does not use real threads.

    Args:
        cache_case: Real CPU engine and backend with a recording GPU boundary.
    """
    engine = cache_case.engine
    tokens = cache_case.tokens[:CHUNK_SIZE]
    memory_obj = cache_case.memory_objs[0]
    assert engine.lookup(tokens, lookup_id="A", pin=True) == CHUNK_SIZE

    def clear_during_transfer(
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: object,
    ) -> None:
        """Force-delete the real cache key while retrieval owns its get reference."""
        assert memory_objs == [memory_obj]
        _assert_counts(memory_objs, [1], refs=2)
        assert engine.clear(tokens=tokens) == 1

    cache_case.connector.batched_to_gpu.side_effect = clear_during_transfer
    assert engine.retrieve(tokens, req_id="A").all()
    engine.lookup_unpin("A")

    assert not engine.lookup_pins
    assert cache_case.keys[0] not in cache_case.backend.hot_cache
    _assert_counts([memory_obj], [0], refs=0)


def test_targeted_clear_releases_pins_of_every_lookup(
    cache_case: _CacheCase, caplog: pytest.LogCaptureFixture
) -> None:
    """Forced removal releases the whole pin count, not just one holder's share.

    Both lookups recorded the same key, so neither ``lookup_unpin`` can reach the
    object once the key is gone. Releasing every pin at removal keeps the count
    from being over-released later, which would log a double-unpin warning.

    Args:
        cache_case: Real CPU cache whose first chunk is pinned by A and B.
        caplog: Captures double-unpin warnings in addition to the count assertions.
    """
    engine = cache_case.engine
    caplog.set_level(logging.WARNING, logger="lmcache")
    tokens = cache_case.tokens[:CHUNK_SIZE]
    memory_obj = cache_case.memory_objs[0]
    for request_id in ("A", "B"):
        assert engine.lookup(tokens, lookup_id=request_id, pin=True) == CHUNK_SIZE
    _assert_counts([memory_obj], [2])

    assert engine.clear(tokens=tokens) == 1
    _assert_counts([memory_obj], [0], refs=0)

    engine.lookup_unpin("A")
    engine.lookup_unpin("B")

    assert not engine.lookup_pins
    assert cache_case.keys[0] not in cache_case.backend.hot_cache
    _assert_counts([memory_obj], [0], refs=0)
    assert "Double unpin" not in caplog.text


def test_unforced_remove_keeps_lookup_pin(cache_case: _CacheCase) -> None:
    """Eviction-path removal must not drop pins recorded by an active lookup.

    ``remove(force=False)`` is the internal eviction path, which pre-filters
    pinned objects through ``can_evict``. A pinned object reaching it means the
    caller is confused, so the pin is preserved rather than silently released.

    Args:
        cache_case: Real CPU cache with three chunks.
    """
    backend = cache_case.backend
    engine = cache_case.engine
    tokens = cache_case.tokens[:CHUNK_SIZE]
    memory_obj = cache_case.memory_objs[0]
    assert engine.lookup(tokens, lookup_id="A", pin=True) == CHUNK_SIZE
    _assert_counts([memory_obj], [1])

    assert backend.remove(cache_case.keys[0], force=False)

    assert cache_case.keys[0] not in backend.hot_cache
    _assert_counts([memory_obj], [1], refs=0)


# Existing cleanup contracts outside active synchronous retrieval


@pytest.mark.parametrize("completion", ["retrieve", "cancel"])
def test_async_cleanup_releases_one_prefetch_pin(
    cache_case: _CacheCase,
    completion: Literal["retrieve", "cancel"],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Async cleanup keeps an external pin and accepts unpinned backend results.

    Args:
        cache_case: Three CPU objects returned through a real completed event.
        completion: Whether successful retrieval or cancellation consumes the result.
        caplog: Captures double-unpin and double-free warnings.
    """
    engine = cache_case.engine
    engine.async_loading = True
    caplog.set_level(logging.WARNING, logger="lmcache")
    # Chunk 0 has a prefetch pin; chunk 1 models a backend returning no pin;
    # chunk 2 has both a prefetch pin and a separate controller-owned pin.
    for key, pins in zip(cache_case.keys, [1, 0, 2], strict=True):
        for _ in range(pins):
            assert cache_case.backend.pin(key)
    prefetched = asyncio.run(
        cache_case.backend.batched_get_non_blocking("A", cache_case.keys)
    )
    _assert_counts(prefetched, [1, 0, 2], refs=2)
    _complete_prefetch(engine, list(zip(cache_case.keys, prefetched, strict=True)))

    if completion == "retrieve":
        assert engine.retrieve(cache_case.tokens, req_id="A").all()
        # Successful retrieve retains the event record. Remove only the record,
        # not its already-consumed references; aborted lookups use lookup_unpin.
        engine.event_manager.pop_event(EventType.LOADING, "A")
    else:
        engine.lookup_unpin("A")
        engine.lookup_unpin("A")

    _assert_counts(cache_case.memory_objs, [0, 0, 1])
    assert engine.event_manager.get_event_status(EventType.LOADING, "A") == (
        EventStatus.NOT_FOUND
    )
    assert not engine.lookup_pins
    assert cache_case.backend.unpin(cache_case.keys[2])
    _assert_counts(cache_case.memory_objs, [0, 0, 0])
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text


@pytest.mark.parametrize("received_pins", [0, 1])
def test_passive_retrieve_keeps_received_object_cleanup(
    cache_case: _CacheCase,
    monkeypatch: pytest.MonkeyPatch,
    received_pins: int,
) -> None:
    """Passive ranks still release their temporary broadcast object's pin and ref.

    Args:
        cache_case: Engine whose broadcast transport is replaced by CPU operations.
        monkeypatch: Substitutes only device/transport boundaries, not token cleanup.
        received_pins: Temporary transport pin acquired on the received object.
    """
    engine = cache_case.engine
    engine.save_only_first_rank = True
    engine.metadata.worker_id = 1
    engine.broadcast_stream = MagicMock()
    device = MagicMock()
    device.stream.return_value = nullcontext()
    device.device_count.return_value = 1
    monkeypatch.setattr(cache_engine_module, "torch_dev", device)
    monkeypatch.setattr(cache_engine_module, "torch_device_type", "cpu")
    metadata = cache_case.memory_objs[0].metadata.to_dict()
    engine.broadcast_object_fn = MagicMock(side_effect=[1, (0, 4, metadata)])

    def copy_received_objects(
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: object,
    ) -> None:
        """Model a transport pin while observing the real broadcast-created object.

        Args:
            memory_objs: Temporary received objects passed to the GPU connector.
            starts: Start token offsets supplied by retrieve.
            ends: End token offsets supplied by retrieve.
            **kwargs: Connector options; req_id identifies this passive request.
        """
        assert starts == [0] and ends == [CHUNK_SIZE]
        assert kwargs == {"req_id": "passive"}
        _assert_counts(memory_objs, [0])
        for _ in range(received_pins):
            memory_objs[0].pin()
        _assert_counts(memory_objs, [received_pins])

    cache_case.connector.batched_to_gpu.side_effect = copy_received_objects

    assert engine.retrieve(cache_case.tokens, req_id="passive").tolist() == (
        [True] * 4 + [False] * 8
    )

    cache_case.connector.batched_to_gpu.assert_called_once()
    received = cast(
        list[MemoryObj], cache_case.connector.batched_to_gpu.call_args.args[0]
    )
    assert received[0] is not cache_case.memory_objs[0]
    _assert_counts(received, [0], refs=0)
    _assert_counts(cache_case.memory_objs, [0, 0, 0])


@pytest.mark.parametrize("pd_mode", ["sync", "async"])
def test_remove_after_retrieve_preserves_pd_ref_cleanup(
    cache_case: _CacheCase,
    monkeypatch: pytest.MonkeyPatch,
    pd_mode: Literal["sync", "async"],
) -> None:
    """PD removal keeps exactly one final reference release in either backend mode.

    Args:
        cache_case: Engine using real token processing and storage routing.
        monkeypatch: Replaces only PD-specific get/remove semantics, without transport.
        pd_mode: Sync removal leaves ref release to retrieve; async removal owns it.
    """
    engine = cache_case.engine
    engine.config.enable_pd = True
    engine.config.pd_backend_mode = pd_mode
    engine.remove_after_retrieve = True
    # PD's blocking get returns its existing buffer without a new reference.
    get_spy = MagicMock(side_effect=cache_case.backend.hot_cache.get)
    monkeypatch.setattr(cache_case.backend, "get_blocking", get_spy)

    def remove_buffer(key: CacheEngineKey) -> bool:
        """Model PD buffer deletion, including async removal's reference release."""
        memory_obj = cache_case.backend.hot_cache.pop(key)
        if pd_mode == "async":
            memory_obj.ref_count_down()
        return True

    remove_spy = MagicMock(side_effect=remove_buffer)
    monkeypatch.setattr(cache_case.backend, "remove", remove_spy)

    assert engine.retrieve(cache_case.tokens, req_id="pd").all()

    assert remove_spy.call_args_list == [call(key) for key in cache_case.keys]
    _assert_counts(cache_case.memory_objs, [0, 0, 0], refs=0)
    assert not engine.lookup_pins
