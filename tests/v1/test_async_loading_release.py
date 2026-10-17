# SPDX-License-Identifier: Apache-2.0
"""Async-loaded memory objects must be released exactly once.

With async loading, ``retrieve()`` releases the objects it loads (unpin +
``ref_count_down``). If the LOADING event stays in the event manager,
``lookup_unpin()`` -> ``cleanup_memory_objs()`` releases them a second time when
the request finishes:

- an object loaded from a peer (P2P) goes to ref count -1;
- an object served from the local CPU cache loses the cache's own reference and
  is freed while still in ``hot_cache``, so a later hit reads reused memory.

The objects below count pins and references; the release loop at the end of
``LMCacheEngine.retrieve()`` is reproduced as written there.
"""

# Standard
from types import SimpleNamespace
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.storage_backend.storage_manager import StorageManager

CHUNK = 256


class CountingMemoryObj:
    def __init__(self, name: str, ref: int, pin: int, in_cache: bool) -> None:
        self.name = name
        self.ref = ref
        self.pin_count = pin
        self.in_cache = in_cache
        self.went_negative = False
        self.freed = False

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0

    def pin(self) -> None:
        self.pin_count += 1

    def unpin(self) -> None:
        self.pin_count -= 1
        self.went_negative |= self.pin_count < 0

    def ref_count_down(self) -> None:
        self.ref -= 1
        self.went_negative |= self.ref < 0
        if self.ref <= 0 and self.pin_count <= 0:
            self.freed = True

    def get_size(self) -> int:
        return 1


def local_cpu_hit(i: int) -> CountingMemoryObj:
    # held by hot_cache (ref 1), pinned by batched_async_contains,
    # referenced again by batched_get_non_blocking
    return CountingMemoryObj(f"local{i}", ref=2, pin=1, in_cache=True)


def p2p_hit(i: int) -> CountingMemoryObj:
    # allocated (ref 1) and pinned by P2PBackend.batched_get_non_blocking
    return CountingMemoryObj(f"p2p{i}", ref=1, pin=1, in_cache=False)


def key(i: int) -> CacheEngineKey:
    return CacheEngineKey("model", 1, 0, i, torch.bfloat16)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def make_engine(event_manager: EventManager) -> LMCacheEngine:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.event_manager = event_manager
    engine.async_loading = True
    engine.lookup_pins = {}
    engine.storage_manager = None
    engine.token_database = SimpleNamespace(
        process_tokens=lambda tokens, mask, request_configs: [
            (i * CHUNK, (i + 1) * CHUNK, key(i)) for i in range(len(tokens) // CHUNK)
        ]
    )
    return engine


def add_done_loading_event(em, loop, req_id, result) -> asyncio.Future:
    fut = loop.create_future()
    fut.set_result(result)
    em.add_event(EventType.LOADING, req_id, fut)
    em.update_event_status(EventType.LOADING, req_id, status=EventStatus.DONE)
    return fut


def retrieve(engine: LMCacheEngine, num_chunks: int, req_id: str = "req") -> None:
    tokens = list(range(num_chunks * CHUNK))
    ret_mask = torch.zeros(len(tokens), dtype=torch.bool)
    chunks, _ = engine._async_process_tokens_internal(
        tokens, None, ret_mask, req_id=req_id
    )
    for _, memory_obj, _, _ in chunks:  # as at the end of LMCacheEngine.retrieve()
        if engine.async_loading and memory_obj.is_pinned:
            memory_obj.unpin()
        memory_obj.ref_count_down()


def assert_released_once(objs) -> None:
    for obj in objs:
        assert not obj.went_negative, f"{obj.name} released twice"
        if obj.in_cache:
            assert (obj.ref, obj.pin_count, obj.freed) == (1, 0, False), (
                f"{obj.name}: ref={obj.ref} pin={obj.pin_count} freed={obj.freed}; "
                "expected only the cache's own reference"
            )
        else:
            assert (obj.ref, obj.pin_count, obj.freed) == (0, 0, True), (
                f"{obj.name}: ref={obj.ref} pin={obj.pin_count} freed={obj.freed}"
            )


@pytest.mark.parametrize("make_obj", [local_cpu_hit, p2p_hit])
def test_retrieved_objects_are_released_once(loop, make_obj):
    em = EventManager()
    objs = [make_obj(i) for i in range(3)]
    add_done_loading_event(em, loop, "req", [[(key(i), o) for i, o in enumerate(objs)]])
    engine = make_engine(em)

    retrieve(engine, 3)
    engine.lookup_unpin("req")  # wait_for_save at the end of the request

    assert_released_once(objs)
    assert em.get_event_status(EventType.LOADING, "req") == EventStatus.NOT_FOUND


def test_unused_prefetched_objects_return_pin_and_ref(loop):
    em = EventManager()
    objs = [local_cpu_hit(0), local_cpu_hit(1), p2p_hit(2)]
    add_done_loading_event(em, loop, "req", [[(key(i), o) for i, o in enumerate(objs)]])
    engine = make_engine(em)

    retrieve(engine, 1)  # the request only needs the first chunk
    engine.lookup_unpin("req")

    assert_released_once(objs)


def test_never_retrieved_prefetch_is_released_by_cleanup(loop):
    em = EventManager()
    objs = [local_cpu_hit(0), p2p_hit(1)]
    add_done_loading_event(em, loop, "req", [[(key(i), o) for i, o in enumerate(objs)]])
    engine = make_engine(em)

    engine.lookup_unpin("req")  # e.g. aborted before the load ran

    assert_released_once(objs)


def test_prefetch_gap_releases_dropped_objects_once(loop):
    # tier 0 expected 2 chunks but returned 1, so tier 1 cannot be used
    objs = [local_cpu_hit(0), p2p_hit(2), p2p_hit(3)]
    res = [[(key(0), objs[0])], [(key(2), objs[1]), (key(3), objs[2])]]
    storage_manager = StorageManager.__new__(StorageManager)
    storage_manager.event_manager = EventManager()
    storage_manager.async_lookup_server = SimpleNamespace(
        send_response_to_scheduler=lambda *args: None
    )
    fut = loop.create_future()
    fut.set_result(res)
    storage_manager.event_manager.add_event(EventType.LOADING, "req", fut)
    storage_manager.prefetch_all_done_callback(
        fut, "req", [0, 256, 512, 768, 1024], [2, 2], keys_per_chunk=1
    )
    engine = make_engine(storage_manager.event_manager)

    retrieve(engine, 4)
    engine.lookup_unpin("req")

    assert_released_once(objs)
