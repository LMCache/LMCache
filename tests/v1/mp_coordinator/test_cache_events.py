# SPDX-License-Identifier: Apache-2.0
"""Tests for MP-server-side cache-event emission: emitter batching and
ordering, sequence/gap semantics on publish failure, the per-adapter L2
listener, and the HTTP sink end-to-end against a coordinator app."""

# Standard
from dataclasses import asdict
import asyncio

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import L1Backend, ObjectKey, Tier
from lmcache.v1.distributed.config import (
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventEmitter,
    CacheEventPublishError,
    CacheEventSink,
    HttpCacheEventSink,
    L1CacheEventListener,
    L2CacheEventListener,
    l1_backend_name,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _entry(hash_byte: int, size_bytes: int = 0) -> CacheEventEntry:
    return CacheEventEntry(
        key=_key(hash_byte).to_encoded_object_key(), size_bytes=size_bytes
    )


class _RecordingSink(CacheEventSink):
    """Sink that records every published list; optionally fails."""

    def __init__(self) -> None:
        self.published: list[list[CacheEventBatch]] = []
        self.fail_next = False

    async def publish(self, batches: list[CacheEventBatch]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise CacheEventPublishError("injected failure")
        self.published.append(batches)


def _emitter(sink: CacheEventSink, incarnation: int = 7) -> CacheEventEmitter:
    return CacheEventEmitter(sink=sink, instance_id="node-a", incarnation=incarnation)


# -- Emitter batching and ordering -------------------------------------------


def test_flush_emits_one_batch_per_run_with_sequential_seqs():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    emitter.record(CacheEventType.DELETE, Tier.L2, "fs", [_entry(1)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.DELETE,
    ]
    assert [b.seq for b in batches] == [1, 2]
    assert all(b.instance_id == "node-a" for b in batches)
    assert all(b.incarnation == 7 for b in batches)
    assert all(b.tier == Tier.L2 for b in batches)
    assert all(b.backend == "fs" for b in batches)
    assert all(b.ts > 0 for b in batches)


def test_consecutive_same_identity_records_coalesce_into_one_batch():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(2, 200)])
    asyncio.run(emitter.flush())

    [[batch]] = sink.published
    assert batch.seq == 1
    assert [e.size_bytes for e in batch.entries] == [100, 200]


def test_interleaved_event_types_preserve_total_order():
    # store k1, delete k1, store k1 again: the re-store must not be
    # reordered before the delete, or the directory ends up empty.
    sink = _RecordingSink()
    emitter = _emitter(sink)
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    emitter.record(CacheEventType.DELETE, Tier.L2, "fs", [_entry(1)])
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 150)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.DELETE,
        CacheEventType.STORE,
    ]
    assert [b.seq for b in batches] == [1, 2, 3]


def test_backend_change_starts_a_new_batch():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    emitter.record(CacheEventType.STORE, Tier.L2, "valkey", [_entry(1, 100)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    assert [b.backend for b in batches] == ["fs", "valkey"]


def test_flush_with_empty_buffer_publishes_nothing():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    asyncio.run(emitter.flush())
    assert sink.published == []


def test_empty_record_is_a_noop():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [])
    asyncio.run(emitter.flush())
    assert sink.published == []


def test_publish_failure_drops_batches_and_leaves_a_seq_gap():
    # Failed flushes consume their seq numbers so the directory sees a
    # gap and can flag the instance for resync.
    sink = _RecordingSink()
    emitter = _emitter(sink)

    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    sink.fail_next = True
    asyncio.run(emitter.flush())
    assert sink.published == []

    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(2, 200)])
    asyncio.run(emitter.flush())
    [[batch]] = sink.published
    assert batch.seq == 2


def test_flushes_do_not_rebatch_dropped_entries():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    sink.fail_next = True
    emitter.record(CacheEventType.STORE, Tier.L2, "fs", [_entry(1, 100)])
    asyncio.run(emitter.flush())
    asyncio.run(emitter.flush())
    assert sink.published == []


# -- L2 listener mapping ------------------------------------------------------


def test_l2_listener_maps_callbacks_to_events():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    listener = L2CacheEventListener(emitter, backend="fs")

    listener.on_l2_keys_stored([_key(1), _key(2)], [100, 200])
    listener.on_l2_keys_accessed([_key(1)])
    listener.on_l2_keys_deleted([_key(2)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200]
    assert access.entries[0].key == _key(1).to_encoded_object_key()
    assert delete.entries[0].key == _key(2).to_encoded_object_key()
    assert all(b.tier == Tier.L2 and b.backend == "fs" for b in batches)


def test_l2_listener_rejects_mismatched_sizes():
    listener = L2CacheEventListener(_emitter(_RecordingSink()), backend="fs")
    with pytest.raises(ValueError):
        listener.on_l2_keys_stored([_key(1), _key(2)], [100])


# -- HTTP sink end-to-end -----------------------------------------------------


def _coordinator_transport() -> httpx.ASGITransport:
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    return httpx.ASGITransport(app=create_app(config))


def test_http_sink_feeds_the_directory_end_to_end():
    async def run() -> None:
        transport = _coordinator_transport()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://coordinator"
        ) as client:
            emitter = CacheEventEmitter(
                sink=HttpCacheEventSink(client, "http://coordinator"),
                instance_id="node-a",
                incarnation=3,
            )
            listener = L2CacheEventListener(emitter, backend="fs")
            listener.on_l2_keys_stored([_key(1), _key(2)], [100, 200])
            listener.on_l2_keys_deleted([_key(2)])
            await emitter.flush()

            resp = await client.post(
                "/directory/lookup",
                json={
                    "keys": [
                        asdict(_key(1).to_encoded_object_key()),
                        asdict(_key(2).to_encoded_object_key()),
                    ]
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            [placement] = results[0]["placements"]
            assert placement["instance_id"] == "node-a"
            assert placement["incarnation"] == 3
            assert placement["tier"] == "l2"
            assert placement["backend"] == "fs"
            assert placement["size_bytes"] == 100
            assert results[1]["placements"] == []

            stats = (await client.get("/directory/stats")).json()
            instance = stats["instances"]["node-a"]
            assert instance["last_seq"] == 2
            assert instance["gap_detected"] is False

    asyncio.run(run())


def test_http_sink_raises_publish_error_on_connect_failure():
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, text="down")
            )
        ) as client:
            sink = HttpCacheEventSink(client, "http://coordinator")
            batch = CacheEventBatch(
                instance_id="node-a",
                incarnation=1,
                seq=1,
                event_type=CacheEventType.STORE,
                tier=Tier.L2,
                backend="fs",
                entries=[_entry(1, 100)],
            )
            with pytest.raises(CacheEventPublishError):
                await sink.publish([batch])

    asyncio.run(run())


# -- L1 listener mapping ------------------------------------------------------


def _meta(size_bytes: int = 0, backend: L1Backend = L1Backend.DRAM) -> L1ObjectMeta:
    return L1ObjectMeta(size_bytes=size_bytes, backend=backend)


def test_l1_listener_maps_callbacks_to_events():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    listener = L1CacheEventListener(emitter, access_backend=L1Backend.DRAM)

    listener.on_l1_keys_write_finished([_key(1), _key(2)], [_meta(100), _meta(200)])
    listener.on_l1_keys_finish_write_and_reserve_read([_key(3)], [_meta(300)])
    listener.on_l1_keys_read_finished([_key(1)])
    listener.on_l1_keys_accessed([_key(2)])
    listener.on_l1_keys_deleted_by_manager([_key(3)], [_meta(300)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    # The two store-side callbacks are consecutive STOREs, so they
    # coalesce; the two access-side callbacks likewise.
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200, 300]
    assert len(access.entries) == 2
    assert delete.entries[0].key == _key(3).to_encoded_object_key()
    assert all(b.tier == Tier.L1 and b.backend == "dram" for b in batches)


def test_l1_listener_ignores_reservations():
    sink = _RecordingSink()
    emitter = _emitter(sink)
    listener = L1CacheEventListener(emitter, access_backend=L1Backend.DRAM)

    listener.on_l1_keys_reserved_read([_key(1)])
    listener.on_l1_keys_reserved_write([_key(2)])
    asyncio.run(emitter.flush())

    assert sink.published == []


def test_l1_listener_rejects_mismatched_metadata():
    listener = L1CacheEventListener(
        _emitter(_RecordingSink()), access_backend=L1Backend.DRAM
    )
    with pytest.raises(ValueError):
        listener.on_l1_keys_write_finished([_key(1), _key(2)], [_meta(100)])


def test_l1_listener_splits_batches_by_medium():
    """A hybrid DRAM+DAX store emits one batch per medium, and deletes
    target the same per-medium identity the stores reported."""
    sink = _RecordingSink()
    emitter = _emitter(sink)
    listener = L1CacheEventListener(emitter, access_backend=L1Backend.DRAM)

    listener.on_l1_keys_write_finished(
        [_key(1), _key(2), _key(3)],
        [
            _meta(100, L1Backend.DRAM),
            _meta(200, L1Backend.DEVDAX),
            _meta(300, L1Backend.DRAM),
        ],
    )
    listener.on_l1_keys_deleted_by_manager([_key(2)], [_meta(200, L1Backend.DEVDAX)])
    asyncio.run(emitter.flush())

    [batches] = sink.published
    assert [(b.event_type, b.backend) for b in batches] == [
        (CacheEventType.STORE, "dram"),
        (CacheEventType.STORE, "devdax"),
        (CacheEventType.DELETE, "devdax"),
    ]
    dram_store, devdax_store, devdax_delete = batches
    assert [e.size_bytes for e in dram_store.entries] == [100, 300]
    assert [e.size_bytes for e in devdax_store.entries] == [200]
    assert devdax_delete.entries[0].key == _key(2).to_encoded_object_key()
    assert devdax_delete.entries[0].size_bytes == 0
    assert all(b.tier == Tier.L1 for b in batches)


def test_l1_backend_name_by_medium():
    memory_config = L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False)
    assert l1_backend_name(L1ManagerConfig(memory_config=memory_config)) == (
        L1Backend.DRAM
    )

    devdax_config = L1MemoryManagerConfig(
        size_in_bytes=1 << 20,
        use_lazy=False,
        devdax_path="/dev/dax0.0",
        shm_name="",
    )
    assert l1_backend_name(L1ManagerConfig(memory_config=devdax_config)) == (
        L1Backend.DEVDAX
    )

    gds = GdsL1Config(file_location="/tmp/gds", size_in_bytes=1 << 20)
    assert (
        l1_backend_name(L1ManagerConfig(memory_config=memory_config, gds_l1_config=gds))
        == L1Backend.GDS
    )
