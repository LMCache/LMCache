# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's HTTP cache-event source."""

# Standard
import asyncio

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.event_source import EventReplayCapability
from lmcache.v1.mp_coordinator.ingest.http_event_source import HttpCacheEventSource
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock


class _RecordingConsumer:
    """Consumer that records admitted batches."""

    def __init__(self) -> None:
        self.batches: list[CacheEventBatch] = []

    def consume(self, batch: CacheEventBatch) -> None:
        """Record an admitted batch."""
        self.batches.append(batch)

    def fence_instance(self, instance_id: str) -> None:
        """Accept an instance fence without storing state."""


def _source() -> tuple[HttpCacheEventSource, _RecordingConsumer]:
    consumer = _RecordingConsumer()
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(consumer)
    gate = EventGate(broadcaster, QuiesceLock())
    return HttpCacheEventSource(gate), consumer


def _batch() -> CacheEventBatch:
    key = ObjectKey(chunk_hash=b"hash", model_name="model", kv_rank=0)
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L1,
        backend="dram",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=1024)],
    )


def test_http_source_delegates_to_event_gate() -> None:
    source, consumer = _source()
    batch = _batch()

    summary = source.ingest([batch])

    assert summary.applied == 1
    assert summary.duplicates == 0
    assert summary.stale == 0
    assert consumer.batches == [batch]


def test_http_source_reports_no_replay_capability() -> None:
    source, _ = _source()

    asyncio.run(source.start())
    status = source.status()
    asyncio.run(source.stop())

    assert status.source_name == "http"
    assert status.replay_capability == EventReplayCapability.NONE
