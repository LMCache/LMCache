# SPDX-License-Identifier: Apache-2.0
"""Tests for the cache-event broadcaster (consumer registration + fan-out)."""

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.cache_control.event_broadcaster import (
    CacheEventBroadcaster,
)


class _RecordingConsumer:
    def __init__(self, name: str, log: list[tuple[str, CacheEventBatch]]) -> None:
        self._name = name
        self._log = log

    def consume(self, batch: CacheEventBatch) -> None:
        self._log.append((self._name, batch))


def _batch(seq: int = 1) -> CacheEventBatch:
    key = ObjectKey(chunk_hash=b"\xaa" * 4, model_name="m", kv_rank=0)
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=1)],
    )


def test_route_fans_out_to_consumers_in_registration_order():
    log: list[tuple[str, CacheEventBatch]] = []
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(_RecordingConsumer("first", log))
    broadcaster.register_consumer(_RecordingConsumer("second", log))

    batch = _batch()
    broadcaster.broadcast(batch)

    assert log == [("first", batch), ("second", batch)]


def test_route_with_no_consumers_is_a_noop():
    CacheEventBroadcaster().broadcast(_batch())


def test_consumer_registered_later_sees_only_later_batches():
    log: list[tuple[str, CacheEventBatch]] = []
    broadcaster = CacheEventBroadcaster()
    first, second = _batch(seq=1), _batch(seq=2)
    broadcaster.broadcast(first)
    broadcaster.register_consumer(_RecordingConsumer("late", log))
    broadcaster.broadcast(second)

    assert log == [("late", second)]
