# SPDX-License-Identifier: Apache-2.0
"""Tests for the L2 usage view (per-salt byte totals derived from the
gate-admitted cache-event stream by the owning eviction manager)."""

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.usage_manager import L2UsageManager


def _key(hash_byte: int, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([hash_byte]) * 4,
        model_name="m",
        kv_rank=0,
        cache_salt=salt,
    )


def _batch(
    keys: list[ObjectKey],
    event_type: CacheEventType = CacheEventType.STORE,
    instance_id: str = "node-a",
    tier: Tier = Tier.L2,
    backend: str = "fs",
    shared: bool = False,
    size_bytes: int = 100,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=1,
        seq=1,
        event_type=event_type,
        tier=tier,
        backend=backend,
        shared=shared,
        entries=[
            CacheEventEntry(key=k.to_encoded_object_key(), size_bytes=size_bytes)
            for k in keys
        ],
    )


def test_aggregates_bytes_per_salt():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1, "alice")], size_bytes=100))
    usage.consume(_batch([_key(2, "alice"), _key(3, "bob")], size_bytes=200))
    assert usage.get("alice") == 300
    assert usage.get("bob") == 200
    assert usage.get_all() == {"alice": 300, "bob": 200}
    assert usage.get_total() == 500


def test_ignores_l1_batches_and_access():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1)], tier=Tier.L1, backend="dram"))
    usage.consume(_batch([_key(1)], event_type=CacheEventType.ACCESS))
    assert usage.get_total() == 0
    assert usage.get_all() == {}


def test_restore_adjusts_delta():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1)], size_bytes=100))
    usage.consume(_batch([_key(1)], size_bytes=250))
    assert usage.get("") == 250
    assert usage.get_total() == 250


def test_delete_subtracts_and_cleans_bucket():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1, "alice")], size_bytes=100))
    usage.consume(_batch([_key(1, "alice")], event_type=CacheEventType.DELETE))
    assert usage.get("alice") == 0
    assert usage.get_all() == {}
    assert usage.get_total() == 0


def test_counts_each_private_copy_but_shared_once():
    """Two private copies occupy bytes twice; N reporters of one shared
    pool describe a single placement, counted once."""
    usage = L2UsageManager()
    for node in ("node-a", "node-b"):
        usage.consume(_batch([_key(1)], instance_id=node, backend="fs"))
        usage.consume(_batch([_key(1)], instance_id=node, backend="s3", shared=True))
    assert usage.get_total() == 300  # 2 private copies + 1 shared


def test_delete_from_any_reporter_removes_shared_bytes():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1)], instance_id="node-a", backend="s3", shared=True))
    usage.consume(
        _batch(
            [_key(1)],
            instance_id="node-b",
            backend="s3",
            shared=True,
            event_type=CacheEventType.DELETE,
        )
    )
    assert usage.get_total() == 0


def test_delete_of_untracked_placement_is_noop():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1)], event_type=CacheEventType.DELETE))
    assert usage.get_total() == 0


def test_get_key_size_sums_the_keys_placements():
    usage = L2UsageManager()
    usage.consume(_batch([_key(1)], backend="fs", size_bytes=100))
    usage.consume(_batch([_key(1)], backend="s3", shared=True, size_bytes=150))
    assert usage.get_key_size(_key(1)) == 250
    assert usage.get_key_size(_key(9)) == 0
