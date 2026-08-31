# SPDX-License-Identifier: Apache-2.0
"""Tests for the fleet usage view (per-salt and per-instance byte totals,
per tier, derived from the gate-admitted cache-event stream)."""

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager


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


def _l1_batch(keys: list[ObjectKey], **kwargs) -> CacheEventBatch:
    kwargs.setdefault("backend", "dram")
    return _batch(keys, tier=Tier.L1, **kwargs)


# -- The tenant axis, on either tier ----------------------------------------


def test_aggregates_bytes_per_salt():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1, "alice")], size_bytes=100))
    usage.consume(_batch([_key(2, "alice"), _key(3, "bob")], size_bytes=200))
    assert usage.get_salt_bytes(Tier.L2, "alice") == 300
    assert usage.get_salt_bytes(Tier.L2, "bob") == 200
    assert usage.get_bytes_by_salt(Tier.L2) == {"alice": 300, "bob": 200}
    assert usage.get_total_bytes(Tier.L2) == 500


def test_ignores_access_events():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], event_type=CacheEventType.ACCESS, backend=""))
    assert usage.get_total_bytes(Tier.L2) == 0
    assert usage.get_bytes_by_salt(Tier.L2) == {}


def test_restore_adjusts_delta():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], size_bytes=100))
    usage.consume(_batch([_key(1)], size_bytes=250))
    assert usage.get_salt_bytes(Tier.L2, "") == 250
    assert usage.get_total_bytes(Tier.L2) == 250


def test_delete_subtracts_and_cleans_bucket():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1, "alice")], size_bytes=100))
    usage.consume(_batch([_key(1, "alice")], event_type=CacheEventType.DELETE))
    assert usage.get_salt_bytes(Tier.L2, "alice") == 0
    assert usage.get_bytes_by_salt(Tier.L2) == {}
    assert usage.get_total_bytes(Tier.L2) == 0


def test_counts_each_private_copy_but_shared_once():
    """Two private copies occupy bytes twice; N reporters of one shared
    pool describe a single placement, counted once."""
    usage = CacheUsageManager()
    for node in ("node-a", "node-b"):
        usage.consume(_batch([_key(1)], instance_id=node, backend="fs"))
        usage.consume(_batch([_key(1)], instance_id=node, backend="s3", shared=True))
    assert usage.get_total_bytes(Tier.L2) == 300  # 2 private copies + 1 shared


def test_delete_from_any_reporter_removes_shared_bytes():
    usage = CacheUsageManager()
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
    assert usage.get_total_bytes(Tier.L2) == 0


def test_delete_of_untracked_placement_is_noop():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], event_type=CacheEventType.DELETE))
    assert usage.get_total_bytes(Tier.L2) == 0


def test_get_key_bytes_sums_the_keys_placements():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], backend="fs", size_bytes=100))
    usage.consume(_batch([_key(1)], backend="s3", shared=True, size_bytes=150))
    assert usage.get_key_bytes(Tier.L2, _key(1)) == 250
    assert usage.get_key_bytes(Tier.L2, _key(9)) == 0


# -- Tier isolation ----------------------------------------------------------


def test_tiers_are_accounted_separately():
    """One key resident in both tiers holds bytes in both; no read
    conflates them."""
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1, "alice")], size_bytes=100))
    usage.consume(_l1_batch([_key(1, "alice")], size_bytes=40))
    assert usage.get_key_bytes(Tier.L2, _key(1, "alice")) == 100
    assert usage.get_key_bytes(Tier.L1, _key(1, "alice")) == 40
    assert usage.get_salt_bytes(Tier.L2, "alice") == 100
    assert usage.get_salt_bytes(Tier.L1, "alice") == 40
    assert usage.get_total_bytes(Tier.L2) == 100
    assert usage.get_total_bytes(Tier.L1) == 40


def test_l1_delete_leaves_l2_bytes_intact():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], size_bytes=100))
    usage.consume(_l1_batch([_key(1)], size_bytes=40))
    usage.consume(_l1_batch([_key(1)], event_type=CacheEventType.DELETE))
    assert usage.get_total_bytes(Tier.L1) == 0
    assert usage.get_total_bytes(Tier.L2) == 100


# -- The capacity axis -------------------------------------------------------


def test_aggregates_l1_bytes_per_instance_and_backend():
    usage = CacheUsageManager()
    usage.consume(_l1_batch([_key(1)], instance_id="node-a", size_bytes=100))
    usage.consume(
        _l1_batch([_key(2)], instance_id="node-a", backend="gds", size_bytes=30)
    )
    usage.consume(_l1_batch([_key(3)], instance_id="node-b", size_bytes=70))
    assert usage.get_bytes_by_instance(Tier.L1) == {
        "node-a": {"dram": 100, "gds": 30},
        "node-b": {"dram": 70},
    }


def test_shared_pool_bytes_belong_to_no_instance():
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], instance_id="node-a", backend="fs"))
    usage.consume(_batch([_key(2)], instance_id="node-a", backend="s3", shared=True))
    assert usage.get_bytes_by_instance(Tier.L2) == {
        "node-a": {"fs": 100},
        "": {"s3": 100},
    }


def test_instance_view_drops_emptied_backends():
    usage = CacheUsageManager()
    usage.consume(_l1_batch([_key(1)], size_bytes=100))
    usage.consume(_l1_batch([_key(1)], event_type=CacheEventType.DELETE))
    assert usage.get_bytes_by_instance(Tier.L1) == {}


# -- Fencing -----------------------------------------------------------------


def test_fence_instance_drops_its_l1_and_keeps_its_l2():
    """A fenced reporter's L1 was its own memory and died with it; its
    L2 sits on storage the fleet shares."""
    usage = CacheUsageManager()
    usage.consume(_l1_batch([_key(1)], instance_id="node-a", size_bytes=100))
    usage.consume(_batch([_key(1)], instance_id="node-a", size_bytes=250))
    usage.consume(_l1_batch([_key(2)], instance_id="node-b", size_bytes=70))

    usage.fence_instance("node-a")

    assert usage.get_total_bytes(Tier.L1) == 70
    assert usage.get_bytes_by_instance(Tier.L1) == {"node-b": {"dram": 70}}
    assert usage.get_key_bytes(Tier.L1, _key(1)) == 0
    assert usage.get_total_bytes(Tier.L2) == 250


def test_fence_instance_spares_shared_pools():
    """A shared pool outlives any one member's departure."""
    usage = CacheUsageManager()
    usage.consume(_batch([_key(1)], instance_id="node-a", backend="s3", shared=True))
    usage.fence_instance("node-a")
    assert usage.get_total_bytes(Tier.L2) == 100


def test_fence_instance_spares_a_shared_l1_pool():
    """Sparing a shared pool is per placement, not per tier: a pool
    several instances mount outlives any one of them on L1 too."""
    usage = CacheUsageManager()
    usage.consume(
        _l1_batch([_key(1)], instance_id="node-a", backend="cxl", shared=True)
    )
    usage.consume(_l1_batch([_key(2)], instance_id="node-a", size_bytes=50))

    usage.fence_instance("node-a")

    assert usage.get_total_bytes(Tier.L1) == 100
    assert usage.get_bytes_by_instance(Tier.L1) == {"": {"cxl": 100}}


def test_fence_instance_is_idempotent():
    usage = CacheUsageManager()
    usage.consume(_l1_batch([_key(1)], size_bytes=100))
    usage.fence_instance("node-a")
    usage.fence_instance("node-a")
    usage.fence_instance("never-seen")
    assert usage.get_total_bytes(Tier.L1) == 0


def test_restore_after_fence_does_not_double_count():
    """The fenced instance re-reports the same key: its bytes count once,
    and a later delete still clears them."""
    usage = CacheUsageManager()
    usage.consume(_l1_batch([_key(1)], size_bytes=100))
    usage.fence_instance("node-a")
    usage.consume(_l1_batch([_key(1)], size_bytes=100))
    assert usage.get_total_bytes(Tier.L1) == 100

    usage.consume(_l1_batch([_key(1)], event_type=CacheEventType.DELETE))
    assert usage.get_total_bytes(Tier.L1) == 0
    assert usage.get_bytes_by_instance(Tier.L1) == {}
