# SPDX-License-Identifier: Apache-2.0
"""RetentionManager: deadlines, budget, expiry, and eviction eligibility."""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.retention_manager import RetentionManager


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=i.to_bytes(8, "little"),
        model_name="test_model",
        kv_rank=0,
    )


def test_stamped_key_is_shielded_until_deadline():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=1000, clock=clock)

    assert manager.is_evictable(_key(1))
    manager.note_stored([_key(1)], [100], ttl_sec=300)
    assert not manager.is_evictable(_key(1))

    clock.advance(299)
    assert not manager.is_evictable(_key(1))
    clock.advance(2)
    # Expired keys are evictable immediately, before any sweep runs.
    assert manager.is_evictable(_key(1))


def test_extend_only_never_shortens():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=1000, clock=clock)

    manager.note_stored([_key(1)], [100], ttl_sec=3600)
    manager.note_stored([_key(1)], [100], ttl_sec=300)
    clock.advance(600)
    # The later, shorter ttl must not have shortened the 1h window.
    assert not manager.is_evictable(_key(1))

    manager.note_stored([_key(1)], [100], ttl_sec=7200)
    clock.advance(3600)
    # And a longer ttl extends it.
    assert not manager.is_evictable(_key(1))


def test_extend_is_budget_free():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=100, clock=clock)

    assert manager.note_stored([_key(1)], [100], ttl_sec=300) == 1
    # Budget is full; extending the same key must still succeed.
    assert manager.note_stored([_key(1)], [100], ttl_sec=600) == 1
    status = manager.report_status()
    assert status["retained_bytes"] == 100
    assert status["extends"] == 1
    assert status["budget_rejections"] == 0


def test_budget_rejects_new_keys_but_store_semantics_unchanged():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=150, clock=clock)

    accepted = manager.note_stored([_key(1), _key(2)], [100, 100], ttl_sec=300)
    assert accepted == 1
    assert not manager.is_evictable(_key(1))
    # The rejected key is simply not shielded.
    assert manager.is_evictable(_key(2))
    assert manager.report_status()["budget_rejections"] == 1


def test_zero_budget_disables_retention():
    manager = RetentionManager(max_retained_bytes=0, clock=FakeClock())
    assert manager.note_stored([_key(1)], [100], ttl_sec=300) == 0
    assert manager.is_evictable(_key(1))


def test_sweep_frees_budget_for_new_keys():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=100, clock=clock)

    manager.note_stored([_key(1)], [100], ttl_sec=300)
    assert manager.note_stored([_key(2)], [100], ttl_sec=300) == 0

    clock.advance(301)
    assert manager.sweep() == 1
    assert manager.note_stored([_key(2)], [100], ttl_sec=300) == 1
    status = manager.report_status()
    assert status["retained_keys"] == 1
    assert status["retained_bytes"] == 100
    assert status["expirations"] == 1


def test_forget_releases_budget():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=100, clock=clock)

    manager.note_stored([_key(1)], [100], ttl_sec=300)
    manager.forget([_key(1), _key(2)])
    assert manager.is_evictable(_key(1))
    assert manager.report_status()["retained_bytes"] == 0
    assert manager.note_stored([_key(2)], [100], ttl_sec=300) == 1


def test_nonpositive_ttl_is_a_noop():
    manager = RetentionManager(max_retained_bytes=1000, clock=FakeClock())
    assert manager.note_stored([_key(1)], [100], ttl_sec=0) == 0
    assert manager.note_stored([_key(1)], [100], ttl_sec=-5) == 0
    assert manager.report_status()["retained_keys"] == 0


def test_sweep_at_original_deadline_spares_extended_key():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=1000, clock=clock)

    manager.note_stored([_key(1)], [100], ttl_sec=10)
    clock.advance(5)
    manager.note_stored([_key(1)], [100], ttl_sec=100)

    clock.advance(10)
    assert manager.sweep() == 0
    assert not manager.is_evictable(_key(1))

    clock.advance(100)
    assert manager.sweep() == 1
    assert manager.is_evictable(_key(1))


def test_deadline_index_stays_in_lockstep_with_entries():
    clock = FakeClock()
    manager = RetentionManager(max_retained_bytes=10_000, clock=clock)

    manager.note_stored([_key(i) for i in range(5)], [100] * 5, ttl_sec=10)
    manager.note_stored([_key(i) for i in range(3)], [100] * 3, ttl_sec=50)
    manager.forget([_key(3)])
    assert len(manager._deadlines) == len(manager._entries) == 4

    clock.advance(11)
    assert manager.sweep() == 1
    assert len(manager._deadlines) == len(manager._entries) == 3

    clock.advance(50)
    assert manager.sweep() == 3
    assert len(manager._deadlines) == len(manager._entries) == 0
