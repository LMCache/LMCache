# SPDX-License-Identifier: Apache-2.0
"""Blend proactive fingerprint eviction: the tracker's rule for when a
chunk lost a key for good, and the blend module acting on storage key
events from the event bus."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.modules.blend.fingerprint_eviction import (
    FingerprintEvictionTracker,
)
from lmcache.v1.multiprocess.modules.blend.module import BlendModule

CHUNK_SIZE = 4
HASH_A = ObjectKey.IntHash2Bytes(101)
HASH_B = ObjectKey.IntHash2Bytes(202)


def _key(chunk_hash: bytes, group: int = 0) -> ObjectKey:
    """One key (row) of the chunk ``chunk_hash``."""
    return ObjectKey(chunk_hash, "model", 0, group)


# =============================================================================
# Tracker: no L2 adapter
# =============================================================================


class TestTrackerWithoutL2:
    def test_evicting_the_only_l1_copy_reports_the_chunk(self) -> None:
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")

        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 0) == [HASH_A]

    def test_one_evicted_key_is_enough_for_a_multi_key_chunk(self) -> None:
        """A chunk with two object groups is gone once either key is."""
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A, 0), _key(HASH_A, 1)], "dram")

        assert tracker.on_l1_evicted([_key(HASH_A, 1)], "dram", 0) == [HASH_A]

    def test_keys_of_one_chunk_report_it_once(self) -> None:
        tracker = FingerprintEvictionTracker()
        keys = [_key(HASH_A, 0), _key(HASH_A, 1), _key(HASH_B)]
        tracker.on_l1_committed(keys, "dram")

        assert tracker.on_l1_evicted(keys, "dram", 0) == [HASH_A, HASH_B]

    def test_a_copy_in_another_l1_keeps_the_chunk(self) -> None:
        """With two L1 managers the key is gone only with its last copy."""
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")
        tracker.on_l1_committed([_key(HASH_A)], "devdax")

        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 0) == []
        assert tracker.on_l1_evicted([_key(HASH_A)], "devdax", 0) == [HASH_A]

    def test_an_unrecorded_copy_is_reported(self) -> None:
        """After ``clear`` the record is empty; an eviction still names a
        copy that left L1."""
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")
        tracker.clear()

        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 0) == [HASH_A]


# =============================================================================
# Tracker: with L2
# =============================================================================


class TestTrackerWithL2:
    def test_l1_eviction_alone_keeps_the_chunk(self) -> None:
        """L2 may still serve the key, so an L1 eviction decides nothing."""
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")

        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 1) == []

    def test_l2_delete_of_a_key_not_in_l1_reports_the_chunk(self) -> None:
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")
        tracker.on_l1_evicted([_key(HASH_A)], "dram", 1)

        assert tracker.on_l2_deleted([_key(HASH_A)], 1) == [HASH_A]

    def test_l2_delete_waits_for_the_l1_copy_to_go(self) -> None:
        """L1 still serves the key after the L2 delete; the chunk is gone
        only once L1 evicts it too."""
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")

        assert tracker.on_l2_deleted([_key(HASH_A)], 1) == []
        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 1) == [HASH_A]

    def test_a_new_l2_store_cancels_the_earlier_delete(self) -> None:
        tracker = FingerprintEvictionTracker()
        tracker.on_l1_committed([_key(HASH_A)], "dram")
        tracker.on_l2_deleted([_key(HASH_A)], 1)
        tracker.on_l2_stored([_key(HASH_A)])

        assert tracker.on_l1_evicted([_key(HASH_A)], "dram", 1) == []

    @pytest.mark.parametrize("num_l2_adapters", [0, 2])
    def test_l2_delete_is_ignored_unless_one_adapter(
        self, num_l2_adapters: int
    ) -> None:
        """With several adapters another one may still hold the key."""
        tracker = FingerprintEvictionTracker()

        assert tracker.on_l2_deleted([_key(HASH_A)], num_l2_adapters) == []


# =============================================================================
# Blend module: storage key events remove fingerprints
# =============================================================================


def _chunk_tokens(index: int) -> list[int]:
    """Distinct token content for chunk ``index``."""
    return list(range(index * 100, index * 100 + CHUNK_SIZE))


@pytest.fixture
def blend():
    """A blend module on an enabled, not started event bus with no L2
    adapter, holding fingerprints for chunks A and B. ``bus.stop()`` drains
    the published events on the calling thread."""
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus = EventBus(EventBusConfig(enabled=True))
    ctx.session_manager.destroy_listeners = []
    ctx.storage_manager.l2_adapters.return_value = []
    module = BlendModule(ctx, MagicMock())
    matcher = module._token_range_matcher
    matcher.on_new_token_hashes(_chunk_tokens(1), [HASH_A])
    matcher.on_new_token_hashes(_chunk_tokens(2), [HASH_B])
    yield module
    module.close()


def _matched(module: BlendModule) -> set[bytes]:
    """The chunk hashes the module's matcher still finds for A and B."""
    matcher = module._token_range_matcher
    return {
        match.hash
        for index in (1, 2)
        for match in matcher.match_sub_sequence(_chunk_tokens(index))
    }


def _l1_event(event_type: EventType, keys: list[ObjectKey]) -> Event:
    return Event(event_type=event_type, metadata={"l1_tag": "dram", "keys": keys})


def _l2_event(event_type: EventType, keys: list[ObjectKey]) -> Event:
    return Event(event_type=event_type, metadata={"keys": keys})


class TestBlendModule:
    def test_l1_eviction_removes_the_fingerprint_without_l2(self, blend) -> None:
        bus = blend.context.event_bus
        evicted: list[int] = []
        bus.subscribe(
            EventType.CB_CHUNKS_EVICTED,
            lambda event: evicted.append(event.metadata["num_chunks"]),
        )

        bus.publish(_l1_event(EventType.L1_WRITE_FINISHED, [_key(HASH_A)]))
        bus.publish(_l1_event(EventType.L1_KEYS_EVICTED, [_key(HASH_A)]))
        bus.stop()

        assert _matched(blend) == {HASH_B}
        assert evicted == [1]

    def test_l1_eviction_keeps_the_fingerprint_with_l2(self, blend) -> None:
        blend.context.storage_manager.l2_adapters.return_value = [MagicMock()]
        bus = blend.context.event_bus

        bus.publish(_l1_event(EventType.L1_WRITE_FINISHED, [_key(HASH_A)]))
        bus.publish(_l1_event(EventType.L1_KEYS_EVICTED, [_key(HASH_A)]))
        bus.stop()

        assert _matched(blend) == {HASH_A, HASH_B}

    def test_l2_delete_then_l1_eviction_removes_the_fingerprint(self, blend) -> None:
        blend.context.storage_manager.l2_adapters.return_value = [MagicMock()]
        bus = blend.context.event_bus

        bus.publish(
            _l1_event(EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, [_key(HASH_A)])
        )
        bus.publish(_l2_event(EventType.L2_KEYS_STORED, [_key(HASH_A)]))
        bus.publish(_l2_event(EventType.L2_KEYS_DELETED, [_key(HASH_A)]))
        bus.stop()
        assert _matched(blend) == {HASH_A, HASH_B}

        bus.publish(_l1_event(EventType.L1_KEYS_EVICTED, [_key(HASH_A)]))
        bus.stop()
        assert _matched(blend) == {HASH_B}

    def test_a_chunk_being_stored_keeps_its_fingerprint(self, blend) -> None:
        """A hash whose registration is still queued belongs to a store
        that has just written the chunk."""
        bus = blend.context.event_bus
        blend._pending_fp_hashes.add(HASH_A)

        bus.publish(_l1_event(EventType.L1_KEYS_EVICTED, [_key(HASH_A), _key(HASH_B)]))
        bus.stop()

        assert _matched(blend) == {HASH_A}

    def test_a_key_blend_never_fingerprinted_is_ignored(self, blend) -> None:
        bus = blend.context.event_bus
        evicted: list[Event] = []
        bus.subscribe(EventType.CB_CHUNKS_EVICTED, evicted.append)

        other = _key(ObjectKey.IntHash2Bytes(303))
        bus.publish(_l1_event(EventType.L1_KEYS_EVICTED, [other]))
        bus.stop()

        assert _matched(blend) == {HASH_A, HASH_B}
        assert evicted == []
