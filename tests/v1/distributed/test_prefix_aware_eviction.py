# SPDX-License-Identifier: Apache-2.0
"""Regression test: a prefix stored across several batches must evict tail-first.

`on_keys_created` reverses within a single call so that "the later keys should be
evicted first" (its own comment). But a serving engine stores a long prefix in
several calls -- vLLM's chunked prefill emits one store per
`max_num_batched_tokens / chunk_size` chunks -- and across calls the earliest
batch stays oldest, so eviction sheds the prefix HEAD.

That is the one thing a prefix cache cannot lose: a match always starts at chunk 0,
so losing the head invalidates every chunk still resident.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import MAX_TRACKED_POSITIONS


def make_key(chunk_hash: int, model: str = "model", kv_rank: int = 0) -> ObjectKey:
    """Create an ObjectKey for testing."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model, kv_rank=kv_rank)


class TestPrefixOrderingAcrossStoreBatches:
    """A prefix stored in batches must still evict from its tail."""

    def test_single_batch_evicts_tail_first(self):
        """Baseline: within one call the existing reversal already does this."""
        policy = LRUEvictionPolicy()
        keys = [make_key(i) for i in range(6)]
        policy.note_key_positions(keys, list(range(6)))
        policy.on_keys_created(keys)

        candidates = policy.get_eviction_candidates(6)
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 5

    def test_multi_batch_evicts_tail_first(self):
        """A prefix split across two stores must evict position 5, not position 0."""
        policy = LRUEvictionPolicy()
        first = [make_key(i) for i in range(3)]  # chunks 0,1,2
        second = [make_key(i) for i in range(3, 6)]  # chunks 3,4,5

        policy.note_key_positions(first, [0, 1, 2])
        policy.on_keys_created(first)
        policy.note_key_positions(second, [3, 4, 5])
        policy.on_keys_created(second)

        candidates = policy.get_eviction_candidates(6)
        evicted = [ObjectKey.Bytes2IntHash(c.chunk_hash) for c in candidates]
        assert evicted[0] == 5, (
            "eviction must start at the prefix TAIL; got %r. Starting at 0 means the "
            "head was shed, which invalidates the whole cached prefix." % evicted
        )
        # the head must be the very last thing to go
        assert evicted[-1] == 0, evicted

    def test_head_survives_partial_eviction(self):
        """After evicting half a batched prefix, the survivors must start at chunk 0."""
        policy = LRUEvictionPolicy()
        for start in (0, 3, 6):
            batch = [make_key(i) for i in range(start, start + 3)]
            policy.note_key_positions(batch, list(range(start, start + 3)))
            policy.on_keys_created(batch)

        actions = policy.get_eviction_actions(0.5)
        evicted = {
            ObjectKey.Bytes2IntHash(k.chunk_hash) for a in actions for k in a.keys
        }
        survivors = sorted(set(range(9)) - evicted)
        assert survivors, "eviction removed everything"
        assert survivors[0] == 0, (
            "surviving prefix must start at chunk 0; got %r" % survivors
        )
        assert survivors == list(range(len(survivors))), (
            "surviving prefix must be contiguous from 0; got %r" % survivors
        )

    def test_unrelated_prefix_does_not_disturb_lru(self):
        """A second prefix restarting at position 0 must not be treated as a
        continuation of the first, or ordinary cross-request LRU breaks."""
        policy = LRUEvictionPolicy()
        first = [make_key(i) for i in range(3)]
        policy.note_key_positions(first, [0, 1, 2])
        policy.on_keys_created(first)

        second = [make_key(100 + i) for i in range(3)]
        policy.note_key_positions(second, [0, 1, 2])
        policy.on_keys_created(second)

        assert policy.get_num_tracked_keys() == 6

    def test_multi_rank_keys_evict_tail_first(self):
        """Tensor parallel: each chunk expands to one key per kv_rank, and the
        expansion is chunk-major / rank-minor. Positions must be per CHUNK, not per
        key, or a batched prefix is mis-ordered on any world_size > 1."""
        policy = LRUEvictionPolicy()
        ranks = (0, 1)

        def batch(chunks):
            keys, pos = [], []
            for c in chunks:
                for r in ranks:
                    keys.append(make_key(c, kv_rank=r))
                    pos.append(c)
            return keys, pos

        first_keys, first_pos = batch([0, 1, 2])
        policy.note_key_positions(first_keys, first_pos)
        policy.on_keys_created(first_keys)
        second_keys, second_pos = batch([3, 4, 5])
        policy.note_key_positions(second_keys, second_pos)
        policy.on_keys_created(second_keys)

        candidates = policy.get_eviction_candidates(len(ranks) * 6)
        evicted = [ObjectKey.Bytes2IntHash(c.chunk_hash) for c in candidates]
        assert evicted[0] == 5, (
            "eviction must start at the prefix TAIL even with per-rank keys; "
            "got %r" % evicted
        )
        assert evicted[-1] == 0, evicted

    def test_full_attention_single_group_prefix_survives(self):
        """A model with no sliding window has one object group needing the WHOLE
        prefix, so head loss is total loss. Same ordering requirement."""
        policy = LRUEvictionPolicy()
        for start in (0, 4, 8):
            keys = [make_key(i) for i in range(start, start + 4)]
            policy.note_key_positions(keys, list(range(start, start + 4)))
            policy.on_keys_created(keys)

        actions = policy.get_eviction_actions(0.5)
        evicted = {
            ObjectKey.Bytes2IntHash(k.chunk_hash) for a in actions for k in a.keys
        }
        survivors = sorted(set(range(12)) - evicted)
        assert survivors[0] == 0, survivors
        assert survivors == list(range(len(survivors))), survivors

    def test_touch_of_absent_key_neither_admits_nor_raises(self):
        """`on_keys_touched` reports access, not storage.

        A key can be evicted between the lookup that matched it and the touch
        that reports the hit. Admitting it would resurrect an entry with no
        backing data; and with every key of the batch absent the positions list
        is empty, which used to reach `min(pos)` and raise.
        """
        policy = LRUEvictionPolicy()
        resident = [make_key(i) for i in range(3)]
        policy.note_key_positions(resident, [0, 1, 2])
        policy.on_keys_created(resident)

        gone = [make_key(i) for i in range(7, 10)]
        policy.note_key_positions(gone, [7, 8, 9])
        policy.on_keys_touched(gone)

        assert len(policy.get_eviction_candidates(10)) == 3, (
            "touching absent keys must not admit them"
        )

        policy.on_keys_touched(gone + [resident[0]])
        candidates = policy.get_eviction_candidates(10)
        assert len(candidates) == 3
        assert ObjectKey.Bytes2IntHash(candidates[-1].chunk_hash) == 0, (
            "the resident key in a mixed batch must still be ordered"
        )


class TestPositionMapIsBounded:
    """Positions are staged for keys that may never be created."""

    def test_lookup_misses_do_not_grow_the_map_without_bound(self):
        """`resolve_obj_keys` runs on retrieve as well as store, so a lookup that
        misses stages positions for keys that are never created and therefore
        never reach `on_keys_removed`. Left unbounded that is one entry per unique
        chunk ever looked up on the server.
        """
        policy = LRUEvictionPolicy()
        cap = MAX_TRACKED_POSITIONS
        for base in range(0, cap * 2, 256):
            keys = [make_key(i) for i in range(base, base + 256)]
            policy.note_key_positions(keys, list(range(base, base + 256)))

        assert not policy._order, "nothing was ever created"
        assert len(policy._key_positions) <= cap, (
            "staged positions for never-created keys must be bounded; got %d"
            % len(policy._key_positions)
        )

    def test_ordering_still_correct_after_the_map_overflows(self):
        """Dropping a staged entry costs at most one batch its ordering: the
        resolver re-reports positions before every create and every touch.
        """
        policy = LRUEvictionPolicy()
        for base in range(0, MAX_TRACKED_POSITIONS * 2, 256):
            keys = [make_key(i) for i in range(base, base + 256)]
            policy.note_key_positions(keys, list(range(base, base + 256)))

        off = 10_000_000
        for start in (0, 3):
            batch = [make_key(off + i) for i in range(start, start + 3)]
            policy.note_key_positions(batch, list(range(start, start + 3)))
            policy.on_keys_created(batch)

        candidates = policy.get_eviction_candidates(6)
        evicted = [ObjectKey.Bytes2IntHash(c.chunk_hash) - off for c in candidates]
        assert evicted[0] == 5, evicted
        assert evicted[-1] == 0, evicted
