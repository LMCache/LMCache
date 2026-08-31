# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator content fingerprint index (fragment lookup):
discovery, token-exact verification, chunk lifecycle, and the key
directory integration that drives it from token bindings."""

# Standard
import random
import threading

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.blend_index import BlendIndex
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory

CHUNK = 4


def _content(*tokens: int) -> np.ndarray:
    return np.asarray(tokens, dtype=np.uint32)


def _index(probe_stride: int = 1) -> BlendIndex:
    return BlendIndex(chunk_size=CHUNK, probe_stride=probe_stride)


def _tuples(matches) -> list[tuple[bytes, int, int]]:
    return [(m.chunk_hash, m.old_st, m.cur_st) for m in matches]


# -- Discovery ---------------------------------------------------------------


def test_content_is_found_at_its_query_offset():
    index = _index()
    index.add(_content(10, 11, 12, 13), b"A", token_offset=0)

    matches = index.match(np.asarray([7, 8, 10, 11, 12, 13, 9], dtype=np.uint64))
    assert _tuples(matches) == [(b"A", 0, 2)]


def test_old_st_is_the_stored_position_not_the_query_position():
    """re-RoPE shifts from where the chunk was stored to where it is
    needed, so the two positions must be reported independently."""
    index = _index()
    index.add(_content(10, 11, 12, 13), b"A", token_offset=1024)

    [match] = index.match(np.asarray([0, 10, 11, 12, 13], dtype=np.uint64))
    assert (match.old_st, match.cur_st) == (1024, 1)


def test_several_chunks_are_found_in_one_query_ascending():
    index = _index()
    index.add(_content(1, 2, 3, 4), b"A", token_offset=0)
    index.add(_content(5, 6, 7, 8), b"B", token_offset=4)

    matches = index.match(np.asarray([9, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.uint64))
    assert _tuples(matches) == [(b"B", 4, 1), (b"A", 0, 5)]


def test_repeated_content_matches_once_at_its_first_position():
    index = _index()
    index.add(_content(1, 2, 3, 4), b"A", token_offset=0)

    matches = index.match(np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.uint64))
    assert _tuples(matches) == [(b"A", 0, 0)]


def test_query_shorter_than_the_window_matches_nothing():
    index = _index()
    index.add(_content(1, 2, 3, 4), b"A", token_offset=0)

    assert index.match(np.asarray([1, 2, 3], dtype=np.uint64)) == []


def test_absent_content_matches_nothing():
    index = _index()
    index.add(_content(1, 2, 3, 4), b"A", token_offset=0)

    assert index.match(np.asarray([5, 6, 7, 8, 9], dtype=np.uint64)) == []


# -- Token-exact verification ------------------------------------------------


def test_near_miss_content_is_rejected():
    """Discovery is by hash but acceptance is by content, so a window
    differing in one token must not match."""
    index = _index()
    index.add(_content(10, 11, 12, 13), b"A", token_offset=0)

    assert index.match(np.asarray([10, 11, 12, 99], dtype=np.uint64)) == []


def test_verification_uses_the_indexed_content_not_the_stored_length():
    index = _index()
    index.add(_content(10, 11, 12, 13), b"A", token_offset=0)

    # A superset window containing the content still matches at its offset.
    [match] = index.match(np.asarray([10, 11, 12, 13, 14], dtype=np.uint64))
    assert match.cur_st == 0


# -- Chunk lifecycle ---------------------------------------------------------


def test_add_is_idempotent():
    index = _index()
    content = _content(1, 2, 3, 4)
    index.add(content, b"A", token_offset=0)
    index.add(content, b"A", token_offset=0)

    stats = index.stats()
    assert (stats.num_contents, stats.num_chunks) == (1, 1)


def test_readding_a_chunk_updates_its_offset():
    index = _index()
    content = _content(1, 2, 3, 4)
    index.add(content, b"A", token_offset=0)
    index.add(content, b"A", token_offset=256)

    [match] = index.match(np.asarray([1, 2, 3, 4], dtype=np.uint64))
    assert match.old_st == 256
    assert index.stats().num_chunks == 1


def test_remove_makes_content_undiscoverable():
    index = _index()
    content = _content(1, 2, 3, 4)
    index.add(content, b"A", token_offset=0)
    index.remove(content, b"A")

    assert index.match(np.asarray([1, 2, 3, 4], dtype=np.uint64)) == []
    assert index.stats().num_contents == 0


def test_remove_of_unknown_chunk_or_content_is_a_noop():
    index = _index()
    content = _content(1, 2, 3, 4)
    index.add(content, b"A", token_offset=0)

    index.remove(content, b"UNKNOWN")
    index.remove(_content(9, 9, 9, 9), b"A")

    assert _tuples(index.match(np.asarray([1, 2, 3, 4], dtype=np.uint64))) == [
        (b"A", 0, 0)
    ]


def test_identical_content_under_two_prefixes_survives_one_eviction():
    """Chunk hashes are prefix-chained, so the same text stored after
    different prefixes yields two chunks. Evicting one must leave the
    other discoverable."""
    index = _index()
    content = _content(1, 2, 3, 4)
    index.add(content, b"A", token_offset=0)
    index.add(content, b"B", token_offset=512)
    assert index.stats().num_chunks == 2

    index.remove(content, b"A")

    assert _tuples(index.match(np.asarray([1, 2, 3, 4], dtype=np.uint64))) == [
        (b"B", 512, 0)
    ]


def test_content_of_the_wrong_length_is_not_indexed():
    """Only full-chunk content can match a full-chunk window."""
    index = _index()
    index.add(_content(1, 2, 3), b"SHORT", token_offset=0)
    index.add(_content(1, 2, 3, 4, 5), b"LONG", token_offset=0)

    assert index.stats().num_contents == 0


# -- Table growth ------------------------------------------------------------


def test_many_contents_stay_matchable_across_growth_and_compaction():
    index = _index()
    contents = {bytes([i]): _content(i, i + 1, i + 2, i + 3) for i in range(1, 200)}
    for chunk_hash, content in contents.items():
        index.add(content, chunk_hash, token_offset=0)

    for chunk_hash, content in contents.items():
        matches = index.match(content.astype(np.uint64))
        assert _tuples(matches) == [(chunk_hash, 0, 0)]

    # Drop most of them: compaction must not lose the survivors.
    survivors = {k: v for k, v in contents.items() if k[0] % 20 == 0}
    for chunk_hash, content in contents.items():
        if chunk_hash not in survivors:
            index.remove(content, chunk_hash)

    assert index.stats().num_contents == len(survivors)
    for chunk_hash, content in survivors.items():
        assert _tuples(index.match(content.astype(np.uint64))) == [(chunk_hash, 0, 0)]


# -- Stride ------------------------------------------------------------------


def test_stride_skips_offsets_between_probes():
    index = _index(probe_stride=CHUNK)
    index.add(_content(10, 11, 12, 13), b"A", token_offset=0)

    # Aligned with a probe position -> found.
    assert index.match(np.asarray([10, 11, 12, 13], dtype=np.uint64))
    # One token off a probe position -> missed (recall traded for CPU).
    assert index.match(np.asarray([0, 10, 11, 12, 13], dtype=np.uint64)) == []


# -- Construction ------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,stride", [(0, 1), (-1, 1), (4, 0), (4, -1)])
def test_invalid_construction_is_rejected(chunk_size: int, stride: int):
    with pytest.raises(ValueError):
        BlendIndex(chunk_size=chunk_size, probe_stride=stride)


# -- Reference comparison ----------------------------------------------------


def _reference(
    contents: dict[bytes, tuple[np.ndarray, int]], query: list[int], chunk: int
) -> list[tuple[bytes, int, int]]:
    """Brute-force scan: each content's first occurrence in the query."""
    found: list[tuple[bytes, int, int]] = []
    seen: set[bytes] = set()
    for start in range(len(query) - chunk + 1):
        window = query[start : start + chunk]
        for chunk_hash, (content, offset) in contents.items():
            if chunk_hash in seen:
                continue
            if window == content.tolist():
                seen.add(chunk_hash)
                found.append((chunk_hash, offset, start))
    return found


def test_matches_a_brute_force_scan_on_random_input():
    """Recall is complete: the occupancy filter carries no identity, so
    fingerprints sharing a slot cannot hide one another. A direct-address
    table keyed slot -> entry would silently drop the losers here."""
    rng = random.Random(1234)
    chunk = 8
    index = BlendIndex(chunk_size=chunk, probe_stride=1)

    # Distinct contents, so a content maps to exactly one chunk hash.
    contents: dict[bytes, tuple[np.ndarray, int]] = {}
    for i in range(60):
        tokens = [rng.randrange(1, 5000) for _ in range(chunk)]
        content = np.asarray(tokens, dtype=np.uint32)
        if any(content.tolist() == c.tolist() for c, _ in contents.values()):
            continue
        chunk_hash = f"c{i}".encode()
        offset = i * chunk
        contents[chunk_hash] = (content, offset)
        index.add(content, chunk_hash, token_offset=offset)

    for _ in range(20):
        query: list[int] = []
        for _ in range(12):
            if rng.random() < 0.5 and contents:
                pick = rng.choice(list(contents.values()))[0]
                query.extend(pick.tolist())
            else:
                query.extend(rng.randrange(1, 5000) for _ in range(chunk))
        expected = _reference(contents, query, chunk)
        actual = _tuples(index.match(np.asarray(query, dtype=np.uint64)))
        assert actual == expected


# -- Concurrency -------------------------------------------------------------


def test_concurrent_add_and_match_stay_consistent():
    index = BlendIndex(chunk_size=CHUNK, probe_stride=1)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(1, 400):
                index.add(_content(i, i + 1, i + 2, i + 3), bytes([i % 256]), i)
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            probe = np.asarray([5, 6, 7, 8], dtype=np.uint64)
            while not stop.is_set():
                index.match(probe)
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert index.match(np.asarray([5, 6, 7, 8], dtype=np.uint64))


# -- Key directory integration -----------------------------------------------


def _key(hash_byte: int, kv_rank: int = 0) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=kv_rank)


def _store(
    keys: list[ObjectKey],
    token_ids: list[int],
    seq: int,
    token_offset: int = 0,
    event_type: CacheEventType = CacheEventType.STORE,
) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=event_type,
        tier=Tier.L1,
        backend="dram",
        entries=[
            CacheEventEntry(
                key=k.to_encoded_object_key(),
                size_bytes=100,
                token_ids=token_ids,
                token_offset=token_offset,
            )
            for k in keys
        ],
    )


def _directory() -> KeyDirectory:
    directory = KeyDirectory()
    directory.enable_blend_lookup(chunk_size=CHUNK, probe_stride=1)
    return directory


def test_a_stored_chunk_becomes_fragment_matchable():
    directory = _directory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1, token_offset=256))

    matches = directory.blend_match(np.asarray([0, 10, 11, 12, 13], dtype=np.uint64))
    assert _tuples(matches) == [(_key(1).chunk_hash, 256, 1)]


def test_deleting_the_last_placement_removes_it_from_fragment_matching():
    """Exact eviction: unlike a tombstoned fingerprint table, a deleted
    chunk stops being matched, so no prefetch is wasted on it."""
    directory = _directory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1))
    directory.consume(_store([_key(1)], [], seq=2, event_type=CacheEventType.DELETE))

    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []
    assert directory.blend_stats().num_contents == 0


def test_chunk_stays_matchable_while_any_rank_holds_it():
    directory = _directory()
    directory.consume(
        _store([_key(1, kv_rank=0), _key(1, kv_rank=1)], [10, 11, 12, 13], seq=1)
    )
    directory.consume(
        _store([_key(1, kv_rank=0)], [], seq=2, event_type=CacheEventType.DELETE)
    )

    query = np.asarray([10, 11, 12, 13], dtype=np.uint64)
    assert _tuples(directory.blend_match(query)) == [(_key(1).chunk_hash, 0, 0)]

    directory.consume(
        _store([_key(1, kv_rank=1)], [], seq=3, event_type=CacheEventType.DELETE)
    )
    assert directory.blend_match(query) == []


def test_restoring_a_chunk_with_new_content_retires_the_old_fingerprint():
    """A chunk hash should never stay discoverable under content it no
    longer holds."""
    directory = _directory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1))
    directory.consume(_store([_key(1)], [20, 21, 22, 23], seq=2))

    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []
    assert _tuples(
        directory.blend_match(np.asarray([20, 21, 22, 23], dtype=np.uint64))
    ) == [(_key(1).chunk_hash, 0, 0)]
    assert directory.blend_stats().num_contents == 1


def test_a_tokenless_store_is_not_fragment_matchable():
    """The emitter could not stamp the chunk, so the directory has no
    content to verify against — a miss, not an error."""
    directory = _directory()
    directory.consume(_store([_key(1)], [], seq=1))

    assert directory.blend_stats().num_contents == 0
    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []


def test_incarnation_fencing_removes_fragment_matches():
    directory = _directory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1))

    directory.fence_instance("node-a")

    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []


def test_chunk_with_unreported_offset_is_not_fragment_matchable():
    """An emitter predating token offsets reports content but no position.
    The binding still fills (so key -> tokens introspection works), but the
    chunk must not be matchable: without the stored position a match would
    re-RoPE the reused KV from the wrong source, which is wrong KV rather
    than a miss."""
    directory = _directory()
    directory.consume(
        CacheEventBatch(
            instance_id="node-a",
            incarnation=1,
            seq=1,
            event_type=CacheEventType.STORE,
            tier=Tier.L1,
            backend="dram",
            entries=[
                CacheEventEntry(
                    key=_key(1).to_encoded_object_key(),
                    size_bytes=100,
                    token_ids=[10, 11, 12, 13],
                )  # no token_offset -> UNKNOWN
            ],
        )
    )

    # Content known, but with no position it cannot be indexed.
    assert directory.get_token_ids([_key(1).chunk_hash]) == [(10, 11, 12, 13)]
    assert directory.blend_stats().num_chunks == 0
    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []


def test_a_later_offset_bearing_store_makes_the_chunk_matchable():
    """The unknown-offset state is repaired by the chunk's next store from
    an emitter that does report positions."""
    directory = _directory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1, token_offset=-1))
    assert directory.blend_stats().num_chunks == 0

    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=2, token_offset=256))

    matches = directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64))
    assert _tuples(matches) == [(_key(1).chunk_hash, 256, 0)]


def test_blend_lookup_is_off_until_enabled():
    """Hashing chunk content costs CPU on every store, so a fleet that
    does not run CacheBlend must not pay for it: a directory that was
    never enabled indexes nothing and matches nothing."""
    directory = KeyDirectory()  # not enabled
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1, token_offset=0))

    assert directory.blend_stats().num_chunks == 0
    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []
    # The token binding itself is still recorded for introspection.
    assert directory.get_token_ids([_key(1).chunk_hash]) == [(10, 11, 12, 13)]


def test_enabling_does_not_retroactively_index_earlier_stores():
    """Documented contract: chunks stored before enable_blend_lookup are
    not back-filled, so the call belongs at startup."""
    directory = KeyDirectory()
    directory.consume(_store([_key(1)], [10, 11, 12, 13], seq=1, token_offset=0))

    directory.enable_blend_lookup(chunk_size=CHUNK, probe_stride=1)

    assert directory.blend_match(np.asarray([10, 11, 12, 13], dtype=np.uint64)) == []
    directory.consume(_store([_key(2)], [20, 21, 22, 23], seq=2, token_offset=0))
    assert directory.blend_match(np.asarray([20, 21, 22, 23], dtype=np.uint64))
