# SPDX-License-Identifier: Apache-2.0
"""Content-fingerprint dedup in the blend matcher (``--enable-dedup-content``).

The default dedup key is the caller's content hash, which is prefix-chained:
the same text behind two prefixes is indexed twice, the second registration
takes over the probe-table slot, and the first entry is orphaned. With
``dedup_content=True`` the first registration wins and duplicates are skipped.
"""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.modules.blend import BlendTokenRangeMatcher

CHUNK_SIZE = 256


def _chunk(seed: int) -> list[int]:
    """Deterministic ``CHUNK_SIZE`` tokens derived from seed; unique per seed."""
    return [seed * CHUNK_SIZE + i + 1 for i in range(CHUNK_SIZE)]


def _register(
    matcher: BlendTokenRangeMatcher,
    tokens: list[int],
    chunk_hash_seeds: list[int],
    position_offset: int = 0,
) -> None:
    matcher.on_new_token_hashes(
        tokens,
        [ObjectKey.IntHash2Bytes(s) for s in chunk_hash_seeds],
        position_offset=position_offset,
    )


def _matched_hashes(matcher: BlendTokenRangeMatcher, query: list[int]) -> set[bytes]:
    return {m.hash for m in matcher.match_sub_sequence(query)}


# -- Off by default -----------------------------------------------------------


def test_duplicate_content_registers_twice_by_default():
    """Without the flag, the second registration takes over the slot."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE)
    first, second = ObjectKey.IntHash2Bytes(101), ObjectKey.IntHash2Bytes(202)

    _register(matcher, _chunk(1), [101])
    _register(matcher, _chunk(1), [202], position_offset=CHUNK_SIZE)

    matches = matcher.match_sub_sequence(_chunk(1))
    assert [m.hash for m in matches] == [second]
    assert matches[0].old_st == CHUNK_SIZE

    # The first entry is orphaned: evicting the second loses the text.
    matcher.remove_chunks([second])
    assert matcher.match_sub_sequence(_chunk(1)) == []
    assert first not in _matched_hashes(matcher, _chunk(1))


# -- Enabled ------------------------------------------------------------------


def test_duplicate_content_is_skipped_when_enabled():
    """The first registration keeps the slot; the duplicate is not indexed."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE, dedup_content=True)
    first, second = ObjectKey.IntHash2Bytes(101), ObjectKey.IntHash2Bytes(202)

    _register(matcher, _chunk(1), [101])
    _register(matcher, _chunk(1), [202], position_offset=CHUNK_SIZE)

    matches = matcher.match_sub_sequence(_chunk(1))
    assert [m.hash for m in matches] == [first]
    assert matches[0].old_st == 0

    # The skipped hash was never indexed, so evicting it changes nothing.
    matcher.remove_chunks([second])
    assert _matched_hashes(matcher, _chunk(1)) == {first}


def test_distinct_content_still_registers_when_enabled():
    """Dedup keys on content only — different text is always indexed."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE, dedup_content=True)
    _register(matcher, _chunk(1) + _chunk(2) + _chunk(3), [101, 102, 103])

    assert _matched_hashes(matcher, _chunk(1) + _chunk(2) + _chunk(3)) == {
        ObjectKey.IntHash2Bytes(101),
        ObjectKey.IntHash2Bytes(102),
        ObjectKey.IntHash2Bytes(103),
    }


def test_duplicate_content_within_one_batch_is_skipped():
    """In-batch duplicates are invisible to the probe table — dedup anyway."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE, dedup_content=True)
    first, second = ObjectKey.IntHash2Bytes(101), ObjectKey.IntHash2Bytes(102)

    _register(matcher, _chunk(1) + _chunk(1), [101, 102])

    matches = matcher.match_sub_sequence(_chunk(1))
    assert [m.hash for m in matches] == [first]
    assert matches[0].old_st == 0
    matcher.remove_chunks([second])
    assert _matched_hashes(matcher, _chunk(1)) == {first}


def test_content_is_reregistered_after_eviction():
    """Evicting the retained entry frees the text for a later registration."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE, dedup_content=True)
    first, second = ObjectKey.IntHash2Bytes(101), ObjectKey.IntHash2Bytes(202)

    _register(matcher, _chunk(1), [101])
    matcher.remove_chunks([first])
    assert matcher.match_sub_sequence(_chunk(1)) == []

    _register(matcher, _chunk(1), [202], position_offset=CHUNK_SIZE)
    matches = matcher.match_sub_sequence(_chunk(1))
    assert [m.hash for m in matches] == [second]
    assert matches[0].old_st == CHUNK_SIZE


def test_repeated_registration_of_same_hash_is_idempotent():
    """The token-hash dedup path is unchanged by the content dedup."""
    matcher = BlendTokenRangeMatcher(chunk_size=CHUNK_SIZE, dedup_content=True)
    _register(matcher, _chunk(1), [101])
    _register(matcher, _chunk(1), [101], position_offset=CHUNK_SIZE)

    matches = matcher.match_sub_sequence(_chunk(1))
    assert [m.hash for m in matches] == [ObjectKey.IntHash2Bytes(101)]
    assert matches[0].old_st == 0


# -- Config -------------------------------------------------------------------


def _parse_mp(argv: list[str]) -> MPServerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parse_args_to_mp_server_config(parser.parse_args(argv))


@pytest.mark.parametrize(
    "argv, expected",
    [([], False), (["--enable-dedup-content"], True)],
)
def test_enable_dedup_content_flag(argv, expected):
    assert _parse_mp(argv).enable_dedup_content is expected
