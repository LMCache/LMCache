# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator global CacheBlend fingerprint directory."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.blend_directory import (
    GlobalBlendMatcher,
    StoreRange,
)

CHUNK = 3
SCOPE = "model-a@"


def store_range(
    prefix: str, tokens: list[int], *, scope: str = SCOPE, old_st_base: int = 0
) -> StoreRange:
    """A StoreRange with one object_key per complete chunk (mirrors publisher)."""
    n_chunks = len(tokens) // CHUNK
    return StoreRange(
        model_scope=scope,
        tokens=tokens,
        object_keys=[f"{prefix}{i}" for i in range(n_chunks)],
        old_st_base=old_st_base,
    )


class TestRegisterMatch:
    def test_full_reuse(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        doc = [1, 2, 3, 4, 5, 6]  # 2 chunks
        assert m.register([store_range("K", doc)]) == 2
        matches = m.match(SCOPE, doc)
        assert [(x.object_key, x.old_st, x.cur_st) for x in matches] == [
            ("K0", 0, 0),
            ("K1", 3, 3),
        ]

    def test_stride_controls_offset(self) -> None:
        """A non-chunk-aligned offset is found at stride 1, missed at stride=chunk."""
        doc = [1, 2, 3, 4, 5, 6]
        req = [9, 1, 2, 3, 4, 5, 6]  # doc shifted by 1 (preamble 9)

        fine = GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=1)
        fine.register([store_range("K", doc)])
        out = fine.match(SCOPE, req)
        assert [(x.object_key, x.cur_st) for x in out] == [("K0", 1), ("K1", 4)]

        coarse = GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=CHUNK)
        coarse.register([store_range("K", doc)])  # probes pos 0,3,6 -> misses
        assert coarse.match(SCOPE, req) == []

    def test_dedup_by_object_key(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])  # single chunk K0
        matches = m.match(SCOPE, [1, 2, 3, 1, 2, 3])  # content repeats
        assert len(matches) == 1 and matches[0].object_key == "K0"

    def test_scope_isolation(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3], scope="model-a@")])
        assert m.match("model-b@", [1, 2, 3]) == []

    def test_request_shorter_than_chunk(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])
        assert m.match(SCOPE, [1, 2]) == []

    def test_no_match(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3])])
        assert m.match(SCOPE, [7, 8, 9]) == []


class TestIdempotencyEviction:
    def test_register_idempotent(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        rng = store_range("K", [1, 2, 3, 4, 5, 6])
        assert m.register([rng]) == 2
        assert m.register([rng]) == 0
        assert len(m.match(SCOPE, [1, 2, 3, 4, 5, 6])) == 2

    def test_remove_evicts(self) -> None:
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        m.register([store_range("K", [1, 2, 3, 4, 5, 6])])  # K0, K1
        assert m.remove(["K0"]) == 1
        matches = m.match(SCOPE, [1, 2, 3, 4, 5, 6])
        assert [x.object_key for x in matches] == ["K1"]  # only K0 gone

    def test_remove_unknown_is_noop(self) -> None:
        assert GlobalBlendMatcher(chunk_size=CHUNK).remove(["nope"]) == 0

    def test_count_mismatch_range_skipped(self) -> None:
        """A range whose object_keys count != chunk count is skipped, not partial."""
        m = GlobalBlendMatcher(chunk_size=CHUNK)
        doc = [1, 2, 3, 4, 5, 6]  # 2 chunks
        bad = StoreRange(
            model_scope=SCOPE,
            tokens=doc,
            object_keys=["K0"],  # only 1 key for 2 chunks -> mismatch
            old_st_base=0,
        )
        assert m.register([bad]) == 0
        assert m.match(SCOPE, doc) == []


class TestValidation:
    def test_bad_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            GlobalBlendMatcher(chunk_size=0)

    def test_bad_probe_stride(self) -> None:
        with pytest.raises(ValueError):
            GlobalBlendMatcher(chunk_size=CHUNK, probe_stride=0)
