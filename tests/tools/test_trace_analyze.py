# SPDX-License-Identifier: Apache-2.0
"""Tests for the request-trace cache analyzer."""

# Standard
from pathlib import Path
import json
import struct

# Third Party
import pytest

# First Party
from lmcache.tools.cache_simulator.trace_analyze import (
    TraceRecord,
    analyze,
    default_capacities,
    load_trace_records,
    parse_size,
    render_html,
)

FIXTURE = Path(__file__).parent / "data" / "trace_formats" / "lmcache.jsonl"
LLAMA = "meta-llama/Llama-3.1-8B"


def _record(ts: int, ids: list[int], session: str = "s") -> TraceRecord:
    return TraceRecord(timestamp_ns=ts, model="m", input_ids=ids, request_id=session)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class TestLoadTraceRecords:
    def test_reads_trace_spans_sorted_by_timestamp(self) -> None:
        records = load_trace_records([FIXTURE])

        assert [r.request_id for r in records] == [
            "sess-r1",
            "sess-r2",
            "sess-r3",
            "sess-r4",
        ]
        assert [r.timestamp_ns for r in records] == [
            1_700_000_001_000_000_000,
            1_700_000_002_000_000_000,
            1_700_000_003_000_000_000,
            1_700_000_004_500_000_000,
        ]
        assert records[0].input_ids == list(range(1, 9))
        assert records[1].input_ids == list(range(1, 13)) + [100, 101]
        assert records[0].model == LLAMA
        assert records[2].model == "Qwen/Qwen3-1.7B"

    def test_model_filter(self) -> None:
        records = load_trace_records([FIXTURE], model=LLAMA)
        assert [r.request_id for r in records] == ["sess-r1", "sess-r2"]

    def test_input_ids_as_json_string(self) -> None:
        # The fixture's sess-r4 carries lmcache.input_ids as one JSON string,
        # as a collector transform may write it.
        (r4,) = [r for r in load_trace_records([FIXTURE]) if r.request_id == "sess-r4"]
        assert r4.input_ids == list(range(300, 308))

    def test_no_match_error_describes_the_file(self, tmp_path: Path) -> None:
        span = {"name": "request", "startTimeUnixNano": "1", "attributes": []}
        path = tmp_path / "other.jsonl"
        path.write_text(
            '{"resourceSpans":[{"scopeSpans":[{"spans":[%s,%s]}]}]}\n'
            % (json.dumps(span), json.dumps(span))
        )
        expected = r"1 line\(s\) with 2 OTLP span\(s\) named request \(2\)"
        with pytest.raises(ValueError, match=expected):
            load_trace_records([path])

    def test_repeat_lookup_keeps_earliest_span(self) -> None:
        # The fixture's last batch looks sess-r1 up again, later.
        (r1,) = [r for r in load_trace_records([FIXTURE]) if r.request_id == "sess-r1"]
        assert r1.timestamp_ns == 1_700_000_001_000_000_000

    def test_accepts_plain_int_encoding(self, tmp_path: Path) -> None:
        # Some exporters write int64 as JSON numbers rather than strings.
        span = {
            "name": "mp.request.trace",
            "startTimeUnixNano": 5,
            "attributes": [
                {"key": "session_id", "value": {"stringValue": "x"}},
                {"key": "lmcache.model", "value": {"stringValue": "m"}},
                {
                    "key": "lmcache.input_ids",
                    "value": {"arrayValue": {"values": [{"intValue": 7}]}},
                },
            ],
        }
        path = tmp_path / "t.jsonl"
        path.write_text(
            '{"resourceSpans":[{"scopeSpans":[{"spans":[%s]}]}]}\n\n' % json.dumps(span)
        )
        (rec,) = load_trace_records([path])
        assert rec == TraceRecord(5, "m", [7], "x")

    def test_dynamo_format(self) -> None:
        records = load_trace_records(
            [FIXTURE.with_name("dynamo.jsonl")], trace_format="dynamo"
        )

        # The size-limit marker, the request_end record and the LMCache span
        # are skipped; dyn-c's response was dropped but its request counts.
        assert [r.request_id for r in records] == ["dyn-a", "dyn-c", "dyn-b"]
        assert records[0].timestamp_ns == 1_792_617_070_080_000_000
        assert records[1].model == "Qwen/Qwen3-1.7B"
        texts = [
            struct.pack(f">{len(r.input_ids)}I", *r.input_ids).rstrip(b"\0").decode()
            for r in records
        ]
        assert texts[0] == (
            "system\nYou are the support assistant for an LLM serving team. "
            "Answer briefly, quote exact flag names, and say when something is "
            "not supported.\nuser\nHow do I turn on the token trace?"
        )
        # Tools come first, as chat templates render them.
        assert texts[2].startswith('[{"function": {"description": "Get the current')


# ---------------------------------------------------------------------------
# Prefix replay
# ---------------------------------------------------------------------------


class TestPrefixReplay:
    RECORDS = [
        _record(1, list(range(1, 9))),  # 2 chunks
        _record(2, list(range(1, 13)) + [100, 101]),  # 3 chunks + 2 tail tokens
        _record(3, [1, 2, 3, 4, 200, 201, 202, 203]),  # 2 chunks
    ]

    def test_known_answer_without_eviction(self) -> None:
        (res,) = analyze(self.RECORDS, [4], [16])

        assert res.total_requests == 3
        assert res.total_tokens == 30
        assert res.per_request_prefix_hit_tokens == [0, 8, 4]
        assert res.prefix_hit_tokens == 12
        assert res.prefix_hit_rate == pytest.approx(0.4)
        assert res.eviction_count == 0
        # Every match lies inside the prefix, so nothing is non-prefix.
        assert res.per_request_non_prefix_hit_tokens == [0, 0, 0]

    def test_lru_eviction_breaks_prefix(self) -> None:
        # Two chunk slots: request 2's third chunk evicts chunk [1..4],
        # which is the first chunk of request 3.
        (res,) = analyze(self.RECORDS, [4], [8])

        assert res.capacity_chunks == 2
        assert res.per_request_prefix_hit_tokens == [0, 8, 0]
        # c1 (by request 2's third chunk), then c2 and c3 (by request 3).
        assert res.eviction_count == 3

    def test_partial_chunk_is_never_hit(self) -> None:
        (res,) = analyze([_record(1, [1, 2, 3]), _record(2, [1, 2, 3])], [4], [64])
        assert res.total_tokens == 6
        assert res.prefix_hit_tokens == 0

    def test_grid_order_and_hash_sharing(self) -> None:
        results = analyze(self.RECORDS, [4, 8], [8, 16])
        assert [(r.chunk_size, r.capacity_tokens) for r in results] == [
            (4, 8),
            (4, 16),
            (8, 8),
            (8, 16),
        ]
        # chunk 8: request 2 hits its first 8 tokens from request 1.
        assert results[3].per_request_prefix_hit_tokens == [0, 8, 0]


# ---------------------------------------------------------------------------
# CacheBlend leg
# ---------------------------------------------------------------------------


class TestBlendReplay:
    STORED = _record(1, list(range(1, 13)))  # chunks [1..4] [5..8] [9..12]
    # The stored chunks [1..4] and [5..8] reappear one token later.
    SHIFTED = _record(2, [9, 1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52])

    def test_shared_segment_at_different_offset(self) -> None:
        (res,) = analyze([self.STORED, self.SHIFTED], [4], [64])

        assert res.per_request_prefix_hit_tokens == [0, 0]
        assert res.per_request_non_prefix_hit_tokens == [0, 8]
        assert res.non_prefix_hit_tokens == 8
        assert res.summary()["requests_with_non_prefix_hits"] == 1

    def test_prefix_covered_matches_are_not_non_prefix(self) -> None:
        repeat = _record(2, list(range(1, 13)))
        (res,) = analyze([self.STORED, repeat], [4], [64])

        assert res.per_request_prefix_hit_tokens == [0, 12]
        assert res.per_request_non_prefix_hit_tokens == [0, 0]

    def test_match_outside_prefix_counts_once(self) -> None:
        # Prefix hit on [1..4]; [5..8] is then found shifted by one token.
        query = _record(2, [1, 2, 3, 4, 60, 5, 6, 7, 8, 61, 62, 63])
        (res,) = analyze([self.STORED, query], [4], [64])

        assert res.per_request_prefix_hit_tokens == [0, 4]
        assert res.per_request_non_prefix_hit_tokens == [0, 4]

    def test_evicted_chunks_no_longer_match(self) -> None:
        filler = _record(2, list(range(300, 312)))  # evicts all stored chunks
        shifted = _record(3, self.SHIFTED.input_ids)
        (res,) = analyze([self.STORED, filler, shifted], [4], [12])

        # Three slots: the filler evicts the stored chunks, the shifted
        # request then evicts the filler.
        assert res.eviction_count == 6
        assert res.per_request_non_prefix_hit_tokens == [0, 0, 0]


# ---------------------------------------------------------------------------
# Report and helpers
# ---------------------------------------------------------------------------


def test_render_html_embeds_every_cell() -> None:
    records = [_record(1, list(range(1, 9))), _record(2, list(range(1, 9)))]
    results = analyze(records, [4, 8], [8, 16])
    page = render_html(results, "two requests")

    assert page.startswith("<!doctype html>")
    assert '"chunk_sizes": [4, 8]' in page
    assert '"capacities_tokens": [8, 16]' in page
    for r in results:
        assert f'"{r.chunk_size}:{r.capacity_tokens}"' in page
    assert "two requests" in page


def test_render_html_prerenders_default_cell() -> None:
    records = [_record(1, list(range(1, 9))), _record(2, list(range(1, 9)))]
    results = analyze(records, [4, 8], [8, 16])
    page = render_html(results, "two requests")
    # Smallest chunk size at the largest capacity: 8 of 16 tokens hit.
    static = page.split('<script id="data"')[0]
    assert '<div class="v" id="t-phr">50.00%</div>' in static
    assert '<div class="v" id="t-req">2</div>' in static
    assert '<div class="v" id="t-tok">16</div>' in static
    assert static.count("<tr") == 3  # header + two capacities


def test_render_html_escapes_script_terminator() -> None:
    results = analyze([_record(1, list(range(1, 9)))], [4], [8])
    page = render_html(results, "</script><b>")
    assert "</script><b>" not in page.split('type="application/json">')[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4096", 4096),
        ("64k", 64 * 1024),
        ("2G", 2 * 2**30),
        ("1.5GiB", int(1.5 * 2**30)),
        ("512MB", 512 * 2**20),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


def test_parse_size_rejects_garbage() -> None:
    with pytest.raises(Exception, match="not a size"):
        parse_size("many")


def test_default_capacities_cover_trace() -> None:
    assert default_capacities(100) == [4096]
    assert default_capacities(12_000) == [4096, 6144, 8192, 11264, 16384]
