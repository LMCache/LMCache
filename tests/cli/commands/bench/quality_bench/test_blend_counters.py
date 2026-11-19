# SPDX-License-Identifier: Apache-2.0
"""Tests for blend-server counter parsing."""

# Standard
import io
import urllib.request

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import (
    FETCH_ATTEMPTS,
    BlendCounters,
    PrometheusBlendCounters,
    UnmeasuredBlendCounters,
    parse_blend_counters,
)

_EXPOSITION = """\
# HELP lmcache_blend_lookup_requested_tokens_total Total tokens
# TYPE lmcache_blend_lookup_requested_tokens_total counter
lmcache_blend_lookup_requested_tokens_total{instance="0"} 600.0
lmcache_blend_lookup_requested_tokens_total{instance="1"} 400.0
lmcache_blend_lookup_requested_tokens_created 1.7e9
lmcache_blend_lookup_non_prefix_hit_tokens_total 350
lmcache_blend_lookup_prefix_hit_tokens_total 100
lmcache_blend_lookup_segmented_prefix_hit_tokens_total 64
lmcache_blend_lookup_hit_tokens_total 450
lm:lmcache_blend_lookup_stale_chunks_total 3
lmcache_blend_fingerprints_registered_total 42
"""


class TestParseBlendCounters:
    def test_reads_and_sums_labelled_series(self) -> None:
        counters = parse_blend_counters(_EXPOSITION)
        assert counters.requested_tokens == 1000.0
        assert counters.non_prefix_hit_tokens == 350.0
        assert counters.prefix_hit_tokens == 100.0
        assert counters.fingerprints_registered == 42.0

    def test_namespace_prefix_still_matches(self) -> None:
        assert parse_blend_counters(_EXPOSITION).stale_chunks == 3.0

    def test_created_series_is_ignored(self) -> None:
        """``_created`` holds a timestamp, not a count."""
        assert parse_blend_counters(_EXPOSITION).requested_tokens == 1000.0

    def test_similar_names_do_not_bleed_in(self) -> None:
        """``lookup_hit_tokens`` should not count as prefix or non-prefix hits."""
        counters = parse_blend_counters("lmcache_blend_lookup_hit_tokens_total 9\n")
        assert counters == BlendCounters()

    def test_segmented_and_plain_prefix_hits_are_kept_apart(self) -> None:
        """The segmented counter ends with the plain prefix counter's name."""
        counters = parse_blend_counters(_EXPOSITION)
        assert counters.segmented_prefix_hit_tokens == 64.0
        assert counters.prefix_hit_tokens == 100.0

    def test_absent_counters_read_zero(self) -> None:
        assert parse_blend_counters("") == BlendCounters()


class TestBlendCounters:
    def test_blend_share(self) -> None:
        assert parse_blend_counters(_EXPOSITION).blend_share == pytest.approx(0.35)

    def test_segmented_prefix_tokens_are_not_blending(self) -> None:
        counters = BlendCounters(
            requested_tokens=100,
            non_prefix_hit_tokens=20,
            segmented_prefix_hit_tokens=50,
        )
        assert counters.blend_share == pytest.approx(0.2)

    def test_blend_share_without_lookups_is_zero(self) -> None:
        assert BlendCounters().blend_share == 0.0

    def test_difference_and_sum(self) -> None:
        a = BlendCounters(requested_tokens=10, non_prefix_hit_tokens=4)
        b = BlendCounters(requested_tokens=3, non_prefix_hit_tokens=1)
        assert (a - b) + b == a


class TestSources:
    def test_unmeasured_source_reads_zero(self) -> None:
        source = UnmeasuredBlendCounters()
        assert not source.measures_engagement
        assert source.read() == BlendCounters()

    def test_transient_fetch_failure_is_retried(self, monkeypatch) -> None:
        attempts: list[str] = []

        def flaky_urlopen(url: str, timeout: float) -> io.BytesIO:
            attempts.append(url)
            if len(attempts) < FETCH_ATTEMPTS:
                raise OSError("connection reset")
            return io.BytesIO(_EXPOSITION.encode())

        monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
        counters = PrometheusBlendCounters("http://blend/metrics").read()
        assert counters.requested_tokens == 1000.0
        assert len(attempts) == FETCH_ATTEMPTS

    def test_persistent_fetch_failure_raises(self, monkeypatch) -> None:
        attempts: list[str] = []

        def down(url: str, timeout: float) -> io.BytesIO:
            attempts.append(url)
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", down)
        with pytest.raises(RuntimeError, match=f"after {FETCH_ATTEMPTS} attempts"):
            PrometheusBlendCounters("http://blend/metrics").read()
        assert len(attempts) == FETCH_ATTEMPTS

    def test_unreachable_endpoint_raises(self) -> None:
        source = PrometheusBlendCounters("http://127.0.0.1:9/metrics")
        assert source.measures_engagement
        with pytest.raises(RuntimeError, match="blend-server metrics"):
            source.read()
