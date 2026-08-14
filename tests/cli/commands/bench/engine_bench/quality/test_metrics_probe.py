# SPDX-License-Identifier: Apache-2.0
"""Tests for reading LMCache cache-hit counters from a Prometheus endpoint."""

# Standard
from unittest.mock import MagicMock, patch
import urllib.error

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.metrics_probe import (
    CacheCounters,
    MetricsProbe,
    _metrics_url,
    _sum_samples,
)

_BODY = (
    "# HELP lmcache:num_hit_tokens_total Total number of tokens hit\n"
    "# TYPE lmcache:num_hit_tokens_total counter\n"
    'lmcache:num_hit_tokens_total{model="m",worker_id="0"} 4096.0\n'
    'lmcache:num_hit_tokens_total{model="m",worker_id="1"} 2048.0\n'
    'lmcache:num_hit_tokens_created{model="m",worker_id="0"} 1700000000.0\n'
    'lmcache:num_requested_tokens_total{model="m",worker_id="0"} 8192.0\n'
)


class TestMetricsUrl:
    def test_strips_the_openai_v1_suffix(self) -> None:
        assert _metrics_url("http://localhost:8000/v1") == (
            "http://localhost:8000/metrics"
        )

    def test_adds_a_missing_scheme(self) -> None:
        assert _metrics_url("localhost:8000") == "http://localhost:8000/metrics"

    def test_keeps_https(self) -> None:
        assert _metrics_url("https://host/") == "https://host/metrics"


class TestSumSamples:
    def test_sums_every_label_set(self) -> None:
        assert _sum_samples(_BODY, "lmcache:num_hit_tokens") == (6144, True)

    def test_excludes_the_created_timestamp_gauge(self) -> None:
        """``_created`` shares the prefix but is not part of the value."""
        total, _ = _sum_samples(_BODY, "lmcache:num_hit_tokens")
        assert total == 6144

    def test_absent_metric_is_reported_as_not_found(self) -> None:
        assert _sum_samples(_BODY, "lmcache:num_stored_tokens") == (0, False)

    def test_a_metric_present_at_zero_is_still_found(self) -> None:
        body = 'lmcache:num_hit_tokens_total{model="m"} 0.0\n'
        assert _sum_samples(body, "lmcache:num_hit_tokens") == (0, True)


class TestCacheCountersDelta:
    def test_difference_of_two_readings(self) -> None:
        delta = CacheCounters(8192, 6144, True).delta(CacheCounters(8000, 6000, True))
        assert (delta.requested_tokens, delta.hit_tokens) == (192, 144)
        assert delta.available is True

    def test_unavailable_when_either_reading_was(self) -> None:
        available = CacheCounters(8192, 6144, True)
        assert available.delta(CacheCounters(0, 0, False)).available is False
        assert CacheCounters(0, 0, False).delta(available).available is False

    def test_counter_reset_clamps_to_zero(self) -> None:
        """An engine restart must not report negative activity."""
        delta = CacheCounters(10, 5, True).delta(CacheCounters(8000, 6000, True))
        assert (delta.requested_tokens, delta.hit_tokens) == (0, 0)


def _probe_with_response(body: str) -> MetricsProbe:
    response = MagicMock()
    response.read.return_value = body.encode()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestMetricsProbeRead:
    def test_reads_both_counters(self) -> None:
        probe = MetricsProbe("http://localhost:8000")
        with patch("urllib.request.urlopen", return_value=_probe_with_response(_BODY)):
            counters = probe.read()
        assert counters.available is True
        assert counters.requested_tokens == 8192
        assert counters.hit_tokens == 6144

    def test_unreachable_endpoint_is_not_fatal(self) -> None:
        """A benchmark collecting answers must not abort over a probe."""
        probe = MetricsProbe("http://localhost:8000")
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("refused")
        ):
            counters = probe.read()
        assert counters.available is False
        assert counters.hit_tokens == 0

    def test_engine_without_lmcache_metrics(self) -> None:
        """The baseline stack exposes no such counters."""
        probe = MetricsProbe("http://localhost:8000")
        body = 'vllm:num_requests_running{model="m"} 1.0\n'
        with patch("urllib.request.urlopen", return_value=_probe_with_response(body)):
            counters = probe.read()
        assert counters.available is False

    def test_partial_counters_are_treated_as_unavailable(self) -> None:
        probe = MetricsProbe("http://localhost:8000")
        body = 'lmcache:num_hit_tokens_total{model="m"} 10.0\n'
        with patch("urllib.request.urlopen", return_value=_probe_with_response(body)):
            assert probe.read().available is False
