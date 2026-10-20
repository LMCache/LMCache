# SPDX-License-Identifier: Apache-2.0
"""Blend-server engagement counters for ``lmcache bench quality``.

Utility of a CacheBlend quality result is dependent on the blend firing correctly.
These counters, read from the blend server's Prometheus ``/metrics``, show how
much of each prompt was actually served by blending.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import re
import time
import urllib.request

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Counter names as the blend server's OpenTelemetry Prometheus exporter
# publishes them (``lmcache_blend.<name>`` with dots mapped to underscores).
# Matched as a suffix, so a recording-rule or namespace prefix such as
# ``lm:`` still matches.
_COUNTER_NAMES: dict[str, str] = {
    "requested_tokens": "lmcache_blend_lookup_requested_tokens",
    "prefix_hit_tokens": "lmcache_blend_lookup_prefix_hit_tokens",
    "non_prefix_hit_tokens": "lmcache_blend_lookup_non_prefix_hit_tokens",
    "segmented_prefix_hit_tokens": "lmcache_blend_lookup_segmented_prefix_hit_tokens",
    "stale_chunks": "lmcache_blend_lookup_stale_chunks",
    "fingerprints_registered": "lmcache_blend_fingerprints_registered",
}

# ``name{labels} value`` or ``name value``. Comments and blanks never match.
_SAMPLE_LINE = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)")

_FETCH_TIMEOUT_S = 10.0

# Counters are read around every candidate question, and a failed read aborts
# the run (losing every pass already made), so a transient scrape failure
# is retried before it counts. Public so that tests can skip the wait.
FETCH_ATTEMPTS = 3
FETCH_RETRY_DELAY_S = 1.0


@dataclass(frozen=True)
class BlendCounters:
    """Cumulative blend-server counters, or the difference of two readings.

    Attributes:
        requested_tokens: Prompt tokens the server was asked to look up.
        prefix_hit_tokens: The subset of ``requested_tokens`` served as an
            exact prefix match.
        non_prefix_hit_tokens: The subset of ``requested_tokens`` served by
            blending reused chunks at new positions.
        segmented_prefix_hit_tokens: The subset of ``requested_tokens``
            served as prefix tokens after a gap. With
            ``--enable-segmented-prefix``, when part of the prefix fails to
            load from L2, the chunks after the gap are still loaded at their
            original positions. Reused without blending, so not
            part of ``blend_share`` -- reported so that a low share caused by
            L2 load failures is told apart from blending not firing.
        stale_chunks: Chunks whose fingerprint matched but whose KV was no
            longer in storage -- their tokens were recomputed instead.
        fingerprints_registered: Chunks added to the fingerprint index. The
            index holds at most 2^20 chunks per server lifetime. Past that,
            new content can no longer be blended.
    """

    requested_tokens: float = 0.0
    prefix_hit_tokens: float = 0.0
    non_prefix_hit_tokens: float = 0.0
    segmented_prefix_hit_tokens: float = 0.0
    stale_chunks: float = 0.0
    fingerprints_registered: float = 0.0

    def __add__(self, other: "BlendCounters") -> "BlendCounters":
        """Return the field-wise sum, e.g. of two per-question deltas."""
        return BlendCounters(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __sub__(self, earlier: "BlendCounters") -> "BlendCounters":
        """Return the counts accumulated since *earlier*."""
        return BlendCounters(
            **{
                f.name: getattr(self, f.name) - getattr(earlier, f.name)
                for f in fields(self)
            }
        )

    @property
    def blend_share(self) -> float:
        """Share of looked-up tokens served by blending. ``0.0`` if none."""
        if self.requested_tokens <= 0:
            return 0.0
        return self.non_prefix_hit_tokens / self.requested_tokens


def parse_blend_counters(exposition: str) -> BlendCounters:
    """Extract :class:`BlendCounters` from Prometheus text exposition.

    Samples of the same counter with different labels are summed. A
    counter's ``_total`` and ``_created`` series are told apart -- only
    ``_total`` (or the bare name) is read.

    Args:
        exposition: The body of a ``/metrics`` response.

    Returns:
        The counters. Any counter absent from *exposition* reads ``0.0``.
    """
    values = dict.fromkeys(_COUNTER_NAMES, 0.0)
    for line in exposition.splitlines():
        match = _SAMPLE_LINE.match(line)
        if match is None:
            continue
        name, raw = match.groups()
        if name.endswith("_created"):
            continue
        base = name[: -len("_total")] if name.endswith("_total") else name
        for field_name, counter in _COUNTER_NAMES.items():
            if base == counter or base.endswith(":" + counter):
                values[field_name] += float(raw)
    return BlendCounters(**values)


class BlendCounterSource(ABC):
    """Base class for source of engagement counters."""

    @property
    @abstractmethod
    def measures_engagement(self) -> bool:
        """Whether :meth:`read` returns real counters."""

    @abstractmethod
    def read(self) -> BlendCounters:
        """Return the current cumulative counters.

        Raises:
            RuntimeError: If the counters could not be read.
        """


class PrometheusBlendCounters(BlendCounterSource):
    """Reads counters from a blend server's Prometheus endpoint."""

    def __init__(self, metrics_url: str) -> None:
        """Bind to *metrics_url*.

        Args:
            metrics_url: Full URL of the ``/metrics`` endpoint.
        """
        self._metrics_url = metrics_url

    @property
    def measures_engagement(self) -> bool:
        """Always ``True``."""
        return True

    def read(self) -> BlendCounters:
        """Fetch and parse the endpoint, retrying a failed fetch.

        Blocks between attempts. Callers on an event loop run it in a
        thread.

        Returns:
            The current cumulative counters.

        Raises:
            RuntimeError: If :data:`FETCH_ATTEMPTS` fetches in a row fail.
        """
        for attempt in range(1, FETCH_ATTEMPTS):
            try:
                return self._fetch()
            except OSError as e:
                logger.warning(
                    "Blend-server metrics fetch %d/%d failed (%s); retrying",
                    attempt,
                    FETCH_ATTEMPTS,
                    e,
                )
                time.sleep(FETCH_RETRY_DELAY_S)
        try:
            return self._fetch()
        except OSError as e:
            raise RuntimeError(
                f"Could not read blend-server metrics from {self._metrics_url} "
                f"after {FETCH_ATTEMPTS} attempts: {e}"
            ) from e

    def _fetch(self) -> BlendCounters:
        """Fetch and parse the endpoint once, raises ``OSError`` on failure."""
        with urllib.request.urlopen(
            self._metrics_url, timeout=_FETCH_TIMEOUT_S
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
        return parse_blend_counters(body)


class UnmeasuredBlendCounters(BlendCounterSource):
    """Stands in when no metrics URL is given -- engagement is unchecked."""

    @property
    def measures_engagement(self) -> bool:
        """Always ``False``."""
        return False

    def read(self) -> BlendCounters:
        """Return all-zero counters."""
        return BlendCounters()
