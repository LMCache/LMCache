# SPDX-License-Identifier: Apache-2.0
"""Reads LMCache cache-hit counters from the engine's Prometheus endpoint.

Without this, a run that never touched the cache and a run whose reuse was
harmless report the same F1.  The counters tell them apart.  They are absent
when the engine runs without LMCache — the baseline configuration — so their
absence is reported rather than treated as an error.
"""

# Standard
from dataclasses import dataclass
import urllib.error
import urllib.request

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# prometheus_client exposes Counter ``x`` as sample ``x_total``, alongside an
# ``x_created`` gauge that must not be summed with it.
_REQUESTED_TOKENS = "lmcache:num_requested_tokens"
_HIT_TOKENS = "lmcache:num_hit_tokens"

_FETCH_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class CacheCounters:
    """A reading of LMCache's cumulative token counters.

    Attributes:
        requested_tokens: Tokens looked up since engine start.
        hit_tokens: Tokens those lookups found cached.
        available: Whether the counters could be read.  ``False`` means the
            engine exposes no LMCache metrics and the token fields are ``0``.
    """

    requested_tokens: int
    hit_tokens: int
    available: bool

    def delta(self, earlier: "CacheCounters") -> "CacheCounters":
        """Return the counter movement since an *earlier* reading.

        Args:
            earlier: A reading taken before the activity of interest.

        Returns:
            The difference, available only when both readings were.  Clamped
            at zero so an engine restart reports no activity, not a negative.
        """
        if not (self.available and earlier.available):
            return CacheCounters(0, 0, False)
        return CacheCounters(
            requested_tokens=max(0, self.requested_tokens - earlier.requested_tokens),
            hit_tokens=max(0, self.hit_tokens - earlier.hit_tokens),
            available=True,
        )


def _metrics_url(engine_url: str) -> str:
    """Build the Prometheus endpoint URL for an engine base URL.

    The endpoint sits at the server root, so a ``/v1`` suffix is stripped.

    Args:
        engine_url: The engine's base URL, with or without a scheme.

    Returns:
        The absolute ``/metrics`` URL.
    """
    url = engine_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return f"{url}/metrics"


def _sum_samples(body: str, metric: str) -> tuple[int, bool]:
    """Sum every labelled sample of *metric* in a Prometheus text body.

    Args:
        body: The Prometheus exposition text.
        metric: Metric name without the ``_total`` suffix.

    Returns:
        A ``(total, found)`` pair.  ``found`` is False when the metric is
        absent, which differs from a metric present at zero.
    """
    total = 0.0
    found = False
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0].strip()
        if name not in (metric, f"{metric}_total"):
            continue
        try:
            total += float(value)
        except ValueError:
            continue
        found = True
    return int(total), found


class MetricsProbe:
    """Reads LMCache token counters from an engine's Prometheus endpoint."""

    def __init__(self, engine_url: str) -> None:
        """Initialize the probe.

        Args:
            engine_url: Base URL of the inference engine.
        """
        self._url = _metrics_url(engine_url)
        self._warned = False

    def read(self) -> CacheCounters:
        """Read the current counters.

        Never raises: a benchmark already collecting answers should report a
        missing diagnostic rather than abort over it.

        Returns:
            The current counters, or an unavailable reading.
        """
        try:
            request = urllib.request.Request(self._url)
            with urllib.request.urlopen(
                request, timeout=_FETCH_TIMEOUT_SECONDS
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._warn_once(f"cannot read {self._url}: {e}")
            return CacheCounters(0, 0, False)

        requested, requested_found = _sum_samples(body, _REQUESTED_TOKENS)
        hits, hits_found = _sum_samples(body, _HIT_TOKENS)
        if not (requested_found and hits_found):
            self._warn_once(
                f"{self._url} exposes no LMCache token counters; cache "
                f"activity will not be reported"
            )
            return CacheCounters(0, 0, False)

        return CacheCounters(
            requested_tokens=requested,
            hit_tokens=hits,
            available=True,
        )

    def _warn_once(self, message: str) -> None:
        """Log *message* once, so a per-request read cannot flood the output."""
        if self._warned:
            return
        self._warned = True
        logger.warning("Cache counters unavailable: %s", message)
