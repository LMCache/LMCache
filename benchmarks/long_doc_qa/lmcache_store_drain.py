# SPDX-License-Identifier: Apache-2.0
"""Keep ``long_doc_qa`` honest about L2 stores that are still in flight.

The benchmark's warm-up round makes LMCache write the documents' KV to L2
asynchronously. With a large working set those writes are still running when
the query round starts, so the query round misses and the benchmark reports
almost no speed-up, with nothing in its output saying the run is invalid
(issue #6794).

This module reads the MP server's Prometheus ``/metrics`` endpoint to wait for
the writes to finish, to report the LMCache lookup hit rate of the query round,
and to produce the warnings the benchmark prints when a result looks
unreliable. It uses only the standard library.
"""

# Standard
from dataclasses import dataclass
from typing import Callable, Optional
import math
import re
import time
import urllib.request

# Prometheus names of the MP server's OTel counters; see
# docs/design/v1/mp_observability/METRICS.md. A counter with several label
# series (for example one per ``cache_salt``) is summed.
STORE_SUBMITTED_METRIC = "lmcache_mp_l2_store_submitted_objects_chunks_total"
STORE_COMPLETED_METRIC = "lmcache_mp_l2_store_completed_objects_chunks_total"
LOOKUP_HIT_METRIC = "lmcache_mp_lookup_hit_tokens_total"
LOOKUP_REQUESTED_METRIC = "lmcache_mp_lookup_requested_tokens_total"


class MetricsEndpointError(RuntimeError):
    """The LMCache ``/metrics`` endpoint is unreachable or is not LMCache's."""


@dataclass(frozen=True)
class StoreSnapshot:
    """Cumulative L2 store counters, in chunks, at one moment.

    Attributes:
        submitted: Chunks handed to L2 for storing.
        completed: Chunks L2 reported as stored.
    """

    submitted: float
    completed: float


@dataclass(frozen=True)
class LookupSnapshot:
    """Cumulative LMCache lookup counters, in tokens, at one moment.

    Attributes:
        hit_tokens: Tokens LMCache found.
        requested_tokens: Tokens LMCache was asked about.
    """

    hit_tokens: float
    requested_tokens: float


@dataclass(frozen=True)
class DrainResult:
    """Outcome of :func:`wait_for_store_drain`.

    Attributes:
        drained: True when every submitted store had completed and the
            counters stayed unchanged for the settle window.
        reason: One sentence saying why the wait ended.
        waited_seconds: Time spent waiting.
        snapshot: The last counters read.
    """

    drained: bool
    reason: str
    waited_seconds: float
    snapshot: StoreSnapshot


def fetch_metrics(base_url: str, timeout: float = 5.0) -> str:
    """Return the text of ``GET <base_url>/metrics``.

    Args:
        base_url: Base URL of the LMCache MP HTTP server, for example
            ``http://localhost:8080``.
        timeout: Seconds before the request gives up.

    Returns:
        The Prometheus exposition text.

    Raises:
        MetricsEndpointError: If the endpoint cannot be reached.
    """
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise MetricsEndpointError(
            f"Cannot read {url}: {exc}. --lmcache-url must be the base URL of the "
            "LMCache MP HTTP server (port 8080 by default)."
        ) from exc


def parse_counter(text: str, name: str) -> Optional[float]:
    """Sum every series of the counter ``name`` in Prometheus text.

    Args:
        text: Prometheus exposition text.
        name: Exact metric name. A longer name that merely starts with it
            (such as a ``_created`` companion series) does not match.

    Returns:
        The sum over all label sets, or None if no series has this name.

    Raises:
        MetricsEndpointError: If a series is NaN, infinite or negative.
    """
    pattern = re.compile(
        r"^" + re.escape(name) + r"(?:\{[^}]*\})?[ \t]+(\S+)", re.MULTILINE
    )
    total: Optional[float] = None
    for match in pattern.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            raise MetricsEndpointError(
                f"{name} has the invalid value {match.group(1)!r}; a counter must be "
                "finite and not negative."
            )
        total = (total or 0.0) + value
    return total


def read_store_snapshot(fetch: Callable[[], str]) -> StoreSnapshot:
    """Read the cumulative L2 store counters.

    A counter with no series yet counts as 0, because a counter appears only
    after its first increment. That is trusted only once the text shows it is
    an LMCache MP endpoint.

    Args:
        fetch: Returns the current ``/metrics`` text.

    Returns:
        The current counters.

    Raises:
        MetricsEndpointError: If the text has no ``lmcache_mp_`` metric at all,
            so the URL points at the wrong server or port.
    """
    text = _require_lmcache_metrics(fetch())
    return StoreSnapshot(
        submitted=parse_counter(text, STORE_SUBMITTED_METRIC) or 0.0,
        completed=parse_counter(text, STORE_COMPLETED_METRIC) or 0.0,
    )


def read_lookup_snapshot(fetch: Callable[[], str]) -> LookupSnapshot:
    """Read the cumulative LMCache lookup counters.

    Args:
        fetch: Returns the current ``/metrics`` text.

    Returns:
        The current counters; a counter with no series yet counts as 0.

    Raises:
        MetricsEndpointError: If the text has no ``lmcache_mp_`` metric.
    """
    text = _require_lmcache_metrics(fetch())
    return LookupSnapshot(
        hit_tokens=parse_counter(text, LOOKUP_HIT_METRIC) or 0.0,
        requested_tokens=parse_counter(text, LOOKUP_REQUESTED_METRIC) or 0.0,
    )


def hit_rate_between(before: LookupSnapshot, after: LookupSnapshot) -> Optional[float]:
    """Return the LMCache lookup hit rate between two snapshots.

    Args:
        before: Snapshot taken before the round.
        after: Snapshot taken after the round.

    Returns:
        ``hit tokens / requested tokens`` over the interval, or None if no
        tokens were requested or the counters are inconsistent (a counter went
        down, or there were more hit tokens than requested tokens).
    """
    requested = after.requested_tokens - before.requested_tokens
    hits = after.hit_tokens - before.hit_tokens
    if requested <= 0 or not 0 <= hits <= requested:
        return None
    return hits / requested


def wait_for_store_drain(
    fetch: Callable[[], str],
    baseline_submitted: float,
    *,
    timeout: float,
    poll_interval: float,
    settle_seconds: float,
    no_new_stores_grace: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> DrainResult:
    """Wait until the stores submitted since ``baseline_submitted`` finish.

    The wait ends as drained only when all of these hold: more chunks were
    submitted than at the baseline, every submitted chunk has completed, and
    both counters stayed unchanged for ``settle_seconds``. Equal counters alone
    are not enough: before the last store is submitted they sit at an equal,
    older value.

    Args:
        fetch: Returns the current ``/metrics`` text.
        baseline_submitted: ``submitted`` read before the round whose stores
            are awaited.
        timeout: Give up after this many seconds.
        poll_interval: Seconds between reads.
        settle_seconds: How long drained-looking counters must stay unchanged.
        no_new_stores_grace: Give up early if nothing was submitted after the
            baseline within this many seconds (L2 not configured, or the round
            stored nothing).
        sleep: Sleep function; replaceable for tests.
        monotonic: Clock; replaceable for tests.

    Returns:
        A :class:`DrainResult`. ``drained`` is False on timeout, when no new
        stores appeared, or when a counter decreased; the caller decides how
        loudly to report that.

    Raises:
        MetricsEndpointError: If a read shows the endpoint is not LMCache's,
            or a counter value is invalid.
    """
    start = monotonic()
    candidate: Optional[StoreSnapshot] = None
    candidate_since = 0.0
    previous: Optional[StoreSnapshot] = None
    while True:
        snapshot = read_store_snapshot(fetch)
        now = monotonic()
        waited = now - start
        if previous is not None and (
            snapshot.submitted < previous.submitted
            or snapshot.completed < previous.completed
        ):
            return DrainResult(
                False,
                "the L2 store counters decreased during the wait (server restart "
                "or a metric series disappeared), so the warm-up's stores cannot "
                "be verified",
                waited,
                snapshot,
            )
        previous = snapshot
        has_new_stores = snapshot.submitted > baseline_submitted
        if has_new_stores and snapshot.completed >= snapshot.submitted:
            if snapshot != candidate:
                candidate, candidate_since = snapshot, now
            if now - candidate_since >= settle_seconds:
                return DrainResult(
                    True,
                    "every submitted L2 store completed and the counters settled",
                    waited,
                    snapshot,
                )
        else:
            candidate = None
        if not has_new_stores and waited >= no_new_stores_grace:
            return DrainResult(
                False,
                "no new L2 store submissions were observed; is an L2 adapter "
                "configured?",
                waited,
                snapshot,
            )
        if waited >= timeout:
            pending = snapshot.submitted - snapshot.completed
            return DrainResult(
                False,
                f"timed out with {pending:.0f} submitted chunks not yet stored",
                waited,
                snapshot,
            )
        sleep(poll_interval)


def low_speedup_warning(
    warmup_ttft: float, query_ttft: float, threshold: float
) -> Optional[str]:
    """Return a warning if the query round is barely faster than warm-up.

    Args:
        warmup_ttft: Mean TTFT of the warm-up round, in seconds.
        query_ttft: Mean TTFT of the query round, in seconds.
        threshold: Minimum warm-up/query ratio considered healthy.

    Returns:
        A warning message, or None if the ratio reaches ``threshold`` or either
        TTFT is not a positive number.
    """
    if not (warmup_ttft > 0 and query_ttft > 0):
        return None
    ratio = warmup_ttft / query_ttft
    if ratio >= threshold:
        return None
    return (
        f"WARNING: the query round's mean TTFT is only {ratio:.2f}x lower than the "
        f"warm-up round's (expected at least {threshold:g}x). If the cache should "
        "have helped, the warm-up round's L2 stores may not have finished before "
        "the query round started. Pass --lmcache-url so the benchmark waits for "
        "them, or raise --sleep-time-after-warmup."
    )


def missing_wait_warning(
    sleep_time_after_warmup: float, lmcache_url: Optional[str]
) -> Optional[str]:
    """Return a warning if nothing waits for the warm-up stores.

    Args:
        sleep_time_after_warmup: Value of ``--sleep-time-after-warmup``.
        lmcache_url: Value of ``--lmcache-url``, or None.

    Returns:
        A warning message when there is no sleep and no ``--lmcache-url``,
        otherwise None.
    """
    if sleep_time_after_warmup > 0 or lmcache_url:
        return None
    return (
        "WARNING: --sleep-time-after-warmup is 0 and --lmcache-url is not set, so "
        "the query round starts as soon as the warm-up round ends. With a large "
        "working set the warm-up round's L2 stores can still be in flight and the "
        "query round then misses. Pass --lmcache-url to wait for the stores."
    )


def _require_lmcache_metrics(text: str) -> str:
    """Return ``text`` if it came from an LMCache MP endpoint, else raise."""
    if not re.search(r"^lmcache_mp_", text, re.MULTILINE):
        raise MetricsEndpointError(
            "The /metrics response has no lmcache_mp_* metric. --lmcache-url must "
            "point at the LMCache MP HTTP server, and its metrics must be enabled."
        )
    return text
