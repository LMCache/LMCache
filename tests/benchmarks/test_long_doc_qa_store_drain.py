# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``benchmarks/long_doc_qa/lmcache_store_drain.py``.

The helper keeps the long_doc_qa benchmark from reporting a cache miss that is
really L2 stores still in flight (issue #6794). The tests use scripted
``/metrics`` text and a fake clock, plus one end-to-end test against a real
local HTTP server.
"""

# Standard
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
import importlib.util
import math
import socket
import sys
import threading

# Third Party
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "long_doc_qa"
    / "lmcache_store_drain.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lmcache_store_drain", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drain = _load_module()

NOT_LMCACHE = "# HELP process_cpu_seconds_total CPU\nprocess_cpu_seconds_total 1.0\n"


def _metrics(
    submitted: float | None = None,
    completed: float | None = None,
    hit: float | None = None,
    requested: float | None = None,
) -> str:
    """Render ``/metrics`` text from an LMCache MP server; None omits a series."""
    lines = ["# HELP lmcache_mp_build_info build", "lmcache_mp_build_info 1.0"]
    for name, value in (
        (drain.STORE_SUBMITTED_METRIC, submitted),
        (drain.STORE_COMPLETED_METRIC, completed),
        (drain.LOOKUP_HIT_METRIC, hit),
        (drain.LOOKUP_REQUESTED_METRIC, requested),
    ):
        if value is not None:
            lines.append(f"# TYPE {name} counter")
            lines.append(f'{name}{{cache_salt="a"}} {value}')
    return "\n".join(lines) + "\n"


class _Clock:
    """A fake clock whose ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Script:
    """A ``fetch`` that replays ``/metrics`` texts, repeating the last one."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    def __call__(self) -> str:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return text


def _wait(fetch, clock: _Clock, baseline: float = 10.0, **overrides):
    kwargs = dict(
        timeout=100.0,
        poll_interval=1.0,
        settle_seconds=3.0,
        no_new_stores_grace=30.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    kwargs.update(overrides)
    return drain.wait_for_store_drain(fetch, baseline, **kwargs)


def test_parse_counter_sums_label_series_and_ignores_lookalikes() -> None:
    name = drain.STORE_SUBMITTED_METRIC
    text = (
        f"# HELP {name} chunks\n"
        f"# TYPE {name} counter\n"
        f'{name}{{cache_salt="a"}} 4.0\n'
        f'{name}{{cache_salt="b"}} 1.5e1\n'
        f"{name}_created 1.7e9\n"
        f"{name}_other 99\n"
    )
    assert drain.parse_counter(text, name) == pytest.approx(19.0)


def test_parse_counter_returns_none_when_no_series() -> None:
    assert drain.parse_counter(_metrics(), drain.STORE_SUBMITTED_METRIC) is None


def test_store_snapshot_counts_missing_series_as_zero_on_an_lmcache_endpoint() -> None:
    snap = drain.read_store_snapshot(lambda: _metrics())
    assert snap == drain.StoreSnapshot(submitted=0.0, completed=0.0)


@pytest.mark.parametrize("reader", ["read_store_snapshot", "read_lookup_snapshot"])
def test_snapshots_reject_a_non_lmcache_endpoint(reader: str) -> None:
    with pytest.raises(drain.MetricsEndpointError):
        getattr(drain, reader)(lambda: NOT_LMCACHE)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ((100, 1000), (400, 2000), 0.3),
        ((0, 0), (0, 0), None),
        ((50, 500), (50, 500), None),
    ],
)
def test_hit_rate_between_uses_the_interval_not_the_totals(
    before: tuple[int, int], after: tuple[int, int], expected: float | None
) -> None:
    rate = drain.hit_rate_between(
        drain.LookupSnapshot(*before), drain.LookupSnapshot(*after)
    )
    assert rate == (pytest.approx(expected) if expected is not None else None)


def test_wait_returns_drained_once_completed_catches_up_and_settles() -> None:
    clock = _Clock()
    fetch = _Script(
        [
            _metrics(10, 10),
            _metrics(14, 10),
            _metrics(14, 12),
            _metrics(14, 14),
        ]
    )
    result = _wait(fetch, clock)
    assert result.drained
    assert result.snapshot == drain.StoreSnapshot(submitted=14, completed=14)
    # The three polls before completion cannot end the wait, and the settle
    # window then needs 3 more seconds of unchanged counters.
    assert result.waited_seconds >= 3.0 + 3.0


def test_equal_counters_at_the_baseline_are_not_drained() -> None:
    """Before the last store is submitted, submitted == completed at an old value."""
    clock = _Clock()
    quiet = [_metrics(10, 10)] * 8
    fetch = _Script(quiet + [_metrics(14, 10), _metrics(14, 14)])
    result = _wait(fetch, clock)
    assert result.drained
    assert result.snapshot.submitted == 14
    assert fetch.calls > len(quiet)


def test_a_new_submission_restarts_the_settle_window() -> None:
    clock = _Clock()
    fetch = _Script(
        [_metrics(14, 14), _metrics(14, 14), _metrics(16, 14), _metrics(16, 16)]
    )
    result = _wait(fetch, clock, settle_seconds=3.0)
    assert result.drained
    assert result.snapshot.submitted == 16
    # Two polls were already stable, but the later submission restarts the
    # window, so the wait lasts at least 3 s past the last counter change.
    assert result.waited_seconds >= 3.0 + 3.0


def test_wait_gives_up_at_the_timeout_when_stores_never_finish() -> None:
    clock = _Clock()
    result = _wait(_Script([_metrics(14, 10)]), clock, timeout=10.0)
    assert not result.drained
    assert result.waited_seconds >= 10.0
    assert result.snapshot == drain.StoreSnapshot(submitted=14, completed=10)


def test_wait_gives_up_early_when_no_new_stores_appear() -> None:
    clock = _Clock()
    result = _wait(
        _Script([_metrics(10, 10)]), clock, timeout=100.0, no_new_stores_grace=5.0
    )
    assert not result.drained
    assert 5.0 <= result.waited_seconds < 100.0


def test_wait_raises_when_the_endpoint_is_not_lmcache() -> None:
    with pytest.raises(drain.MetricsEndpointError):
        _wait(_Script([NOT_LMCACHE]), _Clock())


@pytest.mark.parametrize(
    ("warmup", "query", "expect_warning"),
    [
        (10.0, 9.8, True),
        (10.0, 1.0, False),
        (10.0, 0.0, False),
        (0.0, 1.0, False),
        (math.nan, 1.0, False),
    ],
)
def test_low_speedup_warning(warmup: float, query: float, expect_warning: bool) -> None:
    message = drain.low_speedup_warning(warmup, query, threshold=1.2)
    assert (message is not None) == expect_warning


@pytest.mark.parametrize(
    ("sleep_seconds", "url", "expect_warning"),
    [
        (0.0, None, True),
        (45.0, None, False),
        (0.0, "http://localhost:8080", False),
    ],
)
def test_missing_wait_warning(
    sleep_seconds: float, url: str | None, expect_warning: bool
) -> None:
    message = drain.missing_wait_warning(sleep_seconds, url)
    assert (message is not None) == expect_warning


@pytest.fixture
def metrics_server() -> Iterator[tuple[str, list[str]]]:
    """Serve ``/metrics`` from a list of texts, advancing on every request."""
    texts: list[str] = []
    served = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            body = texts[min(served["count"], len(texts) - 1)].encode()
            served["count"] += 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", texts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_wait_against_a_real_metrics_endpoint(
    metrics_server: tuple[str, list[str]],
) -> None:
    url, texts = metrics_server
    texts.extend(
        [_metrics(10, 10), _metrics(14, 10), _metrics(14, 13), _metrics(14, 14)]
    )

    def fetch() -> str:
        return drain.fetch_metrics(url)

    result = drain.wait_for_store_drain(
        fetch,
        baseline_submitted=10,
        timeout=10.0,
        poll_interval=0.01,
        settle_seconds=0.05,
        no_new_stores_grace=5.0,
    )
    assert result.drained
    assert result.snapshot == drain.StoreSnapshot(submitted=14, completed=14)


def _unused_url() -> str:
    """A URL on a local port that nothing is listening on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_fetch_metrics_reports_an_unreachable_server() -> None:
    with pytest.raises(drain.MetricsEndpointError):
        drain.fetch_metrics(_unused_url(), timeout=1.0)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1"])
def test_counter_values_must_be_finite_and_non_negative(value: str) -> None:
    with pytest.raises(drain.MetricsEndpointError):
        drain.read_store_snapshot(lambda: _metrics(value, value))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "texts",
    [
        # The server restarts, then climbs back to equal counters.
        [_metrics(15, 10), _metrics(0, 0), _metrics(5, 5)],
        # A label series disappears, so the summed completed counter drops.
        [_metrics(15, 10), _metrics(15, 4), _metrics(15, 15)],
    ],
)
def test_wait_does_not_report_drained_when_a_counter_decreases(
    texts: list[str],
) -> None:
    result = _wait(_Script(texts), _Clock(), baseline=0.0)
    assert not result.drained
    assert "decreased" in result.reason


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((10, 10), (1, 20)),  # hit tokens went down
        ((10, 10), (30, 20)),  # more hit tokens than requested tokens
        ((10, 10), (10, 5)),  # requested tokens went down
    ],
)
def test_hit_rate_between_rejects_an_impossible_interval(
    before: tuple[int, int], after: tuple[int, int]
) -> None:
    assert (
        drain.hit_rate_between(
            drain.LookupSnapshot(*before), drain.LookupSnapshot(*after)
        )
        is None
    )
