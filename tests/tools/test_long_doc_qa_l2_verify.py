# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the long_doc_qa L2 verification stage.

``long-doc-qa-l2-verify.sh`` must run every check and report all of them
before exiting, so a flaky threshold verdict still leaves the /metrics
evidence behind. These tests drive it with stub benchmark results and a stub
/metrics endpoint instead of GPUs and a real LMCache server.
"""

# Standard
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import functools
import json
import os
import shutil
import socket
import subprocess
import threading

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = (
    ROOT
    / ".buildkite/k3_tests/multiprocess/scripts/workloads/common"
    / "long-doc-qa-l2-verify.sh"
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None,
    reason="the verify script needs bash and curl",
)

# Baseline aggregate the stub L2 results are compared against.
BASELINE_RESULT = {
    "query_ttft_per_prompt": 1.30,
    "query_round_time_per_prompt": 4.0,
    "warmup_round_time_per_prompt": 5.0,
}

# Every counter and histogram the data-flow and observability checks read,
# all advanced, with the OTel unit suffix the Prometheus bridge appends.
HEALTHY_METRICS = """\
lmcache_mp_l1_write_chunks_total 1190
lmcache_mp_l2_store_submitted_objects_chunks_total 1190
lmcache_mp_l2_store_completed_objects_chunks_total 1190
lmcache_mp_l2_prefetch_lookup_requests_total 95
lmcache_mp_lookup_hit_l2_keys_total 2340
lmcache_mp_l2_prefetch_load_completed_chunks_total 2340
lmcache_mp_l2_store_completed_requests_total{l2_name="mock"} 95
lmcache_mp_l2_load_completed_requests_total{l2_name="mock"} 95
lmcache_mp_lookup_requested_tokens_total{model_name="Qwen/Qwen3-14B"} 600000
lmcache_mp_lookup_hit_tokens_total{model_name="Qwen/Qwen3-14B"} 300000
lmcache_mp_num_chunks_loaded_total{worker_id="0"} 2340
lmcache_mp_l0_l1_store_throughput_GB_per_second_count 60
lmcache_mp_l0_l1_load_throughput_GB_per_second_count 60
lmcache_mp_l2_store_throughput_GB_per_second_count{l2_name="mock"} 60
lmcache_mp_l2_load_throughput_GB_per_second_count{l2_name="mock"} 60
"""

# A server that answered but recorded no L2 activity at all.
IDLE_METRICS = "\n".join(
    line.rsplit(" ", 1)[0] + " 0" for line in HEALTHY_METRICS.splitlines()
)


def l2_result(ttft_speedup: float) -> dict:
    """An L2 aggregate that passes the query-speedup and warmup-overhead
    thresholds; ``ttft_speedup`` decides the TTFT verdict."""
    return {
        "query_ttft_per_prompt": BASELINE_RESULT["query_ttft_per_prompt"]
        / ttft_speedup,
        "query_round_time_per_prompt": 2.0,
        "warmup_round_time_per_prompt": 5.0,
    }


def write_results(results_dir: Path, l2: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline_result.json").write_text(json.dumps(BASELINE_RESULT))
    (results_dir / "l2_result.json").write_text(json.dumps(l2))


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def metrics_root(tmp_path: Path) -> Path:
    root = tmp_path / "metrics_root"
    root.mkdir()
    return root


@pytest.fixture
def metrics_port(metrics_root: Path):
    """Serve ``metrics_root/metrics`` as GET /metrics on a loopback port."""
    handler = functools.partial(_QuietHandler, directory=str(metrics_root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def unreachable_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_verify(results_dir: Path, port: int) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "L2_RESULTS_DIR": str(results_dir),
        "METRICS_HTTP_PORT": str(port),
        # Bound the scrape so an unreachable endpoint fails in one attempt.
        "METRICS_FETCH_ATTEMPTS": "1",
        "METRICS_FETCH_INTERVAL": "0",
    }
    return subprocess.run(
        ["bash", str(VERIFY_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def summary(results_dir: Path) -> dict:
    lines = (results_dir / "verification_summary.txt").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def test_all_checks_pass(tmp_path: Path, metrics_root: Path, metrics_port: int):
    """A healthy run passes every check and exits zero."""
    results_dir = tmp_path / "long_doc_qa_l2"
    write_results(results_dir, l2_result(ttft_speedup=2.0))
    (metrics_root / "metrics").write_text(HEALTHY_METRICS)

    proc = run_verify(results_dir, metrics_port)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert summary(results_dir) == {
        "performance": "PASS",
        "metrics_snapshot": "PASS",
        "data_flow": "PASS",
        "observability": "PASS",
        "overall": "PASS",
    }
    assert (results_dir / "prometheus_metrics.txt").read_text() == HEALTHY_METRICS


def test_threshold_failure_still_collects_metrics(
    tmp_path: Path, metrics_root: Path, metrics_port: int
):
    """The CI flake: TTFT 1.49x against a 1.5x floor. The verdict stays a
    failure, and the L2 data-flow evidence is collected and checked anyway."""
    results_dir = tmp_path / "long_doc_qa_l2"
    write_results(results_dir, l2_result(ttft_speedup=1.49))
    (metrics_root / "metrics").write_text(HEALTHY_METRICS)

    proc = run_verify(results_dir, metrics_port)

    assert proc.returncode != 0
    assert "L2 TTFT speedup: 1.49x" in proc.stdout
    assert "Data Flow Metrics" in proc.stdout
    assert summary(results_dir) == {
        "performance": "FAIL",
        "metrics_snapshot": "PASS",
        "data_flow": "PASS",
        "observability": "PASS",
        "overall": "FAIL",
    }
    assert (results_dir / "prometheus_metrics.txt").read_text() == HEALTHY_METRICS


@pytest.mark.parametrize(
    ("ttft_speedup", "performance"), [(2.0, "PASS"), (1.49, "FAIL")]
)
def test_unreachable_metrics_is_reported_beside_the_performance_verdict(
    tmp_path: Path, ttft_speedup: float, performance: str
):
    """No /metrics answer is a visible collection error that fails the run
    without hiding whatever the threshold check said."""
    results_dir = tmp_path / "long_doc_qa_l2"
    write_results(results_dir, l2_result(ttft_speedup))

    proc = run_verify(results_dir, unreachable_port())

    assert proc.returncode != 0
    assert "could not fetch /metrics" in proc.stdout
    result = summary(results_dir)
    assert result["performance"] == performance
    assert result["metrics_snapshot"].startswith("FAIL (could not fetch /metrics")
    assert "after 1 attempt(s)" in result["metrics_snapshot"]
    assert result["data_flow"] == "SKIPPED"
    assert result["observability"] == "SKIPPED"
    assert result["overall"] == "FAIL"
    assert (results_dir / "prometheus_metrics.txt").read_text() == ""


def test_idle_metrics_fail_data_flow_after_a_passing_threshold(
    tmp_path: Path, metrics_root: Path, metrics_port: int
):
    """Fast numbers with no recorded L2 traffic is a data-flow failure, not
    a pass: the snapshot is kept so the zero counters can be inspected."""
    results_dir = tmp_path / "long_doc_qa_l2"
    write_results(results_dir, l2_result(ttft_speedup=2.0))
    (metrics_root / "metrics").write_text(IDLE_METRICS)

    proc = run_verify(results_dir, metrics_port)

    assert proc.returncode != 0
    result = summary(results_dir)
    assert result["performance"] == "PASS"
    assert result["metrics_snapshot"] == "PASS"
    assert result["data_flow"] == "FAIL"
    assert result["observability"] == "FAIL"
    assert result["overall"] == "FAIL"
    assert (results_dir / "prometheus_metrics.txt").read_text() == IDLE_METRICS
