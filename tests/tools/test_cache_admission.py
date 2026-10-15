# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests for the bounded trace admission experiment."""

# Standard
from pathlib import Path
import hashlib
import itertools
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.tools.cache_simulator.admission import HistoricalAdmission, evaluate
from lmcache.tools.cache_simulator.simulator import simulate


def write_trace(path: Path, rows: list[dict[str, object]]) -> str:
    """Write a fixture trace and return its exact-byte manifest digest."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(blocks: list[int], timestamp: int = 0, tail: int = 0) -> dict[str, object]:
    """Build a Mooncake fixture with an optional partially filled final block."""
    return {
        "timestamp": timestamp,
        "input_length": len(blocks) * 512 - tail,
        "output_length": 1,
        "hash_ids": blocks,
    }


def run_cli(path: Path, digest: str) -> subprocess.CompletedProcess[str]:
    """Exercise the actual executable module, preserving stdout and exit status."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "lmcache.tools.cache_simulator.admission",
            str(path),
            "--expected-sha256",
            digest,
            "--capacity-chunks",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_complete_cli_reports_causal_prefix_work(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, [request([1, 2, 3], tail=100)] * 3)
    completed = run_cli(path, digest)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["qualification"] == "serialized_work_model_only"
    assert report["requests"] == 3
    assert report["input_tokens"] == 4308
    assert report["modes"]["lru"]["hit_tokens"] == 2048
    assert report["modes"]["historical"]["hit_tokens"] == 1024
    assert report["modes"]["no_offload"]["hit_tokens"] == 0
    assert report["unlimited_capacity_reuse_work_ceiling_tokens"] == 2048
    assert "current_lazy_scheduler" in report["not_modeled"]


def test_nonprefix_hits_cannot_be_added_together(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, [request([1, 2]), request([1, 2])])
    report = evaluate(path, digest, 1, 8, 10)
    assert report["modes"]["lru"]["hit_tokens"] == 0
    assert report["unlimited_capacity_reuse_work_ceiling_tokens"] == 1024


def test_history_is_bounded_and_does_not_observe_current_request() -> None:
    history = HistoricalAdmission(max_keys=2, max_request_gap=3)
    assert history.select_prefix((1, 2), 0) == 0
    history.observe((1, 2), 0)
    assert history.select_prefix((1, 2, 3), 1) == 2
    assert history.select_prefix((3, 1, 2), 1) == 0
    assert history.select_prefix((1, 2), 4) == 0
    for index in range(1000):
        history.observe((index,), index + 1)
        assert history.tracked_keys <= 2
    assert history.select_prefix((1, 2), 1001) == 0


def test_training_window_is_not_scored(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, [request([1, 2])] * 3)
    report = evaluate(path, digest, 2, 2, 10, evaluation_start=2)
    assert report["evaluation_requests"] == 1
    assert report["input_tokens"] == 1024
    assert report["modes"]["historical"]["hit_tokens"] == 1024
    assert report["modes"]["historical"]["written_blocks"] == 0


def test_baseline_matches_existing_simulator_and_work_ceiling(tmp_path: Path) -> None:
    """Exhaust small traces, comparing the reused LRU model and prefix ceiling."""
    for sequence in itertools.product(((1, 2), (1, 3), (4,)), repeat=3):
        path = tmp_path / "trace.jsonl"
        digest = write_trace(path, [request(list(blocks)) for blocks in sequence])
        events = [
            {
                "chunk_hashes": list(blocks),
                "seq_len": len(blocks) * 512,
                "chunk_size": 512,
            }
            for blocks in sequence
        ]
        for capacity in (1, 2, 3):
            report = evaluate(path, digest, capacity, 4, 10)
            baseline = simulate(events, capacity, 1, fast=True)
            assert report["modes"]["lru"]["hit_tokens"] == baseline["total_hit_tokens"]
            ceiling = report["unlimited_capacity_reuse_work_ceiling_tokens"]
            for mode in report["modes"].values():
                assert mode["hit_tokens"] <= ceiling


@pytest.mark.parametrize(
    "rows, error",
    [
        ([], "no requests"),
        ([{}], "timestamp"),
        ([request([1]), request([2], timestamp=-1)], "nondecreasing"),
        ([dict(request([1]), timestamp=float("nan"))], "finite"),
        ([dict(request([1]), timestamp=True)], "finite"),
        ([dict(request([1]), input_length=True)], "integer"),
        ([dict(request([1]), input_length=512.0)], "integer"),
        ([dict(request([1]), output_length=-1)], "integer"),
        ([dict(request([1]), hash_ids=[True])], "integer"),
        ([dict(request([1]), hash_ids="1")], "list"),
        ([dict(request([1]), hash_ids=[])], "exactly"),
        ([request([1, 2]), request([3, 2])], "ancestry"),
        ([request([1] * 4097)], "block limit"),
    ],
)
def test_cli_rejects_invalid_evidence(
    tmp_path: Path, rows: list[dict[str, object]], error: str
) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, rows)
    completed = run_cli(path, digest)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert error in completed.stderr


def test_digest_drift_and_partial_json_cannot_produce_success(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, [request([1]), request([1])])
    path.write_text(json.dumps(request([1])) + "\n", encoding="utf-8")
    completed = run_cli(path, digest)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "SHA-256" in completed.stderr
    path.write_text('{"timestamp":', encoding="utf-8")
    completed = run_cli(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "capacity, history, gap, start",
    [(0, 1, 1, 0), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, -1), (1, 1, 1, 1)],
)
def test_invalid_or_empty_evaluation_fails(
    tmp_path: Path, capacity: int, history: int, gap: int, start: int
) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace(path, [request([1])])
    with pytest.raises(ValueError):
        evaluate(path, digest, capacity, history, gap, start)
