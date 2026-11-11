# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression coverage for the L1 long-document QA wrapper.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_long_doc_qa.py
"""

# Standard
from pathlib import Path
from typing import Literal
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys

# Third Party
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = (
    REPO_ROOT
    / ".buildkite/k3_tests/multiprocess/scripts/workloads/common/long-doc-qa.sh"
)
STUB_BENCHMARK = Path(__file__).parent / "fixtures/long_doc_qa.py"
DIAGNOSTICS = WORKLOAD.with_name("long_doc_qa_diagnostics.py")


def _collect_workload_artifacts(workspace: Path) -> set[Path]:
    """Apply the L1 job's artifact globs to a simulated cleanup workspace."""
    pipeline = yaml.safe_load(
        (REPO_ROOT / ".buildkite/k3_tests/multiprocess/pipeline.yml").read_text(
            encoding="utf-8"
        )
    )
    jobs = [
        step
        for group in pipeline["steps"]
        if isinstance(group, dict)
        for step in group.get("steps", [])
        if step.get("command") == ".buildkite/k3_tests/multiprocess/run.sh long_doc_qa"
    ]
    assert len(jobs) == 1
    return {
        path
        for pattern in jobs[0]["artifact_paths"]
        for path in workspace.glob(pattern)
        if path.is_file()
    }


def _run_workload(
    tmp_path: Path,
    query_ttft: float,
    query_round_time: float,
    path_style: Literal["absolute", "relative"],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the real wrapper with isolated files and deterministic benchmark data."""
    working_dir = tmp_path / "caller directory"
    working_dir.mkdir()
    stub_repo = working_dir / "stub repo"
    benchmark = stub_repo / "benchmarks/long_doc_qa/long_doc_qa.py"
    benchmark.parent.mkdir(parents=True)
    shutil.copyfile(STUB_BENCHMARK, benchmark)
    results_dir = working_dir / "test results"

    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "INFERENCE_ENGINE": "vllm",
        "ENGINE_PORT": "8000",
        "ENGINE_BASELINE_PORT": "9000",
        "MODEL": "stub-model",
        "BUILD_ID": "long_doc_qa_cpu_test",
        "LMCACHE_DIR": str(stub_repo),
        "RESULTS_DIR": str(results_dir),
        "DOCUMENT_LENGTH": "10",
        "NUM_DOCUMENTS": "2",
        "OUTPUT_LEN": "3",
        "REPEAT_COUNT": "2",
        "REPEAT_MODE": "tile",
        "SHUFFLE_SEED": "0",
        "MAX_INFLIGHT_REQUESTS": "2",
        "MAX_TTFT_SLOWDOWN_PCT": "-60",
        "MAX_ROUND_TIME_SLOWDOWN_PCT": "-15",
        "STUB_QUERY_TTFT": str(query_ttft),
        "STUB_QUERY_ROUND_TIME": str(query_round_time),
    }
    if path_style == "relative":
        environment["LMCACHE_DIR"] = str(stub_repo.relative_to(working_dir))
        environment["RESULTS_DIR"] = str(results_dir.relative_to(working_dir))

    result = subprocess.run(
        ["bash", str(WORKLOAD)],
        cwd=working_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, results_dir / "long_doc_qa"


@pytest.mark.parametrize("path_style", ["absolute", "relative"])
@pytest.mark.parametrize(
    ("query_ttft", "query_round_time", "expected_exit_code"),
    [
        pytest.param(0.30, 0.80, 0, id="thresholds-pass"),
        pytest.param(0.50, 0.80, 1, id="ttft-fails"),
        pytest.param(0.30, 0.90, 1, id="round-time-fails"),
    ],
)
def test_preserves_both_phases_after_threshold_check(
    tmp_path: Path,
    path_style: Literal["absolute", "relative"],
    query_ttft: float,
    query_round_time: float,
    expected_exit_code: int,
) -> None:
    """Both phases' outputs and diagnostics survive either threshold verdict.

    Verify the job log and the pipeline's artifact selection as well as the
    files, so a container-local results path alone cannot satisfy this contract.
    """
    result, results_dir = _run_workload(
        tmp_path, query_ttft, query_round_time, path_style
    )
    assert result.returncode == expected_exit_code, result.stdout + result.stderr
    verdict = (
        "All thresholds passed"
        if expected_exit_code == 0
        else "Threshold verification FAILED"
    )
    assert verdict in result.stdout

    for phase, ttft, round_time in (
        ("baseline", 1.0, 1.0),
        ("lmcache", query_ttft, query_round_time),
    ):
        # The L2 wrapper also consumes the existing baseline JSON path.
        aggregate = results_dir / f"{phase}_result.json"
        summary = json.loads(aggregate.read_text(encoding="utf-8").splitlines()[-1])
        assert summary == {
            "query_ttft_per_prompt": ttft,
            "query_round_time_per_prompt": round_time,
            "warmup_round_time_per_prompt": 1.2,
        }
        output = (results_dir / f"{phase}_output.txt").read_text(encoding="utf-8")
        assert f"{phase} responses" in output
        assert f"{phase} stderr" in output

        diagnostics = (results_dir / phase / "diagnostics.txt").read_text(
            encoding="utf-8"
        )
        assert diagnostics in result.stdout
        assert f"=== long_doc_qa diagnostics: {phase} ===" in diagnostics
        assert "engine: vllm" in diagnostics
        assert "untrimmed; p95=nearest-rank" in diagnostics
        command_line = next(
            line.removeprefix("command: ")
            for line in diagnostics.splitlines()
            if line.startswith("command: ")
        )
        command = shlex.split(command_line)
        for flag, value in {
            "--port": "9000" if phase == "baseline" else "8000",
            "--model": "stub-model",
            "--document-length": "10",
            "--num-documents": "2",
            "--output-len": "3",
            "--repeat-count": "2",
            "--repeat-mode": "tile",
            "--shuffle-seed": "0",
            "--max-inflight-requests": "2",
        }.items():
            assert command[command.index(flag) + 1] == value
        assert "warmup: samples=2 successful=2 failed=0" in diagnostics
        assert "query: samples=4 successful=4 failed=0" in diagnostics
        assert (
            f"ttft: min={ttft:.6f} median={ttft:.6f} p95={ttft:.6f} max={ttft:.6f}"
        ) in diagnostics

    for round_name, count in (("warmup", 2), ("query", 4)):
        csv_files = sorted(results_dir.rglob(f"{round_name}_round.csv"))
        assert len(csv_files) == 2, (
            f"Expected both phases' {round_name} CSVs under {results_dir}; "
            f"found {csv_files}"
        )
        observed_phases: set[str] = set()
        for csv_file in csv_files:
            phase = csv_file.parent.name
            assert phase in {"baseline", "lmcache"}, csv_file
            assert phase not in observed_phases, f"Duplicate {phase} CSV: {csv_file}"
            observed_phases.add(phase)
            with csv_file.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            offset = 100 if phase == "baseline" else 200
            expected_ttft = 0.6
            if round_name == "query":
                expected_ttft = 1.0 if phase == "baseline" else query_ttft
            assert [int(row["prompt_id"]) for row in rows] == list(
                range(offset, offset + count)
            )
            assert [float(row["ttft"]) for row in rows] == [expected_ttft] * count
            assert all(row["successful"] == "True" for row in rows)

    workspace = tmp_path / "workspace"
    staged = workspace / "ci_results_cpu_test" / "long_doc_qa"
    shutil.copytree(results_dir, staged)
    server_log = workspace / "build_cpu_test_vllm.log"
    server_log.write_text("server log\n", encoding="utf-8")
    expected_artifacts = {path for path in staged.rglob("*") if path.is_file()}
    expected_artifacts.add(server_log)
    for workload in ("long_doc_qa_l2", "high_concurrency"):
        unrelated = staged.parent / workload / "query_round.csv"
        unrelated.parent.mkdir()
        unrelated.write_text("unrelated workload\n", encoding="utf-8")
    assert _collect_workload_artifacts(workspace) == expected_artifacts
