# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Third Party
import pytest


def get_bash_executable() -> str | None:
    """
    Find an available bash executable on the current system.

    Returns:
        Path to bash executable, or None if not found.
    """
    if os.name == "nt":
        for git_bash in [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]:
            if os.path.exists(git_bash):
                return git_bash

    bash = shutil.which("bash")
    if bash:
        if os.name == "nt" and (
            "windowsapps" in bash.lower() or "system32" in bash.lower()
        ):
            return None
        return bash
    return None


def get_workload_script_path() -> str:
    """
    Locate the long-doc-qa.sh workload script relative to the repo root.

    Returns:
        Absolute path to long-doc-qa.sh.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(
        repo_root,
        ".buildkite",
        "k3_tests",
        "multiprocess",
        "scripts",
        "workloads",
        "common",
        "long-doc-qa.sh",
    )
    return script_path


def create_mock_benchmark_script(
    script_path: str,
    baseline_ttft: float = 1.30,
    lmcache_ttft: float = 0.39,
    baseline_round_time: float = 1.20,
    lmcache_round_time: float = 0.85,
) -> None:
    """
    Create a mock long_doc_qa.py script that generates distinguishable
    phase CSV files and writes aggregate JSON summary to stdout.

    Args:
        script_path: Full path where mock long_doc_qa.py will be written.
        baseline_ttft: TTFT value to record for baseline server.
        lmcache_ttft: TTFT value to record for LMCache server.
        baseline_round_time: Query round time per prompt for baseline.
        lmcache_round_time: Query round time per prompt for LMCache.
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    content = f"""#!/usr/bin/env python3
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8000")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--document-length", type=int, default=10000)
parser.add_argument("--num-documents", type=int, default=30)
parser.add_argument("--output-len", type=int, default=200)
parser.add_argument("--repeat-count", type=int, default=2)
parser.add_argument("--repeat-mode", type=str, default="tile")
parser.add_argument("--shuffle-seed", type=int, default=0)
parser.add_argument("--max-inflight-requests", type=int, default=5)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--json-output", action="store_true")

args, _ = parser.parse_known_args()

# Distinguish phase based on port or environment
baseline_port = os.environ.get("ENGINE_BASELINE_PORT", "9000")
is_baseline = (str(args.port) == str(baseline_port)) or (args.port == "9000")

phase_name = "baseline" if is_baseline else "lmcache"
ttft_val = {baseline_ttft} if is_baseline else {lmcache_ttft}
round_time_val = {baseline_round_time} if is_baseline else {lmcache_round_time}

# Write distinguishable CSVs into the current working directory
with open("warmup_round.csv", "w", encoding="utf-8") as f:
    f.write("prompt_id,request_start,ttft,request_end,successful,is_miss\\n")
    for i in range(args.num_documents):
        f.write(f"{{i}},0.0,{{ttft_val + 0.1}},1.0,True,True\\n")

with open("query_round.csv", "w", encoding="utf-8") as f:
    f.write("prompt_id,request_start,ttft,request_end,successful,is_miss\\n")
    for i in range(args.num_documents * args.repeat_count):
        # Add slight variation to test percentile distributions
        sample_ttft = round(ttft_val + (i % 5) * 0.01, 4)
        f.write(f"{{i}},1.0,{{sample_ttft}},2.0,True,False\\n")

if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Mock long_doc_qa output for phase {{phase_name}}\\n")

if args.json_output:
    summary = {{
        "query_ttft_per_prompt": ttft_val,
        "query_round_time_per_prompt": round_time_val,
        "warmup_round_time_per_prompt": round_time_val * 1.5,
    }}
    print(json.dumps(summary))
"""
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def run_stubbed_long_doc_qa(
    tmp_path: str,
    extra_env: dict[str, str] | None = None,
    baseline_ttft: float = 1.30,
    lmcache_ttft: float = 0.39,
    baseline_round_time: float = 1.20,
    lmcache_round_time: float = 0.85,
) -> tuple[subprocess.CompletedProcess[str], str, dict[str, Any]]:
    """
    Execute long-doc-qa.sh in a stubbed CPU environment.

    Args:
        tmp_path: Root temporary directory for mock repository and results.
        extra_env: Optional environment variable overrides.
        baseline_ttft: Mock TTFT for baseline phase.
        lmcache_ttft: Mock TTFT for LMCache phase.
        baseline_round_time: Mock query round time for baseline.
        lmcache_round_time: Mock query round time for LMCache.

    Returns:
        Tuple containing:
            - CompletedProcess object from script execution
            - Path to results directory
            - Parsed diagnostic summary dictionary (empty if missing)
    """
    bash = get_bash_executable()
    if bash is None:
        pytest.skip("Bash executable is required to test long-doc-qa.sh harness")
        raise RuntimeError("Bash executable not found")

    script_path = get_workload_script_path()
    if not os.path.exists(script_path):
        pytest.fail(f"Workload script not found: {script_path}")
        raise FileNotFoundError(f"Workload script not found: {script_path}")

    bin_dir = os.path.join(tmp_path, "bin")
    results_dir = os.path.join(tmp_path, "results")
    stub_lmcache_dir = os.path.join(tmp_path, "stub_lmcache")
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Ensure python3 executable is shimmed if needed
    python_shim = os.path.join(bin_dir, "python3")
    py_exec = sys.executable.replace(os.sep, "/")
    with open(python_shim, "w", encoding="utf-8", newline="\n") as f:
        f.write(f'#!/usr/bin/env bash\nexec "{py_exec}" "$@"\n')
    os.chmod(python_shim, 0o755)

    # Create mock long_doc_qa benchmark script
    mock_benchmark_path = os.path.join(
        stub_lmcache_dir, "benchmarks", "long_doc_qa", "long_doc_qa.py"
    )
    create_mock_benchmark_script(
        mock_benchmark_path,
        baseline_ttft=baseline_ttft,
        lmcache_ttft=lmcache_ttft,
        baseline_round_time=baseline_round_time,
        lmcache_round_time=lmcache_round_time,
    )

    env = os.environ.copy()
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["PYTHON_EXEC"] = py_exec
    env["LMCACHE_DIR"] = stub_lmcache_dir.replace(os.sep, "/")
    env["RESULTS_DIR"] = results_dir.replace(os.sep, "/")
    env["BUILD_ID"] = "test_build_5403"
    env["ENGINE_PORT"] = "8000"
    env["ENGINE_BASELINE_PORT"] = "9000"
    env["MODEL"] = "Qwen/Qwen3-14B"
    env["NUM_DOCUMENTS"] = "5"
    env["DOCUMENT_LENGTH"] = "1000"
    env["OUTPUT_LEN"] = "50"
    env["REPEAT_COUNT"] = "2"
    env["REPEAT_MODE"] = "tile"
    env["SHUFFLE_SEED"] = "0"
    env["MAX_INFLIGHT_REQUESTS"] = "2"

    if extra_env:
        env.update(extra_env)

    script_posix = script_path.replace(os.sep, "/")
    proc = subprocess.run(
        [bash, script_posix],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    diagnostic_summary: dict[str, Any] = {}
    summary_path = os.path.join(results_dir, "long_doc_qa", "diagnostic_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            diagnostic_summary = json.load(f)

    return proc, results_dir, diagnostic_summary


def test_long_doc_qa_phase_isolation_and_file_preservation() -> None:
    """
    Verify that baseline and LMCache write to separate directories under
    RESULTS_DIR, preserving baseline raw CSVs, aggregate JSON, and text outputs
    without being overwritten by the subsequent LMCache invocation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc, results_dir, summary = run_stubbed_long_doc_qa(
            tmpdir,
            baseline_ttft=1.30,
            lmcache_ttft=0.39,  # 70% faster (passes >= 60% requirement)
        )

        assert proc.returncode == 0, (
            f"Script failed with output:\n{proc.stdout}\n{proc.stderr}"
        )
        qa_dir = os.path.join(results_dir, "long_doc_qa")
        baseline_dir = os.path.join(qa_dir, "baseline")
        lmcache_dir = os.path.join(qa_dir, "lmcache")

        # 1. Verify separate phase directories exist
        assert os.path.isdir(baseline_dir)
        assert os.path.isdir(lmcache_dir)

        # 2. Verify baseline files survive the second invocation
        baseline_warmup_csv = os.path.join(baseline_dir, "warmup_round.csv")
        baseline_query_csv = os.path.join(baseline_dir, "query_round.csv")
        baseline_json = os.path.join(baseline_dir, "baseline_result.json")
        baseline_out = os.path.join(baseline_dir, "baseline_output.txt")

        assert os.path.isfile(baseline_warmup_csv)
        assert os.path.isfile(baseline_query_csv)
        assert os.path.isfile(baseline_json)
        assert os.path.isfile(baseline_out)

        # 3. Verify LMCache files exist in its dedicated directory
        lmcache_warmup_csv = os.path.join(lmcache_dir, "warmup_round.csv")
        lmcache_query_csv = os.path.join(lmcache_dir, "query_round.csv")
        lmcache_json = os.path.join(lmcache_dir, "lmcache_result.json")
        lmcache_out = os.path.join(lmcache_dir, "lmcache_output.txt")

        assert os.path.isfile(lmcache_warmup_csv)
        assert os.path.isfile(lmcache_query_csv)
        assert os.path.isfile(lmcache_json)
        assert os.path.isfile(lmcache_out)

        # 4. Verify CSV contents are distinguishable and uncorrupted
        with open(baseline_query_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            baseline_rows = list(reader)
        with open(lmcache_query_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lmcache_rows = list(reader)

        assert len(baseline_rows) > 0
        assert len(lmcache_rows) > 0
        # Baseline TTFT starts at ~1.30, LMCache TTFT starts at ~0.39
        assert float(baseline_rows[0]["ttft"]) == pytest.approx(1.30, abs=0.05)
        assert float(lmcache_rows[0]["ttft"]) == pytest.approx(0.39, abs=0.05)

        # 5. Verify backwards-compatible top-level copies exist
        assert os.path.isfile(os.path.join(qa_dir, "baseline_result.json"))
        assert os.path.isfile(os.path.join(qa_dir, "lmcache_result.json"))

        # 6. Verify diagnostic summary exists and records PASS
        assert summary.get("verdict") == "PASS"
        assert "=== Long Doc QA Diagnostic Summary ===" in proc.stdout


def test_long_doc_qa_threshold_failure_preserves_both_phases() -> None:
    """
    Verify that when performance fails the TTFT requirement (e.g. only 20%
    faster instead of 60%), the script exits with non-zero exit status, but
    both phases' CSVs, JSONs, and diagnostic summaries remain fully intact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc, results_dir, summary = run_stubbed_long_doc_qa(
            tmpdir,
            baseline_ttft=1.30,
            lmcache_ttft=1.04,  # Only 20% faster, fails the 60% requirement
        )

        # Must fail with exit code 1
        assert proc.returncode != 0
        qa_dir = os.path.join(results_dir, "long_doc_qa")
        baseline_dir = os.path.join(qa_dir, "baseline")
        lmcache_dir = os.path.join(qa_dir, "lmcache")

        # Both phase directories and files must remain inspectable on failure
        assert os.path.isdir(baseline_dir)
        assert os.path.isdir(lmcache_dir)

        assert os.path.isfile(os.path.join(baseline_dir, "query_round.csv"))
        assert os.path.isfile(os.path.join(baseline_dir, "baseline_result.json"))
        assert os.path.isfile(os.path.join(lmcache_dir, "query_round.csv"))
        assert os.path.isfile(os.path.join(lmcache_dir, "lmcache_result.json"))

        # Diagnostic summary must record FAIL and capture failure comparison
        assert summary.get("verdict") == "FAIL"
        comp = summary.get("comparison", {}).get("query_ttft_per_prompt", {})
        assert comp.get("passed") is False
        assert comp.get("baseline") == pytest.approx(1.30, abs=0.01)
        assert comp.get("lmcache") == pytest.approx(1.04, abs=0.01)
        assert "=== Long Doc QA Diagnostic Summary ===" in proc.stdout


def test_long_doc_qa_diagnostic_summary_distributions() -> None:
    """
    Verify that diagnostic_summary.json identifies workload settings,
    sample and success counts, and accurate timing percentiles/distributions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc, _, summary = run_stubbed_long_doc_qa(
            tmpdir,
            baseline_ttft=1.30,
            lmcache_ttft=0.39,
        )

        assert proc.returncode == 0
        settings = summary.get("workload_settings", {})
        assert settings.get("model") == "Qwen/Qwen3-14B"
        assert settings.get("document_length") == 1000
        assert settings.get("num_documents") == 5
        assert settings.get("output_len") == 50
        assert settings.get("repeat_count") == 2
        assert settings.get("repeat_mode") == "tile"
        assert settings.get("shuffle_seed") == 0
        assert settings.get("max_inflight_requests") == 2
        assert settings.get("max_ttft_slowdown_pct") == -60.0
        assert settings.get("max_round_time_slowdown_pct") == -15.0

        phases = summary.get("phases", {})
        assert "baseline" in phases
        assert "lmcache" in phases

        for p_key in ["baseline", "lmcache"]:
            p = phases[p_key]
            q_stats = p.get("query_round", {})
            assert q_stats.get("total_samples") == 10  # 5 docs * 2 repeats
            assert q_stats.get("successful_samples") == 10
            ttft_dist = q_stats.get("ttft_stats", {})
            assert "p50" in ttft_dist
            assert "p95" in ttft_dist
            assert "p99" in ttft_dist
            assert "min" in ttft_dist
            assert "max" in ttft_dist
            assert "mean" in ttft_dist
            assert "std" in ttft_dist
            assert ttft_dist["min"] <= ttft_dist["p50"] <= ttft_dist["max"]


def test_long_doc_qa_attempt_archiving() -> None:
    """
    Verify that when BUILDKITE_RETRY_COUNT is present, attempt-specific copies
    of the diagnostic summary and phase files are archived.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        attempt_num = "3"
        proc, results_dir, summary = run_stubbed_long_doc_qa(
            tmpdir,
            extra_env={"BUILDKITE_RETRY_COUNT": attempt_num},
            baseline_ttft=1.30,
            lmcache_ttft=0.39,
        )

        assert proc.returncode == 0
        assert summary.get("attempt") == int(attempt_num)

        attempt_dir = os.path.join(
            results_dir, "long_doc_qa", f"attempt_{attempt_num}"
        )
        assert os.path.isdir(attempt_dir)
        archived_summary = os.path.join(attempt_dir, "diagnostic_summary.json")
        assert os.path.isfile(archived_summary)
        assert os.path.isfile(os.path.join(attempt_dir, "baseline", "query_round.csv"))
        assert os.path.isfile(os.path.join(attempt_dir, "lmcache", "query_round.csv"))


if __name__ == "__main__":
    print(
        "Running test_long_doc_qa_phase_isolation_and_file_preservation...",
        flush=True,
    )
    test_long_doc_qa_phase_isolation_and_file_preservation()
    print(
        "Running test_long_doc_qa_threshold_failure_preserves_both_phases...",
        flush=True,
    )
    test_long_doc_qa_threshold_failure_preserves_both_phases()
    print("Running test_long_doc_qa_diagnostic_summary_distributions...", flush=True)
    test_long_doc_qa_diagnostic_summary_distributions()
    print("Running test_long_doc_qa_attempt_archiving...", flush=True)
    test_long_doc_qa_attempt_archiving()
    print("All long_doc_qa regression tests passed successfully!", flush=True)
