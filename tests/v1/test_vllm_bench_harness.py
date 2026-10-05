# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import shutil
import subprocess
import tempfile

# Third Party
import pytest


def get_bash_executable() -> Optional[str]:
    """
    Find an available bash executable on the current system.

    Returns:
        Optional[str]: Path to bash executable, or None if not found.
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


def get_bench_script_path() -> str:
    """
    Locate the vllm-bench.sh workload script relative to the repo root.

    Returns:
        str: Absolute path to vllm-bench.sh.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(
        repo_root,
        ".buildkite",
        "k3_tests",
        "multiprocess",
        "scripts",
        "workloads",
        "vllm",
        "vllm-bench.sh",
    )
    return script_path


def run_stubbed_benchmark(
    tmp_path: str,
    extra_env: Optional[Dict[str, str]] = None,
    mock_baseline_tp: float = 100.0,
    mock_lmcache_tp: float = 98.0,
) -> Tuple[subprocess.CompletedProcess, List[str], Dict[str, Any]]:
    """
    Execute vllm-bench.sh in a stubbed CPU environment.

    Args:
        tmp_path: Root temporary directory for mock binaries and results.
        extra_env: Optional environment variable overrides.
        mock_baseline_tp: Throughput value returned for baseline server.
        mock_lmcache_tp: Throughput value returned for LMCache server.

    Returns:
        Tuple containing:
            - CompletedProcess object from script execution
            - List of raw invocation arguments captured by mock vllm
            - Parsed comparison manifest dictionary (empty if unparseable)
    """
    bash = get_bash_executable()
    if not bash:
        pytest.skip("Bash executable is required to test vllm-bench.sh harness")

    script_path = get_bench_script_path()
    if not os.path.exists(script_path):
        pytest.fail(f"Benchmark script not found: {script_path}")

    bin_dir = os.path.join(tmp_path, "bin")
    results_dir = os.path.join(tmp_path, "results")
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    mock_log = os.path.join(tmp_path, "vllm_invocations.txt")
    log_posix = mock_log.replace(os.sep, "/")

    # Create mock vllm executable
    vllm_stub = os.path.join(bin_dir, "vllm")
    vllm_content = f"""#!/usr/bin/env bash
echo "$@" >> "{log_posix}"

result_dir=""
result_filename=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-dir)
            result_dir="$2"
            shift 2
            ;;
        --result-filename)
            result_filename="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -n "$result_dir" && -n "$result_filename" ]]; then
    mkdir -p "$result_dir"
    tp="100.0"
    if [[ "$result_filename" == *"baseline"* ]]; then
        tp="${{MOCK_BASELINE_TP:-100.0}}"
    elif [[ "$result_filename" == *"lmcache"* ]]; then
        tp="${{MOCK_LMCACHE_TP:-98.0}}"
    fi

    cat <<EOF > "$result_dir/$result_filename"
{{
  "total_input_tokens": 500000,
  "completed": 50,
  "total_token_throughput": $tp
}}
EOF
fi
exit 0
"""
    with open(vllm_stub, "w", encoding="utf-8", newline="\n") as f:
        f.write(vllm_content)
    os.chmod(vllm_stub, 0o755)

    # Create mock curl executable
    curl_stub = os.path.join(bin_dir, "curl")
    curl_content = """#!/usr/bin/env bash
exit 0
"""
    with open(curl_stub, "w", encoding="utf-8", newline="\n") as f:
        f.write(curl_content)
    os.chmod(curl_stub, 0o755)

    # Set up environment
    env = os.environ.copy()
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["RESULTS_DIR"] = results_dir.replace(os.sep, "/")
    env["NUM_WARMUP"] = "0"
    env["LMCACHE_MP_LAZY_OFFLOAD"] = "true"
    env["BUILD_ID"] = "test_build"
    env["MOCK_BASELINE_TP"] = str(mock_baseline_tp)
    env["MOCK_LMCACHE_TP"] = str(mock_lmcache_tp)

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

    invocations: List[str] = []
    if os.path.exists(mock_log):
        with open(mock_log, "r", encoding="utf-8") as f:
            invocations = [line.strip() for line in f if line.strip()]

    manifest: Dict[str, Any] = {}
    manifest_path = os.path.join(results_dir, "vllm_bench", "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return proc, invocations, manifest


def test_vllm_bench_default_seed_repeatable_and_symmetric() -> None:
    """
    Verify that vllm-bench.sh uses a fixed default seed (42) repeatably
    and passes the identical seed to both baseline and LMCache benchmarks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run 1
        run1_dir = os.path.join(tmpdir, "run1")
        proc1, invocations1, manifest1 = run_stubbed_benchmark(run1_dir)

        assert proc1.returncode == 0
        assert len(invocations1) == 2
        # Baseline invocation has --seed 42
        assert "--seed 42" in invocations1[0]
        # LMCache invocation has --seed 42
        assert "--seed 42" in invocations1[1]
        assert manifest1.get("effective_seed") == 42
        assert manifest1.get("verdict") == "PASS"

        # Run 2 (separate invocation without RANDOM_SEED)
        run2_dir = os.path.join(tmpdir, "run2")
        proc2, invocations2, manifest2 = run_stubbed_benchmark(run2_dir)

        assert proc2.returncode == 0
        assert len(invocations2) == 2
        assert "--seed 42" in invocations2[0]
        assert "--seed 42" in invocations2[1]
        assert manifest2.get("effective_seed") == 42
        # Ensure seeds across both separate runs match exactly
        assert manifest1["effective_seed"] == manifest2["effective_seed"]


def test_vllm_bench_seed_override_honored() -> None:
    """
    Verify that an explicit RANDOM_SEED environment variable override
    is honored and delivered to both benchmark invocations.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_seed = "98765"
        proc, invocations, manifest = run_stubbed_benchmark(
            tmpdir,
            extra_env={"RANDOM_SEED": custom_seed},
        )

        assert proc.returncode == 0
        assert len(invocations) == 2
        assert f"--seed {custom_seed}" in invocations[0]
        assert f"--seed {custom_seed}" in invocations[1]
        assert manifest.get("effective_seed") == int(custom_seed)


def test_vllm_bench_pass_summary_retains_numbers() -> None:
    """
    Verify that when throughput is within the allowed slowdown threshold,
    the script passes and records exact throughputs and metrics in the manifest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc, _, manifest = run_stubbed_benchmark(
            tmpdir,
            mock_baseline_tp=100.0,
            mock_lmcache_tp=98.0,
        )

        assert proc.returncode == 0
        assert manifest.get("verdict") == "PASS"
        assert manifest.get("baseline_throughput") == 100.0
        assert manifest.get("lmcache_throughput") == 98.0
        assert manifest.get("slowdown_percent") == pytest.approx(2.0, 0.01)
        assert manifest.get("max_slowdown_percent") == 5.0
        assert manifest.get("num_prompts") == 50
        assert manifest.get("random_input_len") == 10000
        assert manifest.get("random_output_len") == 1
        assert "timestamp" in manifest
        assert "checkout_sha" in manifest
        # Check manifest was logged to stdout
        assert "=== Benchmark Comparison Manifest ===" in proc.stdout


def test_vllm_bench_fail_summary_retains_numbers() -> None:
    """
    Verify that when throughput exceeds the 5% slowdown limit, the script
    exits with code 1 and retains exact numbers in the manifest on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc, _, manifest = run_stubbed_benchmark(
            tmpdir,
            mock_baseline_tp=100.0,
            mock_lmcache_tp=90.0,  # 10% slowdown, exceeds 5%
        )

        # Must fail with exit code 1
        assert proc.returncode != 0
        # Manifest must still exist and record the failure
        assert manifest.get("verdict") == "FAIL"
        assert manifest.get("baseline_throughput") == 100.0
        assert manifest.get("lmcache_throughput") == 90.0
        assert manifest.get("slowdown_percent") == pytest.approx(10.0, 0.01)
        assert manifest.get("max_slowdown_percent") == 5.0
        assert "=== Benchmark Comparison Manifest ===" in proc.stdout


def test_vllm_bench_attempt_preservation() -> None:
    """
    Verify that when BUILDKITE_RETRY_COUNT is set, the retry attempt is
    recorded in the manifest and archived to an attempt-specific directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        attempt_num = "2"
        proc, _, manifest = run_stubbed_benchmark(
            tmpdir,
            extra_env={"BUILDKITE_RETRY_COUNT": attempt_num},
        )

        assert proc.returncode == 0
        assert manifest.get("attempt") == int(attempt_num)

        attempt_dir = os.path.join(
            tmpdir, "results", "vllm_bench", f"attempt_{attempt_num}"
        )
        assert os.path.isdir(attempt_dir)

        archived_manifest = os.path.join(attempt_dir, "manifest.json")
        assert os.path.isfile(archived_manifest)
        with open(archived_manifest, "r", encoding="utf-8") as f:
            archived_data = json.load(f)
        assert archived_data.get("attempt") == int(attempt_num)
        assert os.path.isfile(os.path.join(attempt_dir, "baseline.json"))
        assert os.path.isfile(os.path.join(attempt_dir, "lmcache.json"))


if __name__ == "__main__":
    print(
        "Running test_vllm_bench_default_seed_repeatable_and_symmetric...", flush=True
    )
    test_vllm_bench_default_seed_repeatable_and_symmetric()
    print("Running test_vllm_bench_seed_override_honored...", flush=True)
    test_vllm_bench_seed_override_honored()
    print("Running test_vllm_bench_pass_summary_retains_numbers...", flush=True)
    test_vllm_bench_pass_summary_retains_numbers()
    print("Running test_vllm_bench_fail_summary_retains_numbers...", flush=True)
    test_vllm_bench_fail_summary_retains_numbers()
    print("Running test_vllm_bench_attempt_preservation...", flush=True)
    test_vllm_bench_attempt_preservation()
    print("All regression tests passed successfully!", flush=True)
