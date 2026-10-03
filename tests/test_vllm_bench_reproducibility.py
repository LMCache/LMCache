# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for vllm-bench.sh reproducibility improvements.

These tests use stub benchmark commands to verify:
- Default seed is repeatable across runs
- Both baseline and LMCache sides receive the same seed
- RANDOM_SEED override is honored
- Comparison manifest is written with correct fields
- Pass/fail verdicts retain actual throughput numbers
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_BENCH_SCRIPT = (
    REPO_ROOT
    / ".buildkite"
    / "k3_tests"
    / "multiprocess"
    / "scripts"
    / "workloads"
    / "vllm"
    / "vllm-bench.sh"
)


def _create_stub_vllm(
    tmp_path: Path,
    throughput_baseline: float = 100.0,
    throughput_lmcache: float = 95.0,
) -> Path:
    """Create a stub vllm command that writes a fake benchmark result.

    The stub uses the --port argument to determine which throughput to write:
    port 19000 (baseline) gets throughput_baseline, port 18000 (LMCache) gets
    throughput_lmcache.
    """
    stub_dir = tmp_path / "stub_bin"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "vllm"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "# Stub vllm command for testing\n"
        "if [[ \"$1\" == \"bench\" && \"$2\" == \"serve\" ]]; then\n"
        "    result_dir=''\n"
        "    result_filename=''\n"
        "    port=''\n"
        "    while [[ $# -gt 0 ]]; do\n"
        "        case \"$1\" in\n"
        "            --result-dir) result_dir=\"$2\"; shift 2 ;;\n"
        "            --result-filename) result_filename=\"$2\"; shift 2 ;;\n"
        "            --port) port=\"$2\"; shift 2 ;;\n"
        "            *) shift ;;\n"
        "        esac\n"
        "    done\n"
        "    mkdir -p \"$result_dir\"\n"
        "    throughput=" + str(throughput_baseline) + "\n"
        "    if [[ \"$port\" == \"18000\" ]]; then\n"
        "        throughput=" + str(throughput_lmcache) + "\n"
        "    fi\n"
        "    echo '{\"total_input_tokens\": 500000, \"completed\": 50, "
        "\"total_token_throughput\": '\"$throughput\"'}' "
        "> \"$result_dir/$result_filename\"\n"
        "    exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub_dir


def _create_stub_python(tmp_path: Path) -> Path:
    """Create a stub python3 that reports fake versions."""
    stub_dir = tmp_path / "stub_python"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python3"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "    if [[ \"$2\" == *\"import vllm\"* ]]; then\n"
        "        echo '0.30.1rc1.dev244+g0da126676'\n"
        "        exit 0\n"
        "    elif [[ \"$2\" == *\"import torch\"* ]]; then\n"
        "        echo '2.13.0+cu130'\n"
        "        exit 0\n"
        "    fi\n"
        "fi\n"
        "# Delegate everything else to real python3\n"
        "exec /usr/bin/python3 \"$@\"\n"
    )
    stub.chmod(0o755)
    return stub_dir


def _create_stub_curl(tmp_path: Path) -> Path:
    """Create a stub curl that returns success for warmup requests."""
    stub_dir = tmp_path / "stub_curl"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "curl"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "# Stub curl for testing - always returns success\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub_dir


def _run_vllm_bench(
    tmp_path: Path,
    env_overrides: dict[str, str] | None = None,
    throughput_baseline: float = 100.0,
    throughput_lmcache: float = 95.0,
) -> subprocess.CompletedProcess:
    """Run vllm-bench.sh with stub commands and return the result."""
    stub_bin = _create_stub_vllm(tmp_path, throughput_baseline, throughput_lmcache)
    stub_python = _create_stub_python(tmp_path)
    stub_curl = _create_stub_curl(tmp_path)

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}:{stub_python}:{stub_curl}:{env['PATH']}"
    env["MODEL"] = "test-model"
    env["NUM_PROMPTS"] = "50"
    env["RANDOM_INPUT_LEN"] = "10000"
    env["RANDOM_OUTPUT_LEN"] = "1"
    env["BUILD_ID"] = f"test_{os.getpid()}"
    env["RESULTS_DIR"] = str(tmp_path / "results")
    env["LAUNCH_BASELINE"] = "true"
    env["LMCACHE_MP_LAZY_OFFLOAD"] = "true"
    env["VLLM_PORT"] = "18000"
    env["VLLM_BASELINE_PORT"] = "19000"

    if env_overrides:
        env.update(env_overrides)

    # Create a fake git repo so git rev-parse works
    git_dir = tmp_path / "fake_repo"
    git_dir.mkdir()
    subprocess.run(
        ["git", "init", "--bare"],
        cwd=git_dir,
        capture_output=True,
        check=True,
    )

    result = subprocess.run(
        ["bash", str(VLLM_BENCH_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    return result


class TestDefaultSeedRepeatable:
    """Test that the default seed is stable across multiple runs."""

    def test_default_seed_is_stable(self, tmp_path: Path) -> None:
        """Two runs without RANDOM_SEED override should use the same seed."""
        result1 = _run_vllm_bench(tmp_path / "run1")
        result2 = _run_vllm_bench(tmp_path / "run2")

        assert result1.returncode == 0, f"Run 1 failed: {result1.stderr}"
        assert result2.returncode == 0, f"Run 2 failed: {result2.stderr}"

        manifest1_path = (
            tmp_path / "run1" / "results" / "vllm_bench" / "comparison_manifest.json"
        )
        manifest2_path = (
            tmp_path / "run2" / "results" / "vllm_bench" / "comparison_manifest.json"
        )

        assert manifest1_path.exists(), "Manifest not found in run 1"
        assert manifest2_path.exists(), "Manifest not found in run 2"

        manifest1 = json.loads(manifest1_path.read_text())
        manifest2 = json.loads(manifest2_path.read_text())

        assert manifest1["seed"] == manifest2["seed"], (
            f"Default seeds differ: {manifest1['seed']} vs {manifest2['seed']}"
        )
        assert manifest1["seed"] == 42, f"Expected default seed 42, got {manifest1['seed']}"


class TestSeedOverride:
    """Test that RANDOM_SEED override is honored."""

    def test_override_seed_is_used(self, tmp_path: Path) -> None:
        """When RANDOM_SEED is set, it should be used in the manifest."""
        result = _run_vllm_bench(tmp_path, env_overrides={"RANDOM_SEED": "12345"})

        assert result.returncode == 0, f"Run failed: {result.stderr}"

        manifest_path = (
            tmp_path / "results" / "vllm_bench" / "comparison_manifest.json"
        )
        assert manifest_path.exists(), "Manifest not found"

        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == 12345, (
            f"Expected seed 12345, got {manifest['seed']}"
        )


class TestManifestContents:
    """Test that the comparison manifest contains all required fields."""

    def test_manifest_has_required_fields(self, tmp_path: Path) -> None:
        """Manifest should contain seed, model, versions, throughputs, and verdict."""
        result = _run_vllm_bench(tmp_path)

        assert result.returncode == 0, f"Run failed: {result.stderr}"

        manifest_path = (
            tmp_path / "results" / "vllm_bench" / "comparison_manifest.json"
        )
        assert manifest_path.exists(), "Manifest not found"

        manifest = json.loads(manifest_path.read_text())

        required_fields = [
            "seed",
            "model",
            "num_prompts",
            "random_input_len",
            "random_output_len",
            "git_sha",
            "vllm_version",
            "torch_version",
            "baseline_throughput",
            "lmcache_throughput",
            "verdict",
            "max_slowdown_percent",
        ]
        for field in required_fields:
            assert field in manifest, f"Missing field: {field}"

        assert manifest["seed"] == 42
        assert manifest["model"] == "test-model"
        assert manifest["num_prompts"] == 50
        assert manifest["random_input_len"] == 10000
        assert manifest["random_output_len"] == 1
        assert manifest["vllm_version"] == "0.30.1rc1.dev244+g0da126676"
        assert manifest["torch_version"] == "2.13.0+cu130"
        assert manifest["max_slowdown_percent"] == 5.0


class TestPassFailVerdicts:
    """Test that pass/fail verdicts retain actual throughput numbers."""

    def test_pass_verdict(self, tmp_path: Path) -> None:
        """When LMCache is within 5% of baseline, verdict should be PASS."""
        result = _run_vllm_bench(
            tmp_path,
            throughput_baseline=100.0,
            throughput_lmcache=96.0,  # 4% slower, within 5% limit
        )

        assert result.returncode == 0, f"Run failed: {result.stderr}"

        manifest_path = (
            tmp_path / "results" / "vllm_bench" / "comparison_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())

        assert manifest["verdict"] == "PASS"
        assert manifest["baseline_throughput"] == 100.0
        assert manifest["lmcache_throughput"] == 96.0

    def test_fail_verdict(self, tmp_path: Path) -> None:
        """When LMCache is more than 5% slower, verdict should be FAIL."""
        result = _run_vllm_bench(
            tmp_path,
            throughput_baseline=100.0,
            throughput_lmcache=90.0,  # 10% slower, exceeds 5% limit
        )

        assert result.returncode != 0, "Expected non-zero exit code for FAIL verdict"

        manifest_path = (
            tmp_path / "results" / "vllm_bench" / "comparison_manifest.json"
        )
        assert manifest_path.exists(), "Manifest should be written even on failure"

        manifest = json.loads(manifest_path.read_text())
        assert manifest["verdict"] == "FAIL"
        assert manifest["baseline_throughput"] == 100.0
        assert manifest["lmcache_throughput"] == 90.0


class TestBothSidesSameSeed:
    """Test that both baseline and LMCache benchmarks receive the same seed."""

    def test_both_sides_use_same_seed(self, tmp_path: Path) -> None:
        """Both baseline.json and lmcache.json should be generated with the same seed."""
        result = _run_vllm_bench(tmp_path)

        assert result.returncode == 0, f"Run failed: {result.stderr}"

        # Check that the script logged the same seed for both runs
        output = result.stdout
        assert "Seed: 42" in output, "Seed 42 not found in output"

        # Count occurrences of "Seed: 42" - should appear at least twice
        # (once for baseline, once for LMCache)
        seed_count = output.count("Seed: 42")
        assert seed_count >= 2, (
            f"Expected at least 2 occurrences of 'Seed: 42', found {seed_count}"
        )
