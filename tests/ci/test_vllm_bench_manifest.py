# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression coverage for reproducible vllm_bench inputs.

Run without loading the torch-dependent tests/conftest.py:

    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_vllm_bench_manifest.py
"""

# Standard
from pathlib import Path
import fnmatch
import json
import os
import subprocess
import sys

# Third Party
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = (
    REPO_ROOT / ".buildkite/k3_tests/multiprocess/scripts/workloads/vllm/vllm-bench.sh"
)
STUB_FIXTURE = Path(__file__).parent / "fixtures/vllm_bench_stub.py"


def _run_workload(tmp_path: Path, extra_env: dict[str, str] | None = None):
    """Run the workload against stub ``vllm``/``curl`` executables."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True)

    vllm_stub = stub_dir / "vllm"
    vllm_stub.write_text(
        "#!/usr/bin/env bash\n"
        f"shift 2\n"
        f'exec "{sys.executable}" "{STUB_FIXTURE}" "$@"\n',
        encoding="utf-8",
    )
    vllm_stub.chmod(0o755)

    # Warmup requests have no server to talk to; the stub succeeds silently.
    curl_stub = stub_dir / "curl"
    curl_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    curl_stub.chmod(0o755)

    results_dir = tmp_path / "results"
    calls_log = tmp_path / "stub_calls.log"

    env = os.environ.copy()
    env.pop("RANDOM_SEED", None)
    env.update(
        {
            "PATH": f"{stub_dir}:{env['PATH']}",
            "RESULTS_DIR": str(results_dir),
            "BUILD_ID": "stub",
            "LAUNCH_BASELINE": "true",
            # The cache-hit replay talks to a live server; out of scope here.
            "LMCACHE_MP_LAZY_OFFLOAD": "true",
            "STUB_CALLS_LOG": str(calls_log),
        }
    )
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(WORKLOAD)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest_path = results_dir / "vllm_bench" / "manifest.json"
    return proc, manifest_path, calls_log


def _read_manifest(path: Path) -> dict:
    assert path.is_file(), f"manifest missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_calls(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), "stub benchmark never ran"
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        side, _, rest = line.partition(" ")
        fields = dict(item.split("=", 1) for item in rest.split())
        calls.append({"side": side, **fields})
    return calls


def test_default_seed_is_stable_across_attempts(tmp_path):
    """Two attempts of the same build must use identical inputs."""
    first, first_manifest_path, _ = _run_workload(tmp_path / "attempt1")
    second, second_manifest_path, _ = _run_workload(tmp_path / "attempt2")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    first_manifest = _read_manifest(first_manifest_path)
    second_manifest = _read_manifest(second_manifest_path)

    assert first_manifest["inputs"]["seed"] == 0
    assert first_manifest["inputs"]["seed_source"] == "default"
    assert first_manifest["inputs"] == second_manifest["inputs"]
    assert first_manifest["results"] == second_manifest["results"]


def test_both_sides_receive_the_same_seed(tmp_path):
    """The baseline and LMCache runs must be driven by one seed."""
    proc, _, calls_log = _run_workload(tmp_path)

    assert proc.returncode == 0, proc.stderr

    calls = _read_calls(calls_log)
    assert {call["side"] for call in calls} == {"baseline", "lmcache"}
    assert {call["seed"] for call in calls} == {"0"}


def test_explicit_seed_override_is_honored(tmp_path):
    """A seed sweep must still be able to pin its own seed."""
    proc, manifest_path, calls_log = _run_workload(tmp_path, {"RANDOM_SEED": "777"})

    assert proc.returncode == 0, proc.stderr

    manifest = _read_manifest(manifest_path)
    assert manifest["inputs"]["seed"] == 777
    assert manifest["inputs"]["seed_source"] == "override"
    assert {call["seed"] for call in _read_calls(calls_log)} == {"777"}


def test_failing_verdict_keeps_the_measured_numbers(tmp_path):
    """A failed throughput check must not hide the numbers behind it."""
    proc, manifest_path, _ = _run_workload(
        tmp_path,
        {"STUB_BASELINE_THROUGHPUT": "100.0", "STUB_LMCACHE_THROUGHPUT": "40.0"},
    )

    assert proc.returncode != 0

    manifest = _read_manifest(manifest_path)
    assert manifest["verdict"] == "FAIL"
    assert manifest["comparison"]["slowdown_percent"] == pytest.approx(60.0)
    assert manifest["results"]["baseline"]["total_token_throughput"] == 100.0
    assert manifest["results"]["lmcache"]["total_token_throughput"] == 40.0


def test_manifest_is_reachable_from_job_artifacts():
    """cleanup.sh stages RESULTS_DIR, so the step must upload it."""
    pipeline = yaml.safe_load(
        (REPO_ROOT / ".buildkite/k3_tests/multiprocess/pipeline.yml").read_text(
            encoding="utf-8"
        )
    )

    bench_steps = [
        step
        for step in _iter_steps(pipeline["steps"])
        if "vllm_bench" in step.get("label", "")
    ]
    assert bench_steps, "vllm_bench step not found in pipeline.yml"

    artifact_globs = bench_steps[0]["artifact_paths"]
    staged_manifest = "ci_results_stub/vllm_bench/manifest.json"
    assert any(_glob_matches(glob, staged_manifest) for glob in artifact_globs), (
        f"no artifact path covers {staged_manifest}: {artifact_globs}"
    )


def _iter_steps(steps):
    """Flatten Buildkite groups into their leaf steps."""
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "group" in step:
            yield from _iter_steps(step.get("steps", []))
        else:
            yield step


def _glob_matches(pattern: str, path: str) -> bool:
    """Match Buildkite-style ``**`` globs against a relative artifact path."""
    pattern_parts = pattern.split("/")
    path_parts = path.split("/")

    def match(pi: int, si: int) -> bool:
        if pi == len(pattern_parts):
            return si == len(path_parts)
        if pattern_parts[pi] == "**":
            return any(match(pi + 1, skip) for skip in range(si, len(path_parts) + 1))
        if si == len(path_parts):
            return False
        if not _part_matches(pattern_parts[pi], path_parts[si]):
            return False
        return match(pi + 1, si + 1)

    return match(0, 0)


def _part_matches(pattern: str, part: str) -> bool:
    """Match one Buildkite artifact glob segment against a path segment."""
    return fnmatch.fnmatchcase(part, pattern)
