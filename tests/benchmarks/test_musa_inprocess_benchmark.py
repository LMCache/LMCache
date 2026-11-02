# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import importlib.util

# Locate the benchmark module relative to the repo root, so the test does not
# depend on ``benchmarks`` being an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_PATH = (
    _REPO_ROOT / "benchmarks" / "inprocess" / "musa" / "bench_inprocess_transfer.py"
)

_spec = importlib.util.spec_from_file_location("bench_inprocess_transfer", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BenchmarkResult = _mod.BenchmarkResult
compare_results = _mod.compare_results


def test_compare_results_requires_native_speedup() -> None:
    """A fast native result passes the Stage2 speedup gate."""
    torch_result = BenchmarkResult(name="torch", seconds_per_iter=1.0)
    native_result = BenchmarkResult(name="native", seconds_per_iter=0.7)

    passed, summary = compare_results(
        torch_result,
        native_result,
        min_speedup=1.2,
    )

    assert passed is True
    assert "speedup=1.429x" in summary


def test_compare_results_fails_when_native_is_slower() -> None:
    """A slower native result fails with the required speedup in the summary."""
    torch_result = BenchmarkResult(name="torch", seconds_per_iter=1.0)
    native_result = BenchmarkResult(name="native", seconds_per_iter=1.1)

    passed, summary = compare_results(
        torch_result,
        native_result,
        min_speedup=1.2,
    )

    assert passed is False
    assert "required>=1.200x" in summary
