# SPDX-License-Identifier: Apache-2.0
"""
Prove the benchmark suite's results are deterministic: run a
representative slice of it twice, in two independent, freshly started
Python processes, and assert the results are bit-for-bit identical.

This is the concrete answer to "do the results repeat themselves" --
not run twice in the same interpreter (which wouldn't catch e.g.
accidental reliance on PYTHONHASHSEED-randomized set/dict iteration
order, since Python re-randomizes that seed per process by default),
but two genuinely separate processes, exactly like two different
people (or two different CI runs, or a person on run N and the same
person on run N+1) invoking this suite independently.

Deliberately self-contained and fast (no ShareGPT corpus, no network):
covers all seed-capable synthetic workloads and every policy family
(including the admission-control double-count fix and the
CostAwareEvictionPolicy deterministic-clock injection -- see
docs/design/v1/storage_backend/cache_policy/admission-control-policy.md
and cost-aware-policy-eval.md's errata notices for why those two
specifically needed this kind of check). The real-ShareGPT tier is
just as deterministic (same seeding discipline -- see
real_dataset_eval.py's module docstring) but isn't included here since
it requires the corpus to be prepared separately first; see README.md.

Usage::

    python benchmarks/cache_policy/verify_reproducibility.py

Exit code 0 if every result matched across both runs; 1 otherwise, with
a diff of the first mismatch printed to stderr. Only the ``--worker``
subprocess needs ``lmcache`` importable (via ``PYTHONPATH``, set
automatically for the subprocess below) -- the orchestrating process
run by a user has no such requirement, so this script works as a
standalone entry point (e.g. a Docker ``CMD``) with no setup beyond a
plain Python interpreter.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIB = 2**20

_CACHE_SIZES_MIB = [10.0, 25.0]
_POLICIES = [
    "LRU", "LFU", "FIFO", "MRU", "COST_AWARE",
    "ADMISSION_LRU", "ADMISSION_COST_AWARE",
    "WINDOWED_ADMISSION_LRU", "WINDOWED_ADMISSION_COST_AWARE",
]
_WORKLOAD_NAMES = [
    "repetitive_short", "novel_long", "mixed_zipfian", "multi_round_chat",
]

# Excluded deliberately: these measure the host machine's performance
# while running the simulator, not the simulated policy's behavior --
# they are expected to differ between runs (and between machines) even
# when every result that matters (hit rate, eviction/rejection counts,
# modeled latency, sketch diagnostics) is identical.
_NON_DETERMINISTIC_FIELDS = {
    "wall_clock_seconds", "requests_per_second", "tokens_per_second",
    "rss_delta_bytes",
}


def run_once() -> list[dict[str, Any]]:
    """
    Run the full small matrix once and return every result as a plain
    dict. Imports lmcache lazily so that only this function (run in
    the ``--worker`` subprocess) needs it importable -- see module
    docstring.
    """
    # First Party
    from lmcache.tools.cache_policy_bench.cost_model import (  # noqa: PLC0415
        CostModel,
        CostModelConfig,
    )
    from lmcache.tools.cache_policy_bench.runner import (  # noqa: PLC0415
        DEFAULT_KV_BYTES_PER_CHUNK,
        run_workload,
    )
    from lmcache.tools.cache_policy_bench.workloads import (  # noqa: PLC0415
        mixed_zipfian,
        multi_round_chat,
        novel_long,
        repetitive_short,
    )

    # Deliberately small (fast: a few seconds total) but touches every
    # policy family and every seed-capable generator, plus the one
    # deterministic (unseeded) generator for completeness.
    workloads = {
        "repetitive_short": repetitive_short(400, vocab_size=30, seed=7),
        "novel_long": novel_long(100, seed=7),
        "mixed_zipfian": mixed_zipfian(600, unique_prefixes=60, seed=7),
        "multi_round_chat": multi_round_chat(15, rounds_per_session=8),
    }

    cost_model = CostModel(CostModelConfig())
    results: list[dict[str, Any]] = []
    for workload_name, requests in workloads.items():
        for cache_mib in _CACHE_SIZES_MIB:
            cache_bytes = int(cache_mib * _MIB)
            for policy_name in _POLICIES:
                result = run_workload(
                    policy_name,
                    requests,
                    cache_bytes,
                    DEFAULT_KV_BYTES_PER_CHUNK,
                    cost_model,
                    workload_name=workload_name,
                )
                row = result.to_dict()
                for field in _NON_DETERMINISTIC_FIELDS:
                    row.pop(field, None)
                results.append(row)
    return results


def _run_worker_subprocess() -> list[dict[str, Any]]:
    """Spawn a fresh ``python ... --worker`` process and parse its stdout."""
    env = dict(os.environ)
    # Ensure the worker can `import lmcache` regardless of whether the
    # caller happened to have PYTHONPATH set -- this script must be
    # runnable as a standalone entry point (e.g. a Docker CMD).
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    )
    proc = subprocess.run(
        [sys.executable, str(Path(__file__)), "--worker"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    return json.loads(proc.stdout)


def _first_mismatch(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    if len(rows_a) != len(rows_b):
        return (-1, {"len": len(rows_a)}, {"len": len(rows_b)})
    for i, (a, b) in enumerate(zip(rows_a, rows_b, strict=True)):
        if a != b:
            return (i, a, b)
    return None


def main() -> int:
    n = len(_WORKLOAD_NAMES) * len(_CACHE_SIZES_MIB) * len(_POLICIES)
    print(
        "Running the reproducibility-check matrix twice, in two "
        "independent processes..."
    )
    print(
        f"({len(_WORKLOAD_NAMES)} workloads x {len(_CACHE_SIZES_MIB)} cache "
        f"sizes x {len(_POLICIES)} policies = {n} results each run)"
    )

    rows_a = _run_worker_subprocess()
    rows_b = _run_worker_subprocess()

    mismatch = _first_mismatch(rows_a, rows_b)
    if mismatch is None:
        print(
            f"PASS: all {len(rows_a)} results were bit-for-bit identical across "
            "two independent process runs."
        )
        return 0

    idx, a, b = mismatch
    print("FAIL: results differ between the two runs.", file=sys.stderr)
    print(f"First mismatch at result index {idx}:", file=sys.stderr)
    print(f"  run 1: {a}", file=sys.stderr)
    print(f"  run 2: {b}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(run_once()))
    else:
        sys.exit(main())
