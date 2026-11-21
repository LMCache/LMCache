# SPDX-License-Identifier: Apache-2.0
"""Cross-night regression check: has a metric shifted to a worse level?

No fixed thresholds: :func:`detect` takes the largest standardized
Mann-Whitney statistic over every split of a metric's series and gets its
p-value by permutation, so a noisy metric is harder to trigger. The per-metric
level is ``[regression].false_alarm_rate`` divided by the metric count; with
few runs nothing can fail.

A confirmed shift (worse or better) starts a new regime: later nights are
tested only against runs since the shift, so a regression fails once and then
becomes the baseline. History is one JSON line per valid run in
``<host.root>/prt-results/history.jsonl``; only runs with the same
:func:`config_hash` count.

Example :func:`check` entry for one metric::

    {"value": 31.0, "median": 24.6, "change": 0.26, "n": 14,
     "p_worse": 0.002, "p_better": 1.0, "regressed": True, "shift": "worse"}
"""

# Standard
from bisect import bisect_left
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any
import hashlib
import json
import math
import random

# First Party
from perf_regression_testing.runner.config import RunSpec

# Metrics where larger is better; all others are better when smaller.
HIGHER_IS_BETTER = frozenset({"decode_p10", "decode_p50"})


def _ranks(xs: list[float]) -> list[float]:
    """1-based ranks of ``xs``, ties sharing their average rank."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _max_z(r: list[float]) -> tuple[float, int]:
    """Return ``(z, s)``: the largest standardized Mann-Whitney U of a recent
    segment over the rest, and the index where that segment starts.

    ``r`` holds ranks, oldest first; a high rank is a bad value.
    """
    n = len(r)
    best, at, tail = -math.inf, n, 0.0
    for m in range(1, n):
        tail += r[n - m]
        k = n - m
        z = (tail - m * (m + 1) / 2 - k * m / 2) / math.sqrt(k * m * (n + 1) / 12)
        if z > best:
            best, at = z, k
    return best, at


@lru_cache(maxsize=None)
def _null(n: int, perms: int) -> list[float]:
    """Sorted :func:`_max_z` of ``perms`` seeded random orderings of ``n`` runs."""
    rng = random.Random(n)
    r = [float(i) for i in range(1, n + 1)]
    out = []
    for _ in range(perms):
        rng.shuffle(r)
        out.append(_max_z(r)[0])
    out.sort()
    return out


# RunSpec fields that change what a run measures; editing others keeps the
# baseline.
_CONFIG_FIELDS = (
    "gpu_count",
    "container",
    "model_path",
    "vllm_env",
    "vllm_args",
    "services",
    "vllm_scratch",
    "files",
    "devices",
    "cap_add",
    "aiperf_env",
    "aiperf_args",
    "trace",
    "window_s",
)


def config_hash(spec: RunSpec) -> str:
    """Hash the resolved settings of one run that affect its results.

    Args:
        spec: Resolved run settings.

    Returns:
        12 hex chars of the SHA-256 of those fields as canonical JSON.
    """
    d = asdict(spec)
    # Unset optional fields hash as absent, so adding one keeps old baselines.
    d["services"] = [
        {k: v for k, v in svc.items() if v is not None} for svc in d["services"]
    ]
    fields = {k: d[k] for k in _CONFIG_FIELDS}
    if spec.vllm_memory_gb is not None:
        fields["vllm_memory_gb"] = spec.vllm_memory_gb
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_history(path: Path) -> list[dict[str, Any]]:
    """Read every recorded run.

    Args:
        path: History file; a missing file means no runs yet.

    Returns:
        One dict per run, oldest first.
    """
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def detect(values: list[float], higher_is_better: bool, alpha: float) -> dict[str, Any]:
    """Test whether the latest values of a series shifted to another level.

    Args:
        values: One metric's values, oldest first; the last is tonight's.
        higher_is_better: True for throughput-like metrics.
        alpha: Per-test false alarm level, for each direction.

    Returns:
        ``{"n", "p_worse", "p_better", "direction", "shift_at"}``: the
        permutation p-values of the most significant split in each
        direction, ``direction`` ``"worse"``/``"better"`` when its p-value is
        at most ``alpha`` (else None), and ``shift_at`` the index where the
        shifted stretch begins (None without a shift).
    """
    n = len(values)
    out: dict[str, Any] = {
        "n": n,
        "p_worse": 1.0,
        "p_better": 1.0,
        "direction": None,
        "shift_at": None,
    }
    if n < 2:
        return out
    r = _ranks([-v if higher_is_better else v for v in values])
    # ~100 null draws beyond the critical value: ~10% Monte Carlo error.
    null = _null(n, math.ceil(100 / alpha))
    at = {}
    for direction, ranks in (("worse", r), ("better", [n + 1 - x for x in r])):
        z, at[direction] = _max_z(ranks)
        # Ranks are exact multiples of 1/2, so 1e-9 only absorbs rounding.
        exceed = len(null) - bisect_left(null, z - 1e-9)
        out[f"p_{direction}"] = (1 + exceed) / (1 + len(null))
    best = min(("worse", "better"), key=lambda d: out[f"p_{d}"])
    if out[f"p_{best}"] <= alpha:
        out["direction"], out["shift_at"] = best, at[best]
    return out


@lru_cache(maxsize=None)
def min_runs(alpha: float) -> int:
    """Fewest runs in a regime with which :func:`detect` can report a shift.

    Args:
        alpha: Per-test false alarm level.

    Returns:
        The smallest ``n`` for which some stretch of the latest runs, each
        worse than every earlier run, reaches ``p_worse <= alpha``.
    """
    n = 2
    while not any(
        detect([*range(n - m), *range(n, n + m)], False, alpha)["direction"]
        for m in range(1, n)
    ):
        n += 1
    return n


def regime(
    history: list[dict[str, Any]],
    workload: str,
    deployment: str,
    cfg_hash: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Return the runs of a metric's current regime, oldest first.

    Args:
        history: Output of :func:`load_history`.
        workload: Workload name.
        deployment: Deployment name.
        cfg_hash: :func:`config_hash` of the runs to use.
        metric: Summary key.

    Returns:
        The pair's runs with this config from the start of the regime that
        its latest run recorded for ``metric``; all of them when that run
        recorded none. Empty when the pair has no run with this config.
    """
    pair = [
        h
        for h in history
        if h["workload"] == workload
        and h["deployment"] == deployment
        and h["config_hash"] == cfg_hash
    ]
    if not pair:
        return []
    start = pair[-1]["regimes"].get(metric)
    ids = [h["run_id"] for h in pair]
    return pair[ids.index(start) :] if start in ids else pair


def check(
    history: list[dict[str, Any]],
    run: dict[str, Any],
    metrics: list[str],
    alpha: float,
) -> dict[str, Any]:
    """Test one run's metrics against their regimes in the history.

    Args:
        history: Output of :func:`load_history`, without ``run``.
        run: Tonight's history record (see :func:`append`); its ``regimes``
            are not read.
        metrics: Summary keys to test.
        alpha: Per-metric, per-direction false alarm level.

    Returns:
        ``{"metrics": {metric: entry}, "regimes": {metric: run id},
        "min_runs": :func:`min_runs`}``. ``regimes`` is what ``run`` must
        record. Each entry has ``value``, ``median`` of the regime before
        tonight, ``change`` (value/median - 1),
        ``n`` (regime runs including tonight), ``p_worse``, ``p_better``,
        ``regressed`` and ``shift`` (``"worse"``, ``"better"`` or None;
        a shift starts a new regime). ``value``-derived fields and
        ``regressed`` are None when tonight has no value for the metric.
    """
    out: dict[str, Any] = {"metrics": {}, "regimes": {}, "min_runs": min_runs(alpha)}
    for m in metrics:
        past = regime(
            history, run["workload"], run["deployment"], run["config_hash"], m
        )
        past = [h for h in past if h["metrics"].get(m) is not None]
        v = run["metrics"].get(m)
        if v is None:
            out["metrics"][m] = {"value": None, "regressed": None, "shift": None}
            out["regimes"][m] = past[0]["run_id"] if past else run["run_id"]
            continue
        runs = [*past, run]
        xs = [h["metrics"][m] for h in runs]
        det = detect(xs, m in HIGHER_IS_BETTER, alpha)
        med = median(xs[:-1]) if past else None
        entry = {
            "value": v,
            "median": med,
            "change": v / med - 1 if med else None,
            "n": det["n"],
            "p_worse": det["p_worse"],
            "p_better": det["p_better"],
            "regressed": det["direction"] == "worse",
            "shift": det["direction"],
        }
        out["metrics"][m] = entry
        s = det["shift_at"] if det["direction"] else 0
        out["regimes"][m] = runs[s]["run_id"]
    return out


def append(path: Path, record: dict[str, Any]) -> None:
    """Add one run to the history.

    Args:
        path: History file; created with its directory if missing.
        record: ``run_id``, ``date``, ``suite``, ``workload``, ``deployment``,
            ``vllm_commit``, ``lmcache_commit``, ``config_hash``, ``metrics``
            (summary keys) and ``regimes`` (from :func:`check`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
