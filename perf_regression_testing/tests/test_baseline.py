# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import random

# First Party
from perf_regression_testing.runner import baseline, config

METRICS = ["decode_p10", "ttft_p95"]
ALPHA = 0.033 / len(METRICS)


def _run(
    run_id: str,
    p10: float | None,
    ttft: float,
    deployment: str = "lmc",
    vllm: str = "v1",
    cfg: str = "c1",
    lmc: str = "l1",
) -> dict:
    return {
        "run_id": run_id,
        "workload": "agentic",
        "deployment": deployment,
        "vllm_commit": vllm,
        "lmcache_commit": lmc,
        "config_hash": cfg,
        "metrics": {"decode_p10": p10, "ttft_p95": ttft},
    }


def _replay(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Check each run against the ones before it, as nightly suites would."""
    history: list[dict] = []
    results: list[dict] = []
    for r in runs:
        res = baseline.check(history, r, METRICS, ALPHA)
        history.append({**r, "regimes": res["regimes"]})
        results.append(res)
    return history, results


def test_missing_history_is_empty(tmp_path: Path) -> None:
    assert baseline.load_history(tmp_path / "none.jsonl") == []


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "history.jsonl"
    baseline.append(path, _run("a", 40.0, 3.0))
    baseline.append(path, _run("b", 41.0, 3.1))
    assert [h["run_id"] for h in baseline.load_history(path)] == ["a", "b"]


def test_no_shift_can_be_reported_before_min_runs() -> None:
    n = baseline.min_runs(ALPHA)
    # The most extreme ordering: the last half worse than everything before.
    worst = [*range(n // 2), *range(100, 100 + n - n // 2)]
    assert baseline.detect(worst, False, ALPHA)["direction"] == "worse"
    short = worst[1:]
    assert baseline.detect(short, False, ALPHA)["direction"] is None


def test_noise_alone_stays_within_the_false_alarm_budget() -> None:
    rng = random.Random(0)
    alarms = nights = 0
    for s in range(60):
        runs = [
            _run(f"{s}-{i}", rng.gauss(40, 9), rng.lognormvariate(2.3, 0.25))
            for i in range(40)
        ]
        _, results = _replay(runs)
        alarms += sum(
            any(c["regressed"] for c in r["metrics"].values()) for r in results
        )
        nights += len(results)
    assert alarms / nights <= 0.033


def test_one_bad_night_on_a_noisy_metric_is_not_a_regression() -> None:
    rng = random.Random(1)
    runs = [_run(f"r{i}", 40.0 + i % 3, rng.uniform(8, 13)) for i in range(15)]
    runs.append(_run("bad", 40.0, 20.0))
    _, results = _replay(runs)
    assert not results[-1]["metrics"]["ttft_p95"]["regressed"]


def test_persistent_shift_fails_once_then_becomes_the_baseline() -> None:
    rng = random.Random(2)
    runs = [_run(f"a{i}", 40.0, rng.uniform(9, 11)) for i in range(12)]
    runs += [_run(f"b{i}", 40.0, rng.uniform(14, 16), lmc="l2") for i in range(10)]
    _, results = _replay(runs)
    flagged = [
        i for i, r in enumerate(results) if r["metrics"]["ttft_p95"]["regressed"]
    ]
    assert len(flagged) == 1
    assert results[flagged[0]]["metrics"]["ttft_p95"]["shift"] == "worse"
    # The shift starts a new regime: later nights compare with b0 onwards.
    assert results[-1]["regimes"]["ttft_p95"] == "b0"
    assert not results[flagged[0]]["metrics"]["decode_p10"]["regressed"]


def test_higher_is_better_metrics_regress_when_they_drop() -> None:
    rng = random.Random(3)
    up = [_run(f"a{i}", rng.uniform(39, 41), 3.0) for i in range(12)]
    up += [_run(f"b{i}", rng.uniform(49, 51), 3.0, vllm="v2") for i in range(8)]
    _, results = _replay(up)
    shifts = [r["metrics"]["decode_p10"]["shift"] for r in results]
    assert [s for s in shifts if s] == ["better"]
    assert not any(r["metrics"]["decode_p10"]["regressed"] for r in results)
    down = [_run(f"a{i}", rng.uniform(39, 41), 3.0) for i in range(12)]
    down += [_run(f"b{i}", rng.uniform(29, 31), 3.0, vllm="v2") for i in range(8)]
    _, results = _replay(down)
    hits = [
        r["metrics"]["decode_p10"]
        for r in results
        if r["metrics"]["decode_p10"]["regressed"]
    ]
    assert len(hits) == 1


def test_check_uses_only_the_same_config_hash() -> None:
    history = [_run(f"o{i}", 40.0, float(i)) for i in range(12)]
    history = [{**h, "config_hash": "c0", "regimes": {}} for h in history]
    history.append({**_run("r1", 40.0, 3.0), "regimes": {}})
    res = baseline.check(history, _run("r2", 40.0, 3.0), METRICS, ALPHA)
    assert res["metrics"]["ttft_p95"]["n"] == 2
    assert res["regimes"]["ttft_p95"] == "r1"


def test_missing_metric_tonight_is_not_tested() -> None:
    history = [{**_run("r1", 40.0, 3.0), "regimes": {}}]
    res = baseline.check(history, _run("r2", None, 3.0), METRICS, ALPHA)
    assert res["metrics"]["decode_p10"]["regressed"] is None
    assert res["regimes"]["decode_p10"] == "r1"


def test_config_hash_tracks_run_settings_not_timeouts() -> None:
    cfg = config.load()
    spec = config.resolve(cfg, "agentic", "vllm_lmcache")
    h = baseline.config_hash(spec)
    assert h == baseline.config_hash(config.resolve(cfg, "agentic", "vllm_lmcache"))
    assert h != baseline.config_hash(config.resolve(cfg, "agentic", "vllm_native"))
    spec.vllm_startup_timeout_s += 1
    assert baseline.config_hash(spec) == h
    spec.vllm_args["max-num-seqs"] = "1"
    assert baseline.config_hash(spec) != h
