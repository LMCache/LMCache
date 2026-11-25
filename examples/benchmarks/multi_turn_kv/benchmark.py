# SPDX-License-Identifier: Apache-2.0
"""Run the mixed-length KV workload without changing serving/cache defaults."""

# Standard
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import math
import re
import statistics
import time
import urllib.request

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.multi_turn_kv import (
    ConversationTurn,
    MultiTurnKVConfig,
    MultiTurnKVWorkload,
)

_COUNTERS = {
    "lookup_hit_tokens": "lmcache_mp_lookup_hit_tokens_total",
    "l2_hit_tokens": "lmcache_mp_lookup_hit_l2_tokens_total",
    "l2_loaded_chunks": "lmcache_mp_l2_prefetch_load_completed_chunks_total",
    "l2_failed_chunks": "lmcache_mp_l2_prefetch_failure_chunks_total",
    "l1_evicted_chunks": "lmcache_mp_l1_evicted_chunks_total",
    "l2_evicted_chunks": "lmcache_mp_l2_evicted_objects_chunks_total",
}
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)")


def read_counters(url: str) -> dict[str, float]:
    """Sum known MP counters across labels from an isolated server.

    Args:
        url: Complete Prometheus metrics URL, including the scheme.

    Returns:
        Present finite counter totals. Missing series stay absent.

    Raises:
        OSError: If the endpoint cannot be read.
        ValueError: If the URL is invalid.
    """
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode()
    totals: dict[str, float] = {}
    names = {value: key for key, value in _COUNTERS.items()}
    for line in body.splitlines():
        match = _SAMPLE.match(line)
        if match is None or match[1] not in names:
            continue
        value = float(match[2])
        if math.isfinite(value) and value >= 0:
            key = names[match[1]]
            totals[key] = totals.get(key, 0.0) + value
    return totals


def counter_deltas(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, float | None]:
    """Calculate deltas, preserving unavailable/reset counters as null.

    A series first created during the run starts at zero. Both scrapes must
    succeed; the server must be isolated and must not restart during a run.

    Args:
        before: Pre-run counters, or None if the scrape was unavailable.
        after: Post-run counters, or None if the scrape was unavailable.

    Returns:
        One delta per known counter; resets and missing final series are null.
    """
    result: dict[str, float | None] = {key: None for key in _COUNTERS}
    if before is not None and after is not None:
        for key, value in after.items():
            delta = value - before.get(key, 0.0)
            if delta >= 0:
                result[key] = delta
    return result


def summarize(values: list[float]) -> dict[str, float | None]:
    """Return inclusive p50/p95/p99, or null for an unobserved distribution.

    Args:
        values: Finite measured values in the desired output units.

    Returns:
        Percentiles using linear interpolation between observed values.
    """
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    if len(values) == 1:
        return {"p50": values[0], "p95": values[0], "p99": values[0]}
    percentiles = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": percentiles[49], "p95": percentiles[94], "p99": percentiles[98]}


def turn_summary(
    turns: list[ConversationTurn], ttft_slo_ms: float | None
) -> dict[str, object]:
    """Summarize attempted turns without discarding failed-request queueing.

    Args:
        turns: Turns in one session, cohort, or phase.
        ttft_slo_ms: Optional limit on client queue time plus observed TTFT.

    Returns:
        Counts, queue/TTFT percentiles, and optional SLO success count.
    """
    successful = [turn for turn in turns if turn.result.successful]
    return {
        "attempted": len(turns),
        "successful": len(successful),
        "failed": len(turns) - len(successful),
        "queue_ms": summarize([turn.queue_seconds * 1000 for turn in turns]),
        "ttft_ms": summarize([turn.result.ttft * 1000 for turn in successful]),
        "queue_plus_ttft_ms": summarize(
            [(turn.queue_seconds + turn.result.ttft) * 1000 for turn in successful]
        ),
        "meeting_queue_plus_ttft_slo": (
            sum(
                (turn.queue_seconds + turn.result.ttft) * 1000 <= ttft_slo_ms
                for turn in successful
            )
            if ttft_slo_ms is not None
            else None
        ),
    }


def main() -> None:
    """Parse explicit traffic limits, run vLLM requests, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--model", required=True, help="Serving tokenizer ID or path")
    parser.add_argument("--metrics-url", help="Isolated MP server's /metrics URL")
    parser.add_argument("--output", default="multi_turn_kv.json")
    parser.add_argument(
        "--label", default="unspecified", help="Deployment/policy label"
    )
    parser.add_argument("--require-l2", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--metrics-settle-seconds", type=float, default=1.0)
    defaults = MultiTurnKVConfig()
    for name in asdict(defaults):
        value = getattr(defaults, name)
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(value), default=value
        )
    args = parser.parse_args()
    if args.require_l2 and not args.metrics_url:
        parser.error("--require-l2 needs --metrics-url")
    if (
        not math.isfinite(args.metrics_settle_seconds)
        or args.metrics_settle_seconds < 0
    ):
        parser.error("--metrics-settle-seconds must be finite and nonnegative")
    if args.ttft_slo_ms is not None and (
        not math.isfinite(args.ttft_slo_ms) or args.ttft_slo_ms <= 0
    ):
        parser.error("--ttft-slo-ms must be finite and positive")
    config = MultiTurnKVConfig(**{key: getattr(args, key) for key in asdict(defaults)})
    collector = StatsCollector()
    monitor = ProgressMonitor(collector, quiet=args.quiet)
    sender = RequestSender(
        args.engine_url,
        args.model,
        ignore_eos=args.ignore_eos,
        on_finished=[
            lambda result, _: collector.on_request_finished(result),
            lambda result, _: monitor.on_request_finished(
                result.request_id, result.successful
            ),
        ],
    )
    workload = MultiTurnKVWorkload(config, sender, collector, monitor, args.model)
    snapshots: list[dict[str, float] | None] = []
    scrape_errors: list[str] = []

    def snapshot() -> None:
        if args.metrics_url is None:
            snapshots.append(None)
            return
        try:
            snapshots.append(read_counters(args.metrics_url))
        except (OSError, ValueError) as exc:
            snapshots.append(None)
            scrape_errors.append(str(exc))

    snapshot()
    workload.log_config()
    monitor.start()
    try:
        workload.run()
    finally:
        monitor.stop()
    if args.metrics_url:
        time.sleep(args.metrics_settle_seconds)
    snapshot()
    deltas = counter_deltas(*snapshots)
    evidence = [
        deltas[key]
        for key in ("lookup_hit_tokens", "l2_hit_tokens", "l2_loaded_chunks")
    ]
    l2_status = (
        "unavailable"
        if any(value is None for value in evidence)
        else "observed"
        if all(value is not None and value > 0 for value in evidence)
        else "not_observed"
    )
    turns = workload.get_turns()
    report = {
        "config": asdict(config),
        "model": args.model,
        "label": args.label,
        "ignore_eos": args.ignore_eos,
        "ttft_slo_ms": args.ttft_slo_ms,
        "initial_working_set_tokens": workload.initial_working_set_tokens,
        "summary": turn_summary(turns, args.ttft_slo_ms),
        "unattempted_turns": config.num_sessions * config.turns - len(turns),
        "cohorts": {
            cohort: turn_summary(
                [turn for turn in turns if turn.cohort == cohort], args.ttft_slo_ms
            )
            for cohort in ("short", "long")
        },
        "phases": {
            phase: turn_summary(
                [turn for turn in turns if turn.phase == phase], args.ttft_slo_ms
            )
            for phase in ("initial", "continuation")
        },
        "sessions": {
            str(session): turn_summary(
                [turn for turn in turns if turn.session_id == session], args.ttft_slo_ms
            )
            for session in range(config.num_sessions)
        },
        "mp_counter_deltas": deltas,
        "mp_counter_snapshots": snapshots,
        "metrics_errors": scrape_errors,
        "l2_activity": l2_status,
        "unavailable_measurements": [
            "per-token inter-token latency",
            "actual L2 transferred bytes",
            "GPU KV-pool usage",
            "GPU temporary/pinned-for-transfer usage",
            "host L1 allocated/staging peaks",
            "per-request fallback reasons",
            "server admission queue time",
        ],
        "turns": [asdict(turn) for turn in turns],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output}; L2 activity: {l2_status}")
    if any(not turn.result.successful for turn in turns):
        raise SystemExit(1)
    if args.require_l2 and l2_status != "observed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
