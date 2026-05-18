#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate Python-vs-native MP benchmark artifacts.

The benchmark launchers stay separate. This helper only reads their saved JSON,
CSV, and stdout artifacts and applies the GOAL.md rule: native C++ MP must beat
Python MP on every comparable measured unit across the supplied runs.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import glob
import hashlib
import json
import sys

LOWER_IS_BETTER = "lower"
HIGHER_IS_BETTER = "higher"
EQUAL_REQUIRED = "equal"


@dataclass(frozen=True)
class MetricKey:
    group: str
    metric: str
    direction: str


MetricSamples = dict[MetricKey, dict[str, list[float]]]


def _flatten(values: list[list[str]] | None) -> list[str]:
    if not values:
        return []
    return [item for group in values for item in group]


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _load_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} did not contain a JSON object")
    return data


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _add_metric(
    samples: MetricSamples,
    group: str,
    metric: str,
    direction: str,
    server: str,
    value: object,
) -> None:
    number = _number(value)
    if number is None:
        return
    key = MetricKey(group=group, metric=metric, direction=direction)
    samples.setdefault(key, {"python": [], "native": []})[server].append(number)


def _nested(values: dict[str, object], *keys: str) -> object:
    current: object = values
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _add_controller_report(samples: MetricSamples, path: Path) -> None:
    report = _load_json(path)
    python_report = report.get("python")
    native_report = report.get("native")
    if not isinstance(python_report, dict) or not isinstance(native_report, dict):
        raise RuntimeError(f"{path} is not a controller comparison report")

    request = str(report.get("request", "unknown"))
    clients = int(_number(python_report.get("client_count")) or 0)
    group = f"controller/{request}/clients={clients}"
    for server, values in (("python", python_report), ("native", native_report)):
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            _add_metric(
                samples,
                group,
                metric,
                LOWER_IS_BETTER,
                server,
                values.get(metric),
            )
        _add_metric(
            samples,
            group,
            "requests_per_s",
            HIGHER_IS_BETTER,
            server,
            values.get("requests_per_s"),
        )
        resource_delta = values.get("server_resources_delta")
        if isinstance(resource_delta, dict):
            for metric in (
                "rss_peak_bytes",
                "total_cpu_s_delta",
                "thread_count_end",
            ):
                _add_metric(
                    samples,
                    group,
                    metric,
                    LOWER_IS_BETTER,
                    server,
                    resource_delta.get(metric),
                )


def _add_vllm_report(
    samples: MetricSamples,
    checks: list[dict[str, object]],
    path: Path,
) -> None:
    report = _load_json(path)
    group = "vllm-smoke"

    top_level_metrics = (
        ("second_generation_elapsed_s", LOWER_IS_BETTER),
        ("second_generation_generate_elapsed_s", LOWER_IS_BETTER),
        ("steady_state_output_tokens_per_s", HIGHER_IS_BETTER),
        ("steady_state_ttft_s_mean", LOWER_IS_BETTER),
    )
    for metric, direction in top_level_metrics:
        values = report.get(metric)
        if not isinstance(values, dict):
            continue
        for server in ("python", "native"):
            _add_metric(samples, group, metric, direction, server, values.get(server))

    resources = report.get("server_resources_delta")
    if isinstance(resources, dict):
        for server in ("python", "native"):
            resource_delta = resources.get(server)
            if not isinstance(resource_delta, dict):
                continue
            for metric in (
                "rss_peak_bytes",
                "total_cpu_s_delta",
                "thread_count_end",
            ):
                _add_metric(
                    samples,
                    group,
                    metric,
                    LOWER_IS_BETTER,
                    server,
                    resource_delta.get(metric),
                )

    request_latency = report.get("mp_request_latency_ms")
    if isinstance(request_latency, dict):
        python_latency = request_latency.get("python")
        native_latency = request_latency.get("native")
        if isinstance(python_latency, dict) and isinstance(native_latency, dict):
            for request_type in sorted(set(python_latency) & set(native_latency)):
                request_group = f"{group}/request={request_type}"
                for server, by_request in (
                    ("python", python_latency),
                    ("native", native_latency),
                ):
                    stats = by_request.get(request_type)
                    if not isinstance(stats, dict):
                        continue
                    _add_metric(
                        samples,
                        request_group,
                        "count",
                        EQUAL_REQUIRED,
                        server,
                        stats.get("count"),
                    )
                    for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
                        _add_metric(
                            samples,
                            request_group,
                            metric,
                            LOWER_IS_BETTER,
                            server,
                            stats.get(metric),
                        )

    python_text = _nested(report, "python", "second_generation", "output_text")
    native_text = _nested(report, "native", "second_generation", "output_text")
    if python_text is not None or native_text is not None:
        checks.append(
            {
                "group": group,
                "check": "second_generation_output_text_equal",
                "passed": python_text == native_text,
            }
        )


def _last_json_object_from_stdout(path: Path) -> dict[str, object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise RuntimeError(f"{path} did not contain a JSON summary line")


def _csv_counts(path: Path) -> tuple[int, int]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    successful = sum(1 for row in rows if row.get("successful") == "True")
    return len(rows), successful


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_long_doc_spec(spec: str) -> tuple[str, str, Path]:
    try:
        server, model, directory = spec.split(":", maxsplit=2)
    except ValueError as exc:
        raise RuntimeError(
            "--long-doc-run must use SERVER:MODEL:DIR, for example "
            "python:Qwen3-8B:/tmp/run"
        ) from exc
    if server not in {"python", "native"}:
        raise RuntimeError("--long-doc-run SERVER must be 'python' or 'native'")
    return server, model, Path(directory)


def _add_long_doc_run(
    samples: MetricSamples,
    response_hashes: dict[str, dict[str, list[str]]],
    spec: str,
) -> None:
    server, model, directory = _split_long_doc_spec(spec)
    if not directory.is_dir():
        raise RuntimeError(f"long-doc artifact directory does not exist: {directory}")

    group = f"long-doc/{model}"
    summary = _last_json_object_from_stdout(directory / "bench.stdout")
    for metric in (
        "query_ttft_per_prompt",
        "query_round_time_per_prompt",
        "warmup_round_time_per_prompt",
    ):
        _add_metric(
            samples,
            group,
            metric,
            LOWER_IS_BETTER,
            server,
            summary.get(metric),
        )

    query_prompt_count, query_success_count = _csv_counts(directory / "query_round.csv")
    warmup_prompt_count, warmup_success_count = _csv_counts(
        directory / "warmup_round.csv"
    )
    for metric, value in (
        ("query_prompt_count", query_prompt_count),
        ("query_success_count", query_success_count),
        ("warmup_prompt_count", warmup_prompt_count),
        ("warmup_success_count", warmup_success_count),
    ):
        _add_metric(samples, group, metric, EQUAL_REQUIRED, server, value)

    response_hash = _sha256(directory / "responses.txt")
    if response_hash is not None:
        response_hashes.setdefault(model, {"python": [], "native": []})[server].append(
            response_hash
        )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_passed(direction: str, python_avg: float, native_avg: float) -> bool:
    if direction == LOWER_IS_BETTER:
        return native_avg < python_avg
    if direction == HIGHER_IS_BETTER:
        return native_avg > python_avg
    if direction == EQUAL_REQUIRED:
        return native_avg == python_avg
    raise RuntimeError(f"unknown metric direction: {direction}")


def _summarize_samples(
    samples: MetricSamples,
    min_runs: int,
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for key in sorted(samples, key=lambda item: (item.group, item.metric)):
        by_server = samples[key]
        python_values = by_server["python"]
        native_values = by_server["native"]
        python_avg = _mean(python_values)
        native_avg = _mean(native_values)
        complete = len(python_values) >= min_runs and len(native_values) >= min_runs
        passed = False
        if complete and python_avg is not None and native_avg is not None:
            passed = _metric_passed(key.direction, python_avg, native_avg)
        ratio = None
        if python_avg not in (None, 0.0) and native_avg is not None:
            ratio = native_avg / python_avg
        comparisons.append(
            {
                "group": key.group,
                "metric": key.metric,
                "direction": key.direction,
                "python_avg": python_avg,
                "native_avg": native_avg,
                "native_over_python": ratio,
                "python_run_count": len(python_values),
                "native_run_count": len(native_values),
                "complete": complete,
                "passed": passed,
            }
        )
    return comparisons


def _add_response_hash_checks(
    checks: list[dict[str, object]],
    response_hashes: dict[str, dict[str, list[str]]],
    min_runs: int,
) -> None:
    for model, by_server in sorted(response_hashes.items()):
        python_hashes = by_server["python"]
        native_hashes = by_server["native"]
        checks.append(
            {
                "group": f"long-doc/{model}",
                "check": "responses_sha256_equal",
                "python_run_count": len(python_hashes),
                "native_run_count": len(native_hashes),
                "complete": len(python_hashes) >= min_runs
                and len(native_hashes) >= min_runs,
                "passed": bool(python_hashes)
                and bool(native_hashes)
                and set(python_hashes) == set(native_hashes)
                and len(python_hashes) >= min_runs
                and len(native_hashes) >= min_runs,
            }
        )


def _print_text_summary(report: dict[str, object]) -> None:
    comparisons = report["comparisons"]
    checks = report["correctness_checks"]
    if not isinstance(comparisons, list) or not isinstance(checks, list):
        return
    print(
        "five_run_compare: "
        f"passed={report['passed']} complete={report['complete']} "
        f"comparisons={len(comparisons)} checks={len(checks)}"
    )
    failing = [
        item
        for item in comparisons
        if isinstance(item, dict) and not item.get("passed")
    ]
    failing_checks = [
        item for item in checks if isinstance(item, dict) and not item.get("passed")
    ]
    if failing:
        print("failing metrics:")
        for item in failing[:20]:
            print(
                "  "
                f"{item['group']} {item['metric']} "
                f"direction={item['direction']} "
                f"python={item['python_avg']} native={item['native_avg']} "
                f"runs={item['python_run_count']}/{item['native_run_count']}"
            )
        if len(failing) > 20:
            print(f"  ... {len(failing) - 20} more")
    if failing_checks:
        print("failing correctness checks:")
        for item in failing_checks:
            print(f"  {item['group']} {item['check']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-report",
        action="append",
        nargs="+",
        default=[],
        help="Controller latency JSON report path or glob. Repeatable.",
    )
    parser.add_argument(
        "--vllm-report",
        action="append",
        nargs="+",
        default=[],
        help="vLLM native-vs-Python smoke JSON report path or glob. Repeatable.",
    )
    parser.add_argument(
        "--long-doc-run",
        action="append",
        default=[],
        help="Long-doc artifact in SERVER:MODEL:DIR form. Repeatable.",
    )
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after writing the report.",
    )
    args = parser.parse_args()
    if args.min_runs < 1:
        raise ValueError("--min-runs must be at least 1")

    samples: MetricSamples = {}
    checks: list[dict[str, object]] = []
    response_hashes: dict[str, dict[str, list[str]]] = {}

    for path in _expand_paths(_flatten(args.controller_report)):
        _add_controller_report(samples, path)
    for path in _expand_paths(_flatten(args.vllm_report)):
        _add_vllm_report(samples, checks, path)
    for spec in args.long_doc_run:
        _add_long_doc_run(samples, response_hashes, spec)
    _add_response_hash_checks(checks, response_hashes, args.min_runs)

    comparisons = _summarize_samples(samples, args.min_runs)
    complete = all(item["complete"] for item in comparisons) and all(
        bool(item.get("complete", True)) for item in checks
    )
    passed = complete and all(item["passed"] for item in comparisons) and all(
        item["passed"] for item in checks
    )
    report: dict[str, object] = {
        "passed": passed,
        "complete": complete,
        "min_runs": args.min_runs,
        "comparisons": comparisons,
        "correctness_checks": checks,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_text_summary(report)
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    sys.exit(main())
