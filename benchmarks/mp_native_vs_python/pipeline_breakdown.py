#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a Python-vs-native LMCache TTFT pipeline breakdown.

This is an artifact reader, not a benchmark runner. It compares saved
long_doc_qa outputs, native /status timing, Python MP coarse logs, and
optional vLLM-only control artifacts.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re


@dataclass(frozen=True)
class ModelInputs:
    name: str
    python_dir: Path
    native_dir: Path
    vllm_only_dir: Path
    hard_target_ms: float


@dataclass(frozen=True)
class QueryStats:
    rows_ms: list[float]

    @property
    def mean_ms(self) -> float:
        if not self.rows_ms:
            raise ValueError("query CSV has no successful TTFT rows")
        return sum(self.rows_ms) / len(self.rows_ms)

    @property
    def steady_ms(self) -> float:
        if len(self.rows_ms) <= 1:
            return self.mean_ms
        rows = self.rows_ms[1:]
        return sum(rows) / len(rows)


def _read_query_stats(directory: Path) -> QueryStats:
    path = directory / "query_round.csv"
    rows_ms: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("successful", "").lower() == "true":
                rows_ms.append(float(row["ttft"]) * 1000.0)
    return QueryStats(rows_ms=rows_ms)


def _read_native_metrics(directory: Path) -> dict[str, object]:
    path = directory / "native_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"{path} must contain a metrics object")
    return metrics


def _request_summary_total_ms(metrics: dict[str, object], key: str, kind: str) -> float:
    summary = metrics.get(key)
    if not isinstance(summary, dict):
        return 0.0
    value = summary.get(kind)
    if not isinstance(value, dict):
        return 0.0
    total_us = value.get("total_us")
    return float(total_us) / 1000.0 if isinstance(total_us, int | float) else 0.0


def _metric_ms_per_prompt(
    metrics: dict[str, object],
    key: str,
    prompt_count: int,
) -> float:
    value = metrics.get(key)
    if not isinstance(value, int | float):
        return 0.0
    return float(value) / float(prompt_count) / 1000.0


def _python_mp_log_summary(directory: Path) -> dict[str, object]:
    path = directory / "mp.log"
    if not path.is_file():
        return {"schema_present": False}

    text = path.read_text(encoding="utf-8", errors="replace")
    prefetch_ms = [
        float(match)
        for match in re.findall(
            r"Prefetch request completed \(L1\+L2\): .*? in ([0-9.]+) ms",
            text,
        )
    ]
    retrieve_ms = [
        float(match) * 1000.0
        for match in re.findall(r"Retrieved 512 tokens in ([0-9.]+) seconds", text)
    ]
    return {
        "schema_present": bool(prefetch_ms or retrieve_ms),
        "prefetch_ms_mean": _mean(prefetch_ms) if prefetch_ms else None,
        "retrieve_ms_mean_coarse": _mean(retrieve_ms) if retrieve_ms else None,
        "prefetch_count": len(prefetch_ms),
        "retrieve_count": len(retrieve_ms),
    }


def _lookup_timing_summary(directory: Path) -> dict[str, object]:
    path = directory / "vllm.log"
    if not path.is_file():
        return {"schema_present": False}

    lookup_rows: list[tuple[str, int, int]] = []
    status_by_request: dict[str, tuple[int, int]] = {}
    retrieve_by_request: dict[str, tuple[int, int]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "LMCache MP lookup submit" in line:
            match = re.search(
                r"request_id=([^ ]+) lookup_us=(\d+) tokens=(\d+)",
                line,
            )
            if match:
                lookup_rows.append(
                    (match.group(1), int(match.group(2)), int(match.group(3)))
                )
        elif "LMCache MP lookup status" in line:
            match = re.search(
                r"request_id=([^ ]+) status_us=(\d+) result=(\d+)",
                line,
            )
            if match:
                status_by_request[match.group(1)] = (
                    int(match.group(2)),
                    int(match.group(3)),
                )
        elif "LMCache MP retrieve completion" in line:
            match = re.search(
                r"request_id=([^ ]+) query_us=(\d+) result_us=(\d+)",
                line,
            )
            if match:
                retrieve_by_request[match.group(1)] = (
                    int(match.group(2)),
                    int(match.group(3)),
                )

    query_hits: list[tuple[int, int]] = []
    query_ids: set[str] = set()
    for request_id, lookup_us, tokens in lookup_rows:
        status = status_by_request.get(request_id)
        if tokens == 513 and status is not None and status[1] > 0:
            query_hits.append((lookup_us, status[0]))
            query_ids.add(request_id)

    retrieve_completion_us = sum(
        query_us + result_us
        for request_id, (query_us, result_us) in retrieve_by_request.items()
        if request_id in query_ids
    )
    combined_us = sum(lookup_us + status_us for lookup_us, status_us in query_hits)
    return {
        "schema_present": bool(query_hits),
        "query_cache_hit_count": len(query_hits),
        "lookup_submit_total_ms": sum(row[0] for row in query_hits) / 1000.0,
        "lookup_status_total_ms": sum(row[1] for row in query_hits) / 1000.0,
        "lookup_status_combined_ms_per_prompt": (
            combined_us / len(query_hits) / 1000.0 if query_hits else None
        ),
        "retrieve_completion_total_us": retrieve_completion_us,
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty list")
    return sum(values) / len(values)


def _bar(value_ms: float | None, *, scale_ms: float, width: int = 46) -> str:
    if value_ms is None:
        return ""
    count = min(width, max(0, round(value_ms / scale_ms)))
    return "#" * count


def _build_model_report(model: ModelInputs) -> dict[str, object]:
    python_stats = _read_query_stats(model.python_dir)
    native_stats = _read_query_stats(model.native_dir)
    vllm_only_stats = _read_query_stats(model.vllm_only_dir)
    native_metrics = _read_native_metrics(model.native_dir)
    prompt_count = len(native_stats.rows_ms)
    native_retrieve_ms = _metric_ms_per_prompt(
        native_metrics,
        "cuda_transfer_retrieve_total_us",
        prompt_count,
    )
    native_retrieve_request_ms = (
        _request_summary_total_ms(native_metrics, "request_type_latency", "retrieve")
        / prompt_count
    )
    return {
        "target_ms": model.hard_target_ms,
        "python_lmcache_mean_ms": python_stats.mean_ms,
        "python_lmcache_steady_ms": python_stats.steady_ms,
        "python_lmcache_rows_ms": python_stats.rows_ms,
        "native_lmcache_mean_ms": native_stats.mean_ms,
        "native_lmcache_steady_ms": native_stats.steady_ms,
        "native_lmcache_rows_ms": native_stats.rows_ms,
        "vllm_only_mean_ms": vllm_only_stats.mean_ms,
        "vllm_only_steady_ms": vllm_only_stats.steady_ms,
        "vllm_only_rows_ms": vllm_only_stats.rows_ms,
        "python_mp_log": _python_mp_log_summary(model.python_dir),
        "native_retrieve_transfer_ms_per_prompt": native_retrieve_ms,
        "native_retrieve_request_latency_ms_per_prompt": native_retrieve_request_ms,
        "native_minus_retrieve_transfer_ms": (
            native_stats.mean_ms - native_retrieve_ms
        ),
        "native_minus_vllm_only_mean_ms": (
            native_stats.mean_ms - vllm_only_stats.mean_ms
        ),
        "native_store_request_latency_total_ms_all_rounds": (
            _request_summary_total_ms(native_metrics, "request_type_latency", "store")
        ),
        "native_lookup_queue_wait_total_ms_all_rounds": (
            _request_summary_total_ms(
                native_metrics,
                "request_type_queue_wait",
                "lookup",
            )
        ),
    }


def _render_text(
    report: dict[str, object],
    *,
    scale_ms: float,
) -> str:
    models = report["models"]
    if not isinstance(models, dict):
        raise TypeError("report models must be a dict")

    lines = [
        "TTFT / prompt pipeline view (ms, lower is better)",
        f"scale: one # ~= {scale_ms:g} ms",
    ]
    for model_name, model_report in models.items():
        if not isinstance(model_report, dict):
            continue
        python_log = model_report.get("python_mp_log")
        python_log = python_log if isinstance(python_log, dict) else {}

        lines.extend(
            [
                "",
                str(model_name),
                _row("2x target", model_report["target_ms"], scale_ms),
                _row("vLLM-only floor", model_report["vllm_only_mean_ms"], scale_ms),
                _row(
                    "Python LMCache",
                    model_report["python_lmcache_mean_ms"],
                    scale_ms,
                ),
                _row("C++ LMCache", model_report["native_lmcache_mean_ms"], scale_ms),
                "  visible LMCache-side work inside those TTFTs:",
                _row(
                    "  Python MP prefetch log mean",
                    python_log.get("prefetch_ms_mean"),
                    scale_ms,
                ),
                _row(
                    "  Python MP retrieve log mean",
                    python_log.get("retrieve_ms_mean_coarse"),
                    scale_ms,
                    suffix="  (coarse 1ms log granularity)",
                ),
                _row(
                    "  C++ native retrieve transfer",
                    model_report["native_retrieve_transfer_ms_per_prompt"],
                    scale_ms,
                ),
            ]
        )
        lookup_timing = model_report.get("native_lookup_timing")
        if isinstance(lookup_timing, dict) and lookup_timing.get("schema_present"):
            lines.append(
                _row(
                    "  C++ lookup+status diag",
                    lookup_timing.get("lookup_status_combined_ms_per_prompt"),
                    scale_ms,
                )
            )
        lines.extend(
            [
                _row(
                    "  C++ TTFT minus retrieve",
                    model_report["native_minus_retrieve_transfer_ms"],
                    scale_ms,
                ),
                _row(
                    "  C++ TTFT minus vLLM-only",
                    model_report["native_minus_vllm_only_mean_ms"],
                    scale_ms,
                ),
            ]
        )
    conclusion = report.get("conclusion")
    if isinstance(conclusion, dict):
        lines.extend(["", "Conclusion:"])
        for key in ("largest_measured_bucket", "why", "next_best_target"):
            value = conclusion.get(key)
            if isinstance(value, str):
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _row(
    label: str,
    value: object,
    scale_ms: float,
    *,
    suffix: str = "",
) -> str:
    if isinstance(value, int | float):
        bar = _bar(float(value), scale_ms=scale_ms)
        return f"  {label:<34} {float(value):7.3f} ms |{bar}{suffix}"
    return f"  {label:<34} {'n/a':>7} ms |{suffix}"


def _build_report(args: argparse.Namespace) -> dict[str, object]:
    models = [
        ModelInputs(
            name="Qwen3-8B",
            python_dir=args.qwen3_8b_python_dir,
            native_dir=args.qwen3_8b_native_dir,
            vllm_only_dir=args.qwen3_8b_vllm_only_dir,
            hard_target_ms=35.237432,
        ),
        ModelInputs(
            name="Qwen3-14B",
            python_dir=args.qwen3_14b_python_dir,
            native_dir=args.qwen3_14b_native_dir,
            vllm_only_dir=args.qwen3_14b_vllm_only_dir,
            hard_target_ms=37.644863,
        ),
    ]
    reports = {model.name: _build_model_report(model) for model in models}
    if args.qwen3_8b_lookup_timing_dir.is_dir():
        reports["Qwen3-8B"]["native_lookup_timing"] = _lookup_timing_summary(
            args.qwen3_8b_lookup_timing_dir
        )
    report: dict[str, object] = {
        "models": reports,
        "conclusion": {
            "largest_measured_bucket": "shared vLLM/client/model first-token path",
            "why": (
                "vLLM-only controls consume tens of milliseconds, while native "
                "retrieve transfer is about 2ms per prompt."
            ),
            "next_best_target": (
                "instrument vLLM first-token path or redefine the benchmark "
                "contract for exact-shape warmup/CUDA graphs with fresh "
                "Python/native baselines"
            ),
        },
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Python-vs-native LMCache pipeline breakdown graph."
    )
    parser.add_argument(
        "--qwen3-8b-python-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-8b-python-1779034941"),
    )
    parser.add_argument(
        "--qwen3-8b-native-dir",
        type=Path,
        default=Path(
            "/tmp/lmcache-long-doc-qa-qwen3-8b-native-faststatus-nolog-1779050718"
        ),
    )
    parser.add_argument(
        "--qwen3-8b-vllm-only-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-8b-vllmonly-1779049198"),
    )
    parser.add_argument(
        "--qwen3-8b-lookup-timing-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookuptiming-1779049672"),
    )
    parser.add_argument(
        "--qwen3-14b-python-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-14b-python-1779035109"),
    )
    parser.add_argument(
        "--qwen3-14b-native-dir",
        type=Path,
        default=Path(
            "/tmp/lmcache-long-doc-qa-qwen3-14b-native-faststatus-nolog-1779050813"
        ),
    )
    parser.add_argument(
        "--qwen3-14b-vllm-only-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-14b-vllmonly-1779052865"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the machine-readable breakdown.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional path for the rendered CLI graph.",
    )
    parser.add_argument(
        "--scale-ms",
        type=float,
        default=2.0,
        help="Milliseconds represented by one bar character.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = _build_report(args)
    text = _render_text(report, scale_ms=args.scale_ms)
    print(text)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.text_output is not None:
        args.text_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
