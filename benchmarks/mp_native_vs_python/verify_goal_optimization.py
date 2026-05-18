#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify saved artifacts against GOAL_OPTIMIZATION.md.

This is an artifact verifier, not a benchmark runner. It reads the preserved
Python/native long_doc_qa outputs, checks the hard TTFT target, and exits
non-zero until the optimization goal is actually met.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import filecmp
import json
import re
import sys


@dataclass(frozen=True)
class ModelArtifacts:
    name: str
    python_dir: Path
    native_dir: Path
    native_baseline_ttft_s: float
    native_hard_target_s: float


@dataclass(frozen=True)
class QueryCsv:
    ttfts: list[float]
    successful_count: int
    cache_hit_count: int
    row_count: int

    @property
    def mean_ttft(self) -> float:
        if not self.ttfts:
            raise ValueError("query CSV has no TTFT values")
        return sum(self.ttfts) / len(self.ttfts)

    @property
    def steady_state_mean_ttft(self) -> float:
        if len(self.ttfts) <= 1:
            return self.mean_ttft
        values = self.ttfts[1:]
        return sum(values) / len(values)


def _read_json(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                data = json.loads(stripped)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            raise
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _read_summary_artifact(directory: Path) -> dict[str, object]:
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return _read_json(summary_path)
    return _read_json(directory / "bench.stdout")


def _read_query_csv(path: Path) -> QueryCsv:
    ttfts: list[float] = []
    successful_count = 0
    cache_hit_count = 0
    row_count = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_count += 1
            ttfts.append(float(row["ttft"]))
            if row.get("successful", "").lower() == "true":
                successful_count += 1
            if row.get("is_miss", "").lower() == "false":
                cache_hit_count += 1
    return QueryCsv(
        ttfts=ttfts,
        successful_count=successful_count,
        cache_hit_count=cache_hit_count,
        row_count=row_count,
    )


def _metric(status: dict[str, object], key: str) -> int:
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("native status must contain a metrics object")
    value = metrics.get(key)
    if not isinstance(value, int):
        raise TypeError(f"native status metrics.{key} must be an int")
    return value


def _optional_metric(status: dict[str, object], key: str) -> int | None:
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return value if isinstance(value, int) else None


def _cache_value(status: dict[str, object], key: str) -> int:
    cache = status.get("cache")
    if not isinstance(cache, dict):
        raise TypeError("native status must contain a cache object")
    value = cache.get(key)
    if not isinstance(value, int):
        raise TypeError(f"native status cache.{key} must be an int")
    return value


def _summary_value(summary: dict[str, object], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"summary {key} must be numeric")
    return float(value)


def _hot_cache_report(status: dict[str, object]) -> dict[str, object]:
    enabled = status.get("cuda_gpu_hot_cache_enabled")
    cache = status.get("cuda_gpu_hot_cache")
    report: dict[str, object] = {
        "enabled": enabled is True,
        "schema_present": isinstance(cache, dict),
    }
    if isinstance(cache, dict):
        entries = cache.get("entries")
        bytes_used = cache.get("bytes")
        if isinstance(entries, int):
            report["entries"] = entries
        if isinstance(bytes_used, int):
            report["bytes"] = bytes_used
    return report


def _lock_timing_report(status: dict[str, object]) -> dict[str, object]:
    metrics = status.get("metrics")
    keys = (
        "transfer_lock_wait_total_us",
        "transfer_lock_wait_max_us",
        "transfer_lock_hold_total_us",
        "transfer_lock_hold_max_us",
    )
    if not isinstance(metrics, dict):
        return {"schema_present": False}
    report: dict[str, object] = {
        "schema_present": all(isinstance(metrics.get(key), int) for key in keys),
    }
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, int):
            report[key] = value
    return report


def _request_type_summary_report(
    status: dict[str, object],
    key: str,
) -> dict[str, object]:
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        return {"schema_present": False}
    summary = metrics.get(key)
    request_types = ("lookup", "store", "retrieve", "free_lookup_locks")
    if not isinstance(summary, dict):
        return {"schema_present": False}

    report: dict[str, object] = {"schema_present": True}
    for request_type in request_types:
        value = summary.get(request_type)
        if not isinstance(value, dict):
            report["schema_present"] = False
            continue
        fields: dict[str, int] = {}
        for key in ("count", "total_us", "max_us"):
            field_value = value.get(key)
            if not isinstance(field_value, int):
                report["schema_present"] = False
                continue
            fields[key] = field_value
        report[request_type] = fields
    return report


def _request_type_latency_report(status: dict[str, object]) -> dict[str, object]:
    return _request_type_summary_report(status, "request_type_latency")


def _request_type_queue_wait_report(
    status: dict[str, object],
) -> dict[str, object]:
    return _request_type_summary_report(status, "request_type_queue_wait")


def _vllm_log_report(path: Path) -> dict[str, object]:
    terms = {
        "enforce_eager": "enforce_eager",
        "cudagraph_disabled": "Cudagraph is disabled under eager mode",
        "triton_jit_during_inference": (
            "Triton kernel JIT compilation during inference"
        ),
    }
    report: dict[str, object] = {
        "path": str(path),
        "present": path.is_file(),
    }
    if not path.is_file():
        for key in terms:
            report[key] = False
        return report

    text = path.read_text(encoding="utf-8", errors="replace")
    for key, term in terms.items():
        report[key] = term in text
    return report


def _connector_timing_report(
    timing_dir: Path,
    *,
    reference_responses: Path,
    max_total_us: int,
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    report: dict[str, object] = {
        "dir": str(timing_dir),
        "present": timing_dir.is_dir(),
        "max_total_us": max_total_us,
    }
    if not timing_dir.is_dir():
        return [f"adapter timing artifact is missing: {timing_dir}"], report

    stdout_path = timing_dir / "bench.stdout"
    status_path = timing_dir / "native_status.json"
    responses_path = timing_dir / "responses.txt"
    vllm_log_path = timing_dir / "vllm.log"
    required_files = (stdout_path, status_path, responses_path, vllm_log_path)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    report["missing_files"] = missing_files
    if missing_files:
        failures.append(
            "adapter timing artifact is missing required files: "
            + ", ".join(missing_files)
        )
        return failures, report

    timing_summary = _read_json(stdout_path)
    report["query_ttft_s"] = _summary_value(
        timing_summary,
        "query_ttft_per_prompt",
    )
    report["query_round_s"] = _summary_value(
        timing_summary,
        "query_round_time_per_prompt",
    )
    report["responses_match_reference"] = filecmp.cmp(
        reference_responses,
        responses_path,
        shallow=False,
    )
    if not report["responses_match_reference"]:
        failures.append("adapter timing responses.txt differs from Python reference")

    status = _read_json(status_path)
    contexts = status.get("registered_contexts")
    use_layerwise_values = []
    if isinstance(contexts, list):
        for context in contexts:
            if isinstance(context, dict):
                use_layerwise_values.append(context.get("use_layerwise"))
    report["registered_use_layerwise_values"] = use_layerwise_values
    report["non_layerwise_path"] = any(value is False for value in use_layerwise_values)
    if not report["non_layerwise_path"]:
        failures.append(
            "adapter timing artifact did not record use_layerwise=false"
        )

    text = vllm_log_path.read_text(encoding="utf-8", errors="replace")
    timing_rows = [
        {
            "query_us": int(match.group("query_us")),
            "result_us": int(match.group("result_us")),
            "blocks": int(match.group("blocks")),
            "result": match.group("result"),
        }
        for match in re.finditer(
            r"LMCache MP retrieve completion request_id=\S+ "
            r"query_us=(?P<query_us>\d+) result_us=(?P<result_us>\d+) "
            r"blocks=(?P<blocks>\d+) result=(?P<result>\S+)",
            text,
        )
    ]
    total_query_us = sum(row["query_us"] for row in timing_rows)
    total_result_us = sum(row["result_us"] for row in timing_rows)
    total_us = total_query_us + total_result_us
    report.update(
        {
            "retrieve_completion_count": len(timing_rows),
            "retrieve_completion_rows": timing_rows,
            "total_query_us": total_query_us,
            "total_result_us": total_result_us,
            "total_us": total_us,
            "max_query_us": max(
                (row["query_us"] for row in timing_rows),
                default=0,
            ),
            "max_result_us": max(
                (row["result_us"] for row in timing_rows),
                default=0,
            ),
        }
    )
    if len(timing_rows) != 4:
        failures.append(
            "adapter timing artifact expected 4 retrieve completion rows, "
            f"found {len(timing_rows)}"
        )
    if any(row["result"] != "True" for row in timing_rows):
        failures.append("adapter timing artifact recorded a failed retrieve")
    if total_us > max_total_us:
        failures.append(
            "adapter timing retrieve completion wait "
            f"{total_us}us exceeds max {max_total_us}us"
        )
    return failures, report


def _vllm_only_report(
    vllm_only_dir: Path,
    *,
    reference_responses: Path,
    native_hard_target_s: float,
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    report: dict[str, object] = {
        "dir": str(vllm_only_dir),
        "present": vllm_only_dir.is_dir(),
        "native_hard_target_s": native_hard_target_s,
    }
    if not vllm_only_dir.is_dir():
        return [f"vLLM-only artifact is missing: {vllm_only_dir}"], report

    stdout_path = vllm_only_dir / "bench.stdout"
    query_path = vllm_only_dir / "query_round.csv"
    responses_path = vllm_only_dir / "responses.txt"
    vllm_log_path = vllm_only_dir / "vllm.log"
    required_files = (stdout_path, query_path, responses_path, vllm_log_path)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    report["missing_files"] = missing_files
    if missing_files:
        failures.append(
            "vLLM-only artifact is missing required files: "
            + ", ".join(missing_files)
        )
        return failures, report

    summary = _read_json(stdout_path)
    query = _read_query_csv(query_path)
    ttft_s = _summary_value(summary, "query_ttft_per_prompt")
    query_round_s = _summary_value(summary, "query_round_time_per_prompt")
    report.update(
        {
            "query_ttft_s": ttft_s,
            "query_round_s": query_round_s,
            "query_ttft_gap_to_target_s": ttft_s - native_hard_target_s,
            "query_rows": query.row_count,
            "successful_query_rows": query.successful_count,
            "benchmark_is_miss_false_rows": query.cache_hit_count,
            "per_prompt_ttft_s": query.ttfts,
            "fastest_prompt_ttft_s": min(query.ttfts) if query.ttfts else None,
            "fastest_prompt_gap_to_target_s": (
                min(query.ttfts) - native_hard_target_s if query.ttfts else None
            ),
            "responses_match_reference": filecmp.cmp(
                reference_responses,
                responses_path,
                shallow=False,
            ),
        }
    )
    if abs(query.mean_ttft - ttft_s) > 1e-9:
        failures.append("vLLM-only summary TTFT does not match query CSV")
    if query.successful_count != query.row_count:
        failures.append("vLLM-only query CSV contains unsuccessful rows")
    if not report["responses_match_reference"]:
        failures.append("vLLM-only responses.txt differs from Python reference")

    log_text = vllm_log_path.read_text(encoding="utf-8", errors="replace")
    log_report = _vllm_log_report(vllm_log_path)
    log_report["kv_transfer_config_present"] = "kv_transfer_config" in log_text
    log_report["prefix_caching_disabled"] = "enable_prefix_caching=False" in log_text
    report["vllm_log"] = log_report
    if log_report["kv_transfer_config_present"]:
        failures.append("vLLM-only log unexpectedly contains kv_transfer_config")
    if not log_report["prefix_caching_disabled"]:
        failures.append("vLLM-only log does not show prefix caching disabled")
    return failures, report


def _lookup_timing_report(
    timing_dir: Path,
    *,
    reference_responses: Path,
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    report: dict[str, object] = {
        "dir": str(timing_dir),
        "present": timing_dir.is_dir(),
    }
    if not timing_dir.is_dir():
        return [f"lookup timing artifact is missing: {timing_dir}"], report

    stdout_path = timing_dir / "bench.stdout"
    query_path = timing_dir / "query_round.csv"
    status_path = timing_dir / "native_status.json"
    responses_path = timing_dir / "responses.txt"
    vllm_log_path = timing_dir / "vllm.log"
    required_files = (
        stdout_path,
        query_path,
        status_path,
        responses_path,
        vllm_log_path,
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    report["missing_files"] = missing_files
    if missing_files:
        failures.append(
            "lookup timing artifact is missing required files: "
            + ", ".join(missing_files)
        )
        return failures, report

    timing_summary = _read_json(stdout_path)
    query = _read_query_csv(query_path)
    ttft_s = _summary_value(timing_summary, "query_ttft_per_prompt")
    report.update(
        {
            "query_ttft_s": ttft_s,
            "query_round_s": _summary_value(
                timing_summary,
                "query_round_time_per_prompt",
            ),
            "per_prompt_ttft_s": query.ttfts,
            "query_rows": query.row_count,
            "successful_query_rows": query.successful_count,
            "benchmark_is_miss_false_rows": query.cache_hit_count,
            "responses_match_reference": filecmp.cmp(
                reference_responses,
                responses_path,
                shallow=False,
            ),
        }
    )
    if abs(query.mean_ttft - ttft_s) > 1e-9:
        failures.append("lookup timing summary TTFT does not match query CSV")
    if query.successful_count != query.row_count:
        failures.append("lookup timing query CSV contains unsuccessful rows")
    if query.cache_hit_count != query.row_count:
        failures.append("lookup timing query CSV contains cache-miss rows")
    if not report["responses_match_reference"]:
        failures.append("lookup timing responses.txt differs from Python reference")

    status = _read_json(status_path)
    contexts = status.get("registered_contexts")
    use_layerwise_values = []
    if isinstance(contexts, list):
        for context in contexts:
            if isinstance(context, dict):
                use_layerwise_values.append(context.get("use_layerwise"))
    report["registered_use_layerwise_values"] = use_layerwise_values
    report["non_layerwise_path"] = any(value is False for value in use_layerwise_values)
    if not report["non_layerwise_path"]:
        failures.append("lookup timing artifact did not record use_layerwise=false")
    report["native_request_type_latency"] = _request_type_latency_report(status)
    report["native_request_type_queue_wait"] = _request_type_queue_wait_report(status)

    text = vllm_log_path.read_text(encoding="utf-8", errors="replace")
    submit_rows = [
        {
            "request_id": match.group("request_id"),
            "lookup_us": int(match.group("lookup_us")),
            "tokens": int(match.group("tokens")),
            "aligned_tokens": int(match.group("aligned_tokens")),
        }
        for match in re.finditer(
            r"LMCache MP lookup submit request_id=(?P<request_id>\S+) "
            r"lookup_us=(?P<lookup_us>\d+) tokens=(?P<tokens>\d+) "
            r"aligned_tokens=(?P<aligned_tokens>\d+)",
            text,
        )
    ]
    status_rows = [
        {
            "request_id": match.group("request_id"),
            "status_us": int(match.group("status_us")),
            "result": match.group("result"),
        }
        for match in re.finditer(
            r"LMCache MP lookup status request_id=(?P<request_id>\S+) "
            r"status_us=(?P<status_us>\d+) result=(?P<result>\S+)",
            text,
        )
    ]
    retrieve_rows = [
        {
            "request_id": match.group("request_id"),
            "query_us": int(match.group("query_us")),
            "result_us": int(match.group("result_us")),
            "blocks": int(match.group("blocks")),
            "result": match.group("result"),
        }
        for match in re.finditer(
            r"LMCache MP retrieve completion request_id=(?P<request_id>\S+) "
            r"query_us=(?P<query_us>\d+) result_us=(?P<result_us>\d+) "
            r"blocks=(?P<blocks>\d+) result=(?P<result>\S+)",
            text,
        )
    ]

    submit_by_id = {str(row["request_id"]): row for row in submit_rows}
    status_by_id = {str(row["request_id"]): row for row in status_rows}
    cache_hit_lookup_rows = [
        {
            "request_id": str(status_row["request_id"]),
            "lookup_us": int(submit_by_id[str(status_row["request_id"])]["lookup_us"]),
            "status_us": int(status_row["status_us"]),
            "tokens": int(submit_by_id[str(status_row["request_id"])]["tokens"]),
            "aligned_tokens": int(
                submit_by_id[str(status_row["request_id"])]["aligned_tokens"]
            ),
            "result": str(status_row["result"]),
        }
        for status_row in status_rows
        if str(status_row["result"]) != "0"
        and str(status_row["request_id"]) in submit_by_id
    ]

    submit_total_us = sum(int(row["lookup_us"]) for row in submit_rows)
    status_total_us = sum(int(row["status_us"]) for row in status_rows)
    retrieve_query_total_us = sum(int(row["query_us"]) for row in retrieve_rows)
    retrieve_result_total_us = sum(int(row["result_us"]) for row in retrieve_rows)
    cache_hit_lookup_total_us = sum(
        int(row["lookup_us"]) for row in cache_hit_lookup_rows
    )
    cache_hit_status_total_us = sum(
        int(row["status_us"]) for row in cache_hit_lookup_rows
    )
    report.update(
        {
            "lookup_submit_count": len(submit_rows),
            "lookup_submit_rows": submit_rows,
            "lookup_submit_total_us": submit_total_us,
            "lookup_submit_max_us": max(
                (int(row["lookup_us"]) for row in submit_rows),
                default=0,
            ),
            "lookup_status_count": len(status_rows),
            "lookup_status_rows": status_rows,
            "lookup_status_total_us": status_total_us,
            "lookup_status_max_us": max(
                (int(row["status_us"]) for row in status_rows),
                default=0,
            ),
            "cache_hit_lookup_count": len(cache_hit_lookup_rows),
            "cache_hit_lookup_rows": cache_hit_lookup_rows,
            "cache_hit_lookup_submit_total_us": cache_hit_lookup_total_us,
            "cache_hit_lookup_status_total_us": cache_hit_status_total_us,
            "cache_hit_lookup_total_us": (
                cache_hit_lookup_total_us + cache_hit_status_total_us
            ),
            "retrieve_completion_count": len(retrieve_rows),
            "retrieve_completion_rows": retrieve_rows,
            "retrieve_completion_query_total_us": retrieve_query_total_us,
            "retrieve_completion_result_total_us": retrieve_result_total_us,
            "retrieve_completion_total_us": (
                retrieve_query_total_us + retrieve_result_total_us
            ),
        }
    )
    if len(submit_rows) != 11:
        failures.append(
            "lookup timing artifact expected 11 lookup submit rows, "
            f"found {len(submit_rows)}"
        )
    if len(status_rows) != 11:
        failures.append(
            "lookup timing artifact expected 11 lookup status rows, "
            f"found {len(status_rows)}"
        )
    if len(cache_hit_lookup_rows) != 4:
        failures.append(
            "lookup timing artifact expected 4 query cache-hit lookup rows, "
            f"found {len(cache_hit_lookup_rows)}"
        )
    if len(retrieve_rows) != 4:
        failures.append(
            "lookup timing artifact expected 4 retrieve completion rows, "
            f"found {len(retrieve_rows)}"
        )
    if any(row["result"] != "True" for row in retrieve_rows):
        failures.append("lookup timing artifact recorded a failed retrieve")
    missing_status_ids = [
        str(row["request_id"])
        for row in submit_rows
        if str(row["request_id"]) not in status_by_id
    ]
    report["lookup_submit_ids_without_status"] = missing_status_ids
    if missing_status_ids:
        failures.append(
            "lookup timing artifact has lookup submits without status rows: "
            + ", ".join(missing_status_ids)
        )
    return failures, report


def _check_model(
    artifacts: ModelArtifacts,
    *,
    query_round_tolerance_s: float,
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    py_summary = _read_summary_artifact(artifacts.python_dir)
    native_summary = _read_summary_artifact(artifacts.native_dir)
    native_status = _read_json(artifacts.native_dir / "native_status.json")
    py_query = _read_query_csv(artifacts.python_dir / "query_round.csv")
    native_query = _read_query_csv(artifacts.native_dir / "query_round.csv")

    py_ttft = _summary_value(py_summary, "query_ttft_per_prompt")
    native_ttft = _summary_value(native_summary, "query_ttft_per_prompt")
    py_round = _summary_value(py_summary, "query_round_time_per_prompt")
    native_round = _summary_value(native_summary, "query_round_time_per_prompt")
    native_baseline_speedup = artifacts.native_baseline_ttft_s / native_ttft
    native_vs_python_speedup = py_ttft / native_ttft if native_ttft > 0 else None
    gates = {
        "gate_1_native_vs_old_native_3x": {
            "passed": native_baseline_speedup >= 3.0,
            "old_native_ttft_s": artifacts.native_baseline_ttft_s,
            "optimized_native_ttft_s": native_ttft,
            "speedup": native_baseline_speedup,
        },
        "gate_2_native_no_slower_than_python_ttft": {
            "passed": native_ttft <= py_ttft,
            "python_ttft_s": py_ttft,
            "native_ttft_s": native_ttft,
        },
        "gate_3_native_2x_faster_than_python_ttft": {
            "passed": native_ttft <= artifacts.native_hard_target_s
            and native_ttft <= py_ttft / 2.0,
            "python_ttft_s": py_ttft,
            "native_ttft_s": native_ttft,
            "native_hard_target_s": artifacts.native_hard_target_s,
            "speedup": native_vs_python_speedup,
        },
        "gate_4_query_round_parity": {
            "passed": native_round - py_round <= query_round_tolerance_s,
            "python_query_round_s": py_round,
            "native_query_round_s": native_round,
            "tolerance_s": query_round_tolerance_s,
        },
    }

    if abs(py_query.mean_ttft - py_ttft) > 1e-9:
        failures.append(
            f"{artifacts.name}: Python summary TTFT does not match query CSV"
        )
    if abs(native_query.mean_ttft - native_ttft) > 1e-9:
        failures.append(
            f"{artifacts.name}: native summary TTFT does not match query CSV"
        )
    if not gates["gate_1_native_vs_old_native_3x"]["passed"]:
        failures.append(
            f"{artifacts.name}: Gate 1 failed; native TTFT "
            f"{native_ttft:.6f}s is not 3x faster than old native "
            f"{artifacts.native_baseline_ttft_s:.6f}s"
        )
    if not gates["gate_2_native_no_slower_than_python_ttft"]["passed"]:
        failures.append(
            f"{artifacts.name}: Gate 2 failed; native TTFT "
            f"{native_ttft:.6f}s is slower than Python {py_ttft:.6f}s"
        )
    if native_ttft > artifacts.native_hard_target_s:
        failures.append(
            f"{artifacts.name}: Gate 3 failed; native TTFT "
            f"{native_ttft:.6f}s exceeds hard target "
            f"{artifacts.native_hard_target_s:.6f}s"
        )
    if native_ttft > py_ttft / 2.0:
        failures.append(
            f"{artifacts.name}: Gate 3 failed; native TTFT "
            f"{native_ttft:.6f}s is not 2x faster than Python "
            f"{py_ttft:.6f}s"
        )
    if native_round - py_round > query_round_tolerance_s:
        failures.append(
            f"{artifacts.name}: Gate 4 failed; native query-round "
            f"{native_round:.6f}s is slower than Python {py_round:.6f}s "
            "beyond tolerance"
        )
    if not filecmp.cmp(
        artifacts.python_dir / "responses.txt",
        artifacts.native_dir / "responses.txt",
        shallow=False,
    ):
        failures.append(f"{artifacts.name}: responses.txt differs")
    for label, query in (("Python", py_query), ("native", native_query)):
        if query.row_count == 0:
            failures.append(f"{artifacts.name}: {label} query CSV has no rows")
        if query.successful_count != query.row_count:
            failures.append(
                f"{artifacts.name}: {label} successful query count "
                f"{query.successful_count} != {query.row_count}"
            )
        if query.cache_hit_count != query.row_count:
            failures.append(
                f"{artifacts.name}: {label} cache-hit query count "
                f"{query.cache_hit_count} != {query.row_count}"
            )

    expected_counts = {
        "store_count": 7,
        "retrieve_count": 4,
        "lookup_count": 11,
        "unsupported_count": 0,
        "transfer_lock_failure_count": 0,
    }
    for key, expected in expected_counts.items():
        actual = _metric(native_status, key)
        if actual != expected:
            failures.append(
                f"{artifacts.name}: metrics.{key} {actual} != {expected}"
            )
    if _cache_value(native_status, "dram_bytes") > _cache_value(
        native_status,
        "dram_capacity_bytes",
    ):
        failures.append(f"{artifacts.name}: native DRAM usage exceeds capacity")
    if _cache_value(native_status, "disk_bytes") != 0:
        failures.append(f"{artifacts.name}: native cache spilled to disk")

    retrieve_count = _metric(native_status, "retrieve_count")
    retrieve_total_us = _metric(native_status, "cuda_transfer_retrieve_total_us")
    retrieve_max_us = _metric(native_status, "cuda_transfer_retrieve_max_us")
    retrieve_per_prompt_ms = retrieve_total_us / max(retrieve_count, 1) / 1000.0
    retrieve_per_prompt_s = retrieve_per_prompt_ms / 1000.0
    retrieve_max_s = retrieve_max_us / 1_000_000.0
    native_fastest_query_ttft_s = min(native_query.ttfts)
    zero_retrieve_mean_floor_s = max(0.0, native_ttft - retrieve_per_prompt_s)
    zero_retrieve_fastest_floor_s = max(
        0.0,
        native_fastest_query_ttft_s - retrieve_max_s,
    )
    print(
        f"{artifacts.name}: Python TTFT={py_ttft:.6f}s, "
        f"native TTFT={native_ttft:.6f}s, "
        f"target<={artifacts.native_hard_target_s:.6f}s, "
        f"native retrieve={retrieve_per_prompt_ms:.3f}ms/prompt"
    )
    report = {
        "model": artifacts.name,
        "python_dir": str(artifacts.python_dir),
        "native_dir": str(artifacts.native_dir),
        "python_ttft_s": py_ttft,
        "native_ttft_s": native_ttft,
        "native_hard_target_s": artifacts.native_hard_target_s,
        "native_baseline_ttft_s": artifacts.native_baseline_ttft_s,
        "native_vs_python_speedup": native_vs_python_speedup,
        "python_steady_state_ttft_s": py_query.steady_state_mean_ttft,
        "native_steady_state_ttft_s": native_query.steady_state_mean_ttft,
        "native_vs_python_steady_state_speedup": (
            py_query.steady_state_mean_ttft / native_query.steady_state_mean_ttft
            if native_query.steady_state_mean_ttft > 0
            else None
        ),
        "native_vs_old_native_speedup": native_baseline_speedup,
        "python_query_round_s": py_round,
        "native_query_round_s": native_round,
        "python_vllm_log": _vllm_log_report(artifacts.python_dir / "vllm.log"),
        "native_vllm_log": _vllm_log_report(artifacts.native_dir / "vllm.log"),
        "gates": gates,
        "query_rows": native_query.row_count,
        "successful_query_rows": native_query.successful_count,
        "cache_hit_query_rows": native_query.cache_hit_count,
        "native_retrieve_per_prompt_ms": retrieve_per_prompt_ms,
        "native_retrieve_max_s": retrieve_max_s,
        "native_fastest_query_ttft_s": native_fastest_query_ttft_s,
        "native_zero_retrieve_mean_ttft_floor_s": zero_retrieve_mean_floor_s,
        "native_zero_retrieve_fastest_ttft_floor_s": (
            zero_retrieve_fastest_floor_s
        ),
        "native_zero_retrieve_mean_gap_to_target_s": (
            zero_retrieve_mean_floor_s - artifacts.native_hard_target_s
        ),
        "native_zero_retrieve_fastest_gap_to_target_s": (
            zero_retrieve_fastest_floor_s - artifacts.native_hard_target_s
        ),
        "native_store_count": _metric(native_status, "store_count"),
        "native_retrieve_count": retrieve_count,
        "native_lookup_count": _metric(native_status, "lookup_count"),
        "native_lookup_result_fast_path_count": _optional_metric(
            native_status,
            "lookup_result_fast_path_count",
        ),
        "native_unsupported_count": _metric(native_status, "unsupported_count"),
        "native_transfer_lock_failure_count": _metric(
            native_status,
            "transfer_lock_failure_count",
        ),
        "native_cuda_gpu_hot_cache": _hot_cache_report(native_status),
        "native_transfer_lock_timing": _lock_timing_report(native_status),
        "native_request_type_latency": _request_type_latency_report(
            native_status
        ),
        "native_request_type_queue_wait": _request_type_queue_wait_report(
            native_status
        ),
        "failures": failures,
    }
    return failures, report


def _completion_status(
    failures: list[str],
    model_reports: list[dict[str, object]],
    vllm_only_reports: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    if not failures:
        return {
            "status": "complete",
            "decision_required": False,
            "reason": "all GOAL_OPTIMIZATION.md acceptance checks passed",
            "allowed_next_decisions": [],
        }

    max_retrieve_ms = max(
        float(report["native_retrieve_per_prompt_ms"]) for report in model_reports
    )
    max_zero_retrieve_floor_s = max(
        float(report["native_zero_retrieve_mean_ttft_floor_s"])
        for report in model_reports
    )
    completion: dict[str, object] = {
        "status": "blocked",
        "decision_required": True,
        "comparison_basis": (
            "native warm-cache LMCache TTFT versus "
            "Python warm-cache LMCache TTFT"
        ),
        "reason": (
            "native retrieve is already about 2ms per prompt in the saved "
            "artifacts, but end-to-end long_doc_qa TTFT still misses the hard "
            "2x targets; the remaining latency is outside native MP transfer "
            "under the current benchmark contract"
        ),
        "max_native_retrieve_per_prompt_ms": max_retrieve_ms,
        "max_native_zero_retrieve_mean_ttft_floor_s": max_zero_retrieve_floor_s,
        "allowed_next_decisions": [
            "keep_original_goal_and_identify_a_new_native_controlled_mechanism",
            "redefine_benchmark_contract_and_rerun_python_native_baselines",
            "change_target_metric_to_transfer_isolated_latency",
            "close_goal_as_blocked",
        ],
    }
    if vllm_only_reports:
        controls: dict[str, dict[str, float | bool]] = {}
        for model, report in vllm_only_reports.items():
            if report.get("present") is not True:
                continue
            ttft = report.get("query_ttft_s")
            fastest = report.get("fastest_prompt_ttft_s")
            target = report.get("native_hard_target_s")
            if not isinstance(ttft, int | float) or not isinstance(
                fastest, int | float
            ) or not isinstance(target, int | float):
                continue
            controls[model] = {
                "query_ttft_s": float(ttft),
                "fastest_prompt_ttft_s": float(fastest),
                "native_hard_target_s": float(target),
                "mean_above_target": float(ttft) > float(target),
                "fastest_above_target": float(fastest) > float(target),
            }
        if controls:
            completion["vllm_only_controls"] = controls
    return completion


def _aggregate_gates(model_reports: list[dict[str, object]]) -> dict[str, object]:
    gate_names = {
        gate_name
        for report in model_reports
        for gate_name in report["gates"].keys()
    }
    aggregate: dict[str, object] = {}
    for gate_name in sorted(gate_names):
        model_results = {
            str(report["model"]): bool(report["gates"][gate_name]["passed"])
            for report in model_reports
        }
        aggregate[gate_name] = {
            "passed": all(model_results.values()),
            "models": model_results,
        }
    return aggregate


def _check_text_artifact(
    path: Path,
    required_terms: tuple[str, ...],
) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, list(required_terms)
    text = path.read_text(encoding="utf-8", errors="replace")
    missing_terms = [term for term in required_terms if term not in text]
    return not missing_terms, missing_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify saved long_doc_qa artifacts against GOAL_OPTIMIZATION.md"
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
        "--query-round-tolerance-ms",
        type=float,
        default=2.0,
        help="Allowed native-vs-Python query-round overhead before failing.",
    )
    parser.add_argument(
        "--nsys-report",
        type=Path,
        default=Path("/tmp/lmcache-native-roundtrip-nsys.nsys-rep"),
        help="Expected Nsight Systems report for CUDA evidence.",
    )
    parser.add_argument(
        "--perf-stat-output",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-native-perf-stat.txt"),
        help=(
            "Expected perf stat output from the long_doc_qa benchmark window. "
            "It must contain OS scheduler/CPU counters."
        ),
    )
    parser.add_argument(
        "--allow-missing-perf-stat",
        action="store_true",
        help="Do not fail when --perf-stat-output is missing.",
    )
    parser.add_argument(
        "--nvidia-smi-topo-output",
        type=Path,
        default=Path("/tmp/lmcache-native-topology-nvidia-smi-topo.txt"),
        help="Expected nvidia-smi topo -m output for GPU locality evidence.",
    )
    parser.add_argument(
        "--numactl-hardware-output",
        type=Path,
        default=Path("/tmp/lmcache-native-topology-numactl-hardware.txt"),
        help="Expected numactl --hardware output for NUMA locality evidence.",
    )
    parser.add_argument(
        "--allow-missing-topology",
        action="store_true",
        help="Do not fail when topology evidence files are missing.",
    )
    parser.add_argument(
        "--qwen3-8b-adapter-timing-dir",
        type=Path,
        default=Path(
            "/tmp/lmcache-long-doc-qa-qwen3-8b-native-adaptertiming-1779048612"
        ),
        help=(
            "Expected Qwen3-8B diagnostic artifact with "
            "LMCACHE_MP_CONNECTOR_TIMING=1."
        ),
    )
    parser.add_argument(
        "--adapter-timing-max-total-us",
        type=int,
        default=1000,
        help=(
            "Maximum allowed total retrieve completion wait in the adapter "
            "timing diagnostic artifact."
        ),
    )
    parser.add_argument(
        "--allow-missing-adapter-timing",
        action="store_true",
        help="Do not fail when the adapter timing diagnostic artifact is missing.",
    )
    parser.add_argument(
        "--qwen3-8b-vllm-only-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-8b-vllmonly-1779049198"),
        help="Expected Qwen3-8B vLLM-only control artifact without LMCache.",
    )
    parser.add_argument(
        "--qwen3-14b-vllm-only-dir",
        type=Path,
        default=Path("/tmp/lmcache-long-doc-qa-qwen3-14b-vllmonly-1779052865"),
        help="Expected Qwen3-14B vLLM-only control artifact without LMCache.",
    )
    parser.add_argument(
        "--allow-missing-vllm-only",
        action="store_true",
        help="Do not fail when the vLLM-only control artifact is missing.",
    )
    parser.add_argument(
        "--qwen3-8b-lookup-timing-dir",
        type=Path,
        default=Path(
            "/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookuptiming-1779049672"
        ),
        help=(
            "Expected Qwen3-8B diagnostic artifact with lookup/status "
            "timing logs from LMCACHE_MP_CONNECTOR_TIMING=1."
        ),
    )
    parser.add_argument(
        "--allow-missing-lookup-timing",
        action="store_true",
        help="Do not fail when the lookup timing diagnostic artifact is missing.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write a machine-readable audit report.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the machine-readable audit report to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [
        ModelArtifacts(
            name="Qwen3-8B",
            python_dir=args.qwen3_8b_python_dir,
            native_dir=args.qwen3_8b_native_dir,
            native_baseline_ttft_s=0.520,
            native_hard_target_s=0.035237432,
        ),
        ModelArtifacts(
            name="Qwen3-14B",
            python_dir=args.qwen3_14b_python_dir,
            native_dir=args.qwen3_14b_native_dir,
            native_baseline_ttft_s=0.543,
            native_hard_target_s=0.037644863,
        ),
    ]

    failures: list[str] = []
    model_reports: list[dict[str, object]] = []
    for artifacts in models:
        model_failures, model_report = _check_model(
            artifacts,
            query_round_tolerance_s=args.query_round_tolerance_ms / 1000.0,
        )
        failures.extend(model_failures)
        model_reports.append(model_report)

    nsys_present = args.nsys_report.is_file() and args.nsys_report.stat().st_size > 0
    perf_present = False
    perf_missing_terms: list[str] = []
    if not nsys_present:
        failures.append(f"Nsight report is missing or empty: {args.nsys_report}")
    if args.perf_stat_output is not None and not args.allow_missing_perf_stat:
        if not args.perf_stat_output.is_file():
            failures.append(f"perf stat output is missing: {args.perf_stat_output}")
        else:
            perf_present = True
            perf_text = args.perf_stat_output.read_text(
                encoding="utf-8",
                errors="replace",
            )
            required_perf_terms = (
                "cycles",
                "context-switches",
                "cpu-migrations",
                "cache-misses",
            )
            perf_missing_terms = [
                term for term in required_perf_terms if term not in perf_text
            ]
            if perf_missing_terms:
                failures.append(
                    "perf stat output is missing required counters: "
                    + ", ".join(perf_missing_terms)
                )
    elif args.perf_stat_output is not None:
        perf_present = args.perf_stat_output.is_file()

    topology_reports: dict[str, dict[str, object]] = {}
    topology_specs = (
        (
            "nvidia_smi_topo",
            args.nvidia_smi_topo_output,
            ("GPU0", "CPU Affinity", "NUMA Affinity"),
        ),
        (
            "numactl_hardware",
            args.numactl_hardware_output,
            ("available:", "node 0 cpus", "node distances"),
        ),
    )
    for name, path, required_terms in topology_specs:
        present = path.is_file()
        ok, missing_terms = _check_text_artifact(path, required_terms)
        topology_reports[name] = {
            "path": str(path),
            "required": not args.allow_missing_topology,
            "present": present,
            "missing_required_terms": missing_terms,
        }
        if not args.allow_missing_topology and not ok:
            if not present:
                failures.append(f"topology evidence is missing: {path}")
            else:
                failures.append(
                    f"topology evidence {path} is missing required terms: "
                    + ", ".join(missing_terms)
                )

    adapter_timing_failures: list[str] = []
    adapter_timing_report: dict[str, object] = {
        "required": not args.allow_missing_adapter_timing,
        "dir": str(args.qwen3_8b_adapter_timing_dir),
    }
    adapter_timing_missing_allowed = (
        args.allow_missing_adapter_timing
        and not args.qwen3_8b_adapter_timing_dir.is_dir()
    )
    if adapter_timing_missing_allowed:
        adapter_timing_report["present"] = False
    else:
        adapter_timing_failures, adapter_timing_report = _connector_timing_report(
            args.qwen3_8b_adapter_timing_dir,
            reference_responses=args.qwen3_8b_python_dir / "responses.txt",
            max_total_us=args.adapter_timing_max_total_us,
        )
        adapter_timing_report["required"] = not args.allow_missing_adapter_timing
        if not args.allow_missing_adapter_timing:
            failures.extend(adapter_timing_failures)

    vllm_only_reports: dict[str, dict[str, object]] = {}
    for model, control_dir, reference_responses, hard_target_s in (
        (
                "Qwen3-8B",
            args.qwen3_8b_vllm_only_dir,
            args.qwen3_8b_python_dir / "responses.txt",
            0.035237432,
        ),
        (
                "Qwen3-14B",
            args.qwen3_14b_vllm_only_dir,
            args.qwen3_14b_python_dir / "responses.txt",
            0.037644863,
        ),
    ):
        vllm_only_report: dict[str, object] = {
            "required": not args.allow_missing_vllm_only,
            "dir": str(control_dir),
        }
        vllm_only_missing_allowed = (
            args.allow_missing_vllm_only and not control_dir.is_dir()
        )
        if vllm_only_missing_allowed:
            vllm_only_report["present"] = False
        else:
            vllm_only_failures, vllm_only_report = _vllm_only_report(
                control_dir,
                reference_responses=reference_responses,
                native_hard_target_s=hard_target_s,
            )
            vllm_only_report["required"] = not args.allow_missing_vllm_only
            if not args.allow_missing_vllm_only:
                failures.extend(f"{model}: {failure}" for failure in vllm_only_failures)
        vllm_only_reports[model] = vllm_only_report

    lookup_timing_failures: list[str] = []
    lookup_timing_report: dict[str, object] = {
        "required": not args.allow_missing_lookup_timing,
        "dir": str(args.qwen3_8b_lookup_timing_dir),
    }
    lookup_timing_missing_allowed = (
        args.allow_missing_lookup_timing
        and not args.qwen3_8b_lookup_timing_dir.is_dir()
    )
    if lookup_timing_missing_allowed:
        lookup_timing_report["present"] = False
    else:
        lookup_timing_failures, lookup_timing_report = _lookup_timing_report(
            args.qwen3_8b_lookup_timing_dir,
            reference_responses=args.qwen3_8b_python_dir / "responses.txt",
        )
        lookup_timing_report["required"] = not args.allow_missing_lookup_timing
        if not args.allow_missing_lookup_timing:
            failures.extend(lookup_timing_failures)

    report = {
        "passed": not failures,
        "failures": failures,
        "completion": _completion_status(
            failures,
            model_reports,
            vllm_only_reports,
        ),
        "gates": _aggregate_gates(model_reports),
        "models": model_reports,
        "evidence": {
            "nsight_report": str(args.nsys_report),
            "nsight_report_present": nsys_present,
            "perf_stat_output": str(args.perf_stat_output)
            if args.perf_stat_output is not None
            else None,
            "perf_stat_required": not args.allow_missing_perf_stat,
            "perf_stat_present": perf_present,
            "perf_stat_missing_required_terms": perf_missing_terms,
            "topology": topology_reports,
            "adapter_timing": adapter_timing_report,
            "vllm_only_control": vllm_only_reports["Qwen3-8B"],
            "vllm_only_controls": vllm_only_reports,
            "lookup_timing": lookup_timing_report,
        },
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if failures:
        print("\nGOAL_OPTIMIZATION.md audit: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nGOAL_OPTIMIZATION.md audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
