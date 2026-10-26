from pathlib import Path

script = r'''#!/usr/bin/env bash
# Run long_doc_qa workload test against LMCache and baseline inference-engine servers.
# Compares performance between LMCache-enabled and baseline engine.
# Adapted from the old Docker-based run-long-doc-qa.sh.
set -e
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

# Configuration
ENGINE_PORT="${ENGINE_PORT:-8000}"
ENGINE_BASELINE_PORT="${ENGINE_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_DIR="${LMCACHE_DIR:-$REPO_ROOT}"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-30}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-5}"

# Negative values mean LMCache must be faster than baseline by at least that %.
MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:--60}"
MAX_ROUND_TIME_SLOWDOWN_PCT="${MAX_ROUND_TIME_SLOWDOWN_PCT:--15}"

# Resolve paths before benchmark commands change their working directory.
LMCACHE_DIR="$(cd "$LMCACHE_DIR" 2>/dev/null && pwd || echo "$LMCACHE_DIR")"
mkdir -p "$RESULTS_DIR/long_doc_qa"
LONG_DOC_QA_DIR="$(cd "$RESULTS_DIR/long_doc_qa" && pwd)"
BASELINE_DIR="$LONG_DOC_QA_DIR/baseline"
LMCACHE_OUT_DIR="$LONG_DOC_QA_DIR/lmcache"
mkdir -p "$BASELINE_DIR" "$LMCACHE_OUT_DIR"

DIAGNOSTIC_SAVED=0
CURRENT_VERDICT="UNKNOWN"

save_diagnostic_summary() {
    local verdict="$1"
    CURRENT_VERDICT="$verdict"

    local summary_file="$LONG_DOC_QA_DIR/diagnostic_summary.json"
    local attempt="${BUILDKITE_RETRY_COUNT:-0}"
    local attempt_dir="$LONG_DOC_QA_DIR/attempt_${attempt}"

    mkdir -p "$LONG_DOC_QA_DIR"

    local checkout_sha
    checkout_sha="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"

    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date +"%Y-%m-%dT%H:%M:%SZ")"

    DIAG_VERDICT="$verdict" \
    DIAG_CHECKOUT_SHA="$checkout_sha" \
    DIAG_TIMESTAMP="$timestamp" \
    DIAG_ATTEMPT="$attempt" \
    DIAG_SUMMARY_FILE="$summary_file" \
    LONG_DOC_QA_DIR="$LONG_DOC_QA_DIR" \
    MODEL="$MODEL" \
    DOCUMENT_LENGTH="$DOCUMENT_LENGTH" \
    NUM_DOCUMENTS="$NUM_DOCUMENTS" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    REPEAT_COUNT="$REPEAT_COUNT" \
    REPEAT_MODE="$REPEAT_MODE" \
    SHUFFLE_SEED="$SHUFFLE_SEED" \
    MAX_INFLIGHT_REQUESTS="$MAX_INFLIGHT_REQUESTS" \
    MAX_TTFT_SLOWDOWN_PCT="$MAX_TTFT_SLOWDOWN_PCT" \
    MAX_ROUND_TIME_SLOWDOWN_PCT="$MAX_ROUND_TIME_SLOWDOWN_PCT" \
    ENGINE_PORT="$ENGINE_PORT" \
    ENGINE_BASELINE_PORT="$ENGINE_BASELINE_PORT" \
    ENGINE_NAME="${ENGINE_NAME:-vllm}" \
    BUILD_ID="$BUILD_ID" \
    "${PYTHON_EXEC}" - <<'PY'
import csv
import json
import math
import os
import sys


def calc_stats(values):
    if not values:
        return None

    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)

    def percentile(p):
        if n == 1:
            return values[0]
        idx = (n - 1) * (p / 100.0)
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        weight = idx - low
        return values[low] * (1.0 - weight) + values[high] * weight

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": values[0],
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": values[-1],
    }


def normalize_key(key):
    """Normalize CSV headers so small naming differences do not break parsing."""
    return (
        str(key or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_success(value, column_name):
    """Interpret common success flags and HTTP-style status columns."""
    value = str(value or "").strip().lower()

    if value in {"true", "1", "yes", "y", "success", "successful", "ok"}:
        return True

    if value in {"false", "0", "no", "n", "failed", "failure", "error", ""}:
        return False

    # Some CSVs use a status column containing an HTTP status code.
    if normalize_key(column_name) in {"status", "status_code", "http_status"}:
        try:
            code = int(value)
            return 200 <= code < 300
        except ValueError:
            return value in {"completed", "done"}

    return False


def parse_round_csv(csv_path):
    """
    Read per-request benchmark CSV data.

    Important: successful_samples counts successful requests, independently
    of whether those requests have a usable TTFT value. ttft_stats summarizes
    only finite, non-negative TTFT values from successful requests.
    """
    if not os.path.isfile(csv_path):
        return {
            "present": False,
            "total_samples": 0,
            "successful_samples": 0,
            "ttft_stats": None,
        }

    success_aliases = {
        "success", "successful", "is_success", "is_successful",
        "request_success", "status", "status_code", "http_status",
    }
    ttft_aliases = {
        "ttft", "ttft_s", "ttft_seconds", "time_to_first_token",
        "time_to_first_token_s", "time_to_first_token_seconds",
    }

    total = 0
    successful_count = 0
    successful_ttfts = []

    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {
                    "present": True,
                    "total_samples": 0,
                    "successful_samples": 0,
                    "ttft_stats": None,
                }

            original_headers = reader.fieldnames
            normalized_headers = {
                normalize_key(header): header for header in original_headers
            }

            success_header = next(
                (normalized_headers[key] for key in success_aliases
                 if key in normalized_headers),
                None,
            )
            ttft_header = next(
                (normalized_headers[key] for key in ttft_aliases
                 if key in normalized_headers),
                None,
            )

            for row in reader:
                if not row or not any(
                    str(value or "").strip() for value in row.values()
                ):
                    continue

                total += 1
                success = False
                if success_header is not None:
                    success = parse_success(row.get(success_header), success_header)

                if not success:
                    continue

                successful_count += 1

                if ttft_header is None:
                    continue

                raw_ttft = str(row.get(ttft_header) or "").strip()
                if not raw_ttft:
                    continue

                try:
                    ttft = float(raw_ttft)
                except (TypeError, ValueError):
                    continue

                if math.isfinite(ttft) and ttft >= 0:
                    successful_ttfts.append(ttft)

    except (OSError, csv.Error) as exc:
        print(f"Warning: could not parse benchmark CSV {csv_path}: {exc}", file=sys.stderr)
        return {
            "present": True,
            "total_samples": total,
            "successful_samples": successful_count,
            "ttft_stats": calc_stats(successful_ttfts),
        }

    if total and success_header is None:
        print(
            f"Warning: no recognized success column in {csv_path}; "
            f"available columns: {original_headers}",
            file=sys.stderr,
        )

    if successful_count and ttft_header is None:
        print(
            f"Warning: no recognized TTFT column in {csv_path}; "
            f"available columns: {original_headers}",
            file=sys.stderr,
        )

    return {
        "present": True,
        "total_samples": total,
        "successful_samples": successful_count,
        "ttft_stats": calc_stats(successful_ttfts),
    }


def load_json(path):
    """Load JSON, accepting either a normal JSON document or JSONL output."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            for line in reversed(content.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (OSError, UnicodeError):
        pass
    return None


def safe_int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def safe_float_env(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


base_dir = os.environ.get("LONG_DOC_QA_DIR", "")
baseline_dir = os.path.join(base_dir, "baseline")
lmcache_dir = os.path.join(base_dir, "lmcache")

baseline_json = (
    load_json(os.path.join(baseline_dir, "baseline_result.json"))
    or load_json(os.path.join(base_dir, "baseline_result.json"))
)
lmcache_json = (
    load_json(os.path.join(lmcache_dir, "lmcache_result.json"))
    or load_json(os.path.join(base_dir, "lmcache_result.json"))
)

baseline_phase = {
    "phase": "baseline",
    "port": os.environ.get("ENGINE_BASELINE_PORT", "9000"),
    "warmup_round": parse_round_csv(os.path.join(baseline_dir, "warmup_round.csv")),
    "query_round": parse_round_csv(os.path.join(baseline_dir, "query_round.csv")),
    "aggregate": baseline_json,
}

lmcache_phase = {
    "phase": "lmcache",
    "port": os.environ.get("ENGINE_PORT", "8000"),
    "warmup_round": parse_round_csv(os.path.join(lmcache_dir, "warmup_round.csv")),
    "query_round": parse_round_csv(os.path.join(lmcache_dir, "query_round.csv")),
    "aggregate": lmcache_json,
}

max_ttft_slowdown = safe_float_env("MAX_TTFT_SLOWDOWN_PCT", -60)
max_round_time_slowdown = safe_float_env("MAX_ROUND_TIME_SLOWDOWN_PCT", -15)


def compare_metric(name, lmcache_value, baseline_value, threshold):
    try:
        lmcache_value = float(lmcache_value)
        baseline_value = float(baseline_value)
    except (TypeError, ValueError):
        return {
            "name": name,
            "baseline": None,
            "lmcache": None,
            "diff_percent": None,
            "threshold_percent": threshold,
            "passed": False,
        }

    if not math.isfinite(lmcache_value) or not math.isfinite(baseline_value):
        valid = False
    else:
        valid = True

    if not valid or baseline_value <= 0:
        return {
            "name": name,
            "baseline": baseline_value if valid else None,
            "lmcache": lmcache_value if valid else None,
            "diff_percent": None,
            "threshold_percent": threshold,
            "passed": False,
        }

    diff_pct = ((lmcache_value - baseline_value) / baseline_value) * 100.0
    return {
        "name": name,
        "baseline": round(baseline_value, 4),
        "lmcache": round(lmcache_value, 4),
        "diff_percent": round(diff_pct, 2),
        "threshold_percent": threshold,
        "passed": diff_pct <= threshold,
    }


b_agg = baseline_json or {}
l_agg = lmcache_json or {}

comparison = {
    "query_ttft_per_prompt": compare_metric(
        "query_ttft_per_prompt",
        l_agg.get("query_ttft_per_prompt"),
        b_agg.get("query_ttft_per_prompt"),
        max_ttft_slowdown,
    ),
    "query_round_time_per_prompt": compare_metric(
        "query_round_time_per_prompt",
        l_agg.get("query_round_time_per_prompt"),
        b_agg.get("query_round_time_per_prompt"),
        max_round_time_slowdown,
    ),
    "warmup_round_time_per_prompt": compare_metric(
        "warmup_round_time_per_prompt",
        l_agg.get("warmup_round_time_per_prompt"),
        b_agg.get("warmup_round_time_per_prompt"),
        0.0,
    ),
}

summary = {
    "verdict": os.environ.get("DIAG_VERDICT", "UNKNOWN"),
    "build_id": os.environ.get("BUILD_ID", "local"),
    "attempt": safe_int_env("DIAG_ATTEMPT", 0),
    "checkout_sha": os.environ.get("DIAG_CHECKOUT_SHA", "unknown"),
    "timestamp": os.environ.get("DIAG_TIMESTAMP", ""),
    "workload_settings": {
        "engine": os.environ.get("ENGINE_NAME", "vllm"),
        "model": os.environ.get("MODEL", ""),
        "document_length": safe_int_env("DOCUMENT_LENGTH", 10000),
        "num_documents": safe_int_env("NUM_DOCUMENTS", 30),
        "output_len": safe_int_env("OUTPUT_LEN", 200),
        "repeat_count": safe_int_env("REPEAT_COUNT", 2),
        "repeat_mode": os.environ.get("REPEAT_MODE", "tile"),
        "shuffle_seed": safe_int_env("SHUFFLE_SEED", 0),
        "max_inflight_requests": safe_int_env("MAX_INFLIGHT_REQUESTS", 5),
        "max_ttft_slowdown_pct": max_ttft_slowdown,
        "max_round_time_slowdown_pct": max_round_time_slowdown,
    },
    "phases": {
        "baseline": baseline_phase,
        "lmcache": lmcache_phase,
    },
    "comparison": comparison,
}

out_file = os.environ.get("DIAG_SUMMARY_FILE", "")
if out_file:
    try:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
            f.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to write diagnostic summary to {out_file}: {exc}", file=sys.stderr)

print()
print("============================================")
print("=== Long Doc QA Diagnostic Summary ===")
print("============================================")
print(f"Verdict: {summary['verdict']}")
print(f"Attempt: {summary['attempt']}")
print(f"Timestamp: {summary['timestamp']}")
print(f"Checkout SHA: {summary['checkout_sha']}")
print()
print("Phase Highlights:")

for phase_name in ("baseline", "lmcache"):
    phase = summary["phases"][phase_name]
    aggregate = phase["aggregate"] or {}
    print(f"  [{phase_name.upper()}] (port {phase['port']})")
    print(
        "    Aggregates: "
        f"query_ttft={aggregate.get('query_ttft_per_prompt')}, "
        f"query_round_time={aggregate.get('query_round_time_per_prompt')}, "
        f"warmup_round_time={aggregate.get('warmup_round_time_per_prompt')}"
    )

    for round_name in ("warmup_round", "query_round"):
        info = phase[round_name]
        if not info["present"]:
            print(f"    {round_name} CSV: not found")
            continue

        stats = info["ttft_stats"]
        if stats:
            print(
                f"    {round_name} TTFT (N={stats['count']}): "
                f"mean={stats['mean']:.4f}s, p50={stats['p50']:.4f}s, "
                f"p95={stats['p95']:.4f}s, p99={stats['p99']:.4f}s, "
                f"min={stats['min']:.4f}s, max={stats['max']:.4f}s"
            )
        else:
            print(
                f"    {round_name} TTFT: 0 valid TTFT values "
                f"(successful={info['successful_samples']}, "
                f"total={info['total_samples']})"
            )
    print()

print("Comparisons:")
for key in ("query_ttft_per_prompt", "query_round_time_per_prompt"):
    item = summary["comparison"][key]
    if item["baseline"] is not None and item["lmcache"] is not None:
        diff = item["diff_percent"]
        if diff < 0:
            diff_text = f"{abs(diff):.1f}% faster"
        elif diff > 0:
            diff_text = f"{diff:.1f}% slower"
        else:
            diff_text = "no change"

        threshold = item["threshold_percent"]
        threshold_text = (
            f"need >= {abs(threshold):.0f}% faster"
            if threshold < 0
            else f"max {threshold}% slower"
        )
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"  {key}: {item['lmcache']:.4f}s vs baseline "
            f"{item['baseline']:.4f}s ({diff_text}, {threshold_text}) -- {status}"
        )
    else:
        print(f"  {key}: missing or invalid data -- FAIL")

print("============================================")
print()
PY

    DIAGNOSTIC_SAVED=1

    # Preserve attempt-specific copies only on retries.
    if [[ "${BUILDKITE_RETRY_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
        mkdir -p "$attempt_dir"
        if [ -f "$summary_file" ]; then
            cp "$summary_file" "$attempt_dir/diagnostic_summary.json" 2>/dev/null || true
        fi
        if [ -d "$LONG_DOC_QA_DIR/baseline" ]; then
            mkdir -p "$attempt_dir/baseline"
            cp -r "$LONG_DOC_QA_DIR/baseline/." "$attempt_dir/baseline/" 2>/dev/null || true
        fi
        if [ -d "$LONG_DOC_QA_DIR/lmcache" ]; then
            mkdir -p "$attempt_dir/lmcache"
            cp -r "$LONG_DOC_QA_DIR/lmcache/." "$attempt_dir/lmcache/" 2>/dev/null || true
        fi
    fi

    # Keep diagnostic logs available for Buildkite artifact collection.
    local diag_log="/tmp/build_${BUILD_ID}_attempt_${attempt}_long_doc_qa_diagnostic.log"
    cp "$summary_file" "$diag_log" 2>/dev/null || true
    cp "$summary_file" "/tmp/build_${BUILD_ID}_long_doc_qa_diagnostic.log" 2>/dev/null || true

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "$summary_file" 2>/dev/null || true
        buildkite-agent artifact upload "$LONG_DOC_QA_DIR/baseline/*" 2>/dev/null || true
        buildkite-agent artifact upload "$LONG_DOC_QA_DIR/lmcache/*" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "${CURRENT_VERDICT:-}" != "FAIL" ]; then
        # Avoid masking the original benchmark failure if diagnostics also fail.
        save_diagnostic_summary "FAIL" || true
    fi
}
trap on_exit EXIT

echo "=== Long Doc QA Test ==="
echo "Model: $MODEL"
echo "Engine Port (with LMCache): $ENGINE_PORT"
echo "${ENGINE_NAME:-vllm} Baseline Port (without LMCache): $ENGINE_BASELINE_PORT"
echo "Document length: $DOCUMENT_LENGTH"
echo "Number of documents: $NUM_DOCUMENTS"
echo "Output length: $OUTPUT_LEN"
echo "Results dir: $LONG_DOC_QA_DIR"
echo
echo "Performance thresholds (relative to baseline, negative = must be faster):"
echo "  Max TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}%"
echo "  Max round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}%"
echo

run_long_doc_qa() {
    local port="$1"
    local result_file="$2"
    local description="$3"
    local phase_dir="$LONG_DOC_QA_DIR/$description"

    echo "=== Running long_doc_qa ($description) ==="
    mkdir -p "$phase_dir"

    local output_file="$phase_dir/${description}_output.txt"
    local phase_result_file="$phase_dir/${description}_result.json"

    local -a benchmark_command=(
        "${PYTHON_EXEC}" "$LMCACHE_DIR/benchmarks/long_doc_qa/long_doc_qa.py"
        --port "$port"
        --model "$MODEL"
        --document-length "$DOCUMENT_LENGTH"
        --num-documents "$NUM_DOCUMENTS"
        --output-len "$OUTPUT_LEN"
        --repeat-count "$REPEAT_COUNT"
        --repeat-mode "$REPEAT_MODE"
        --shuffle-seed "$SHUFFLE_SEED"
        --max-inflight-requests "$MAX_INFLIGHT_REQUESTS"
        --output "$output_file"
        --json-output
    )

    # Run from phase_dir so CSV files from each phase stay separate.
    (
        cd "$phase_dir"
        "${benchmark_command[@]}" 2>> "$output_file" | tee "$phase_result_file"
    )

    "${PYTHON_EXEC}" "$COMMON_WORKLOAD_DIR/long_doc_qa_diagnostics.py" \
        --phase "$description" \
        --engine "${INFERENCE_ENGINE:-${ENGINE_NAME:-vllm}}" \
        --results-dir "$phase_dir" \
        --command "${benchmark_command[@]}" | tee "$phase_dir/diagnostics.txt"

    cp -f "$phase_result_file" "$LONG_DOC_QA_DIR/${description}_result.json" 2>/dev/null || true
    cp -f "$output_file" "$LONG_DOC_QA_DIR/${description}_output.txt" 2>/dev/null || true

    if [[ -n "$result_file" && "$result_file" != "$phase_result_file" ]]; then
        cp -f "$phase_result_file" "$result_file" 2>/dev/null || true
    fi

    echo "$description benchmark completed"
    echo
}

extract_json_field() {
    local json_file="$1"
    local field="$2"

    "${PYTHON_EXEC}" - "$json_file" "$field" <<'PY'
import json
import sys

path, field = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        for line in reversed(content.splitlines()):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    value = data.get(field) if isinstance(data, dict) else None
    if value is None:
        print("null")
    elif isinstance(value, (int, float)):
        print(value)
    else:
        print("null")
except (OSError, UnicodeError, TypeError, ValueError):
    print("null")
PY
}

get_result_paths() {
    # Set globals for callers.
    LMCache_RESULT="$LONG_DOC_QA_DIR/lmcache/lmcache_result.json"
    BASELINE_RESULT="$LONG_DOC_QA_DIR/baseline/baseline_result.json"

    if [ ! -f "$LMCache_RESULT" ] && [ -f "$LONG_DOC_QA_DIR/lmcache_result.json" ]; then
        LMCache_RESULT="$LONG_DOC_QA_DIR/lmcache_result.json"
    fi
    if [ ! -f "$BASELINE_RESULT" ] && [ -f "$LONG_DOC_QA_DIR/baseline_result.json" ]; then
        BASELINE_RESULT="$LONG_DOC_QA_DIR/baseline_result.json"
    fi
}

compare_results() {
    get_result_paths

    echo "=== Comparing benchmark results ==="
    if [ ! -f "$LMCache_RESULT" ] || [ ! -f "$BASELINE_RESULT" ]; then
        echo "Result files not found"
        return 1
    fi

    local lmcache_query_ttft lmcache_query_round_time lmcache_warmup_round_time
    local baseline_query_ttft baseline_query_round_time baseline_warmup_round_time

    lmcache_query_ttft="$(extract_json_field "$LMCache_RESULT" "query_ttft_per_prompt")"
    lmcache_query_round_time="$(extract_json_field "$LMCache_RESULT" "query_round_time_per_prompt")"
    lmcache_warmup_round_time="$(extract_json_field "$LMCache_RESULT" "warmup_round_time_per_prompt")"

    baseline_query_ttft="$(extract_json_field "$BASELINE_RESULT" "query_ttft_per_prompt")"
    baseline_query_round_time="$(extract_json_field "$BASELINE_RESULT" "query_round_time_per_prompt")"
    baseline_warmup_round_time="$(extract_json_field "$BASELINE_RESULT" "warmup_round_time_per_prompt")"

    "${PYTHON_EXEC}" - \
        "$lmcache_query_ttft" "$baseline_query_ttft" \
        "$lmcache_query_round_time" "$baseline_query_round_time" \
        "$lmcache_warmup_round_time" "$baseline_warmup_round_time" <<'PY'
import math
import sys

def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def format_comparison(name, lmcache_value, baseline_value):
    lmcache = safe_float(lmcache_value)
    baseline = safe_float(baseline_value)
    if lmcache is None or baseline is None:
        return f"{name}: Unable to compare (invalid values)"
    if baseline <= 0:
        diff = "N/A"
    else:
        pct = ((lmcache - baseline) / baseline) * 100
        label = "faster" if pct < 0 else "slower" if pct > 0 else "unchanged"
        diff = f"{abs(pct):.2f}% {label}"
    return (
        f"{name}:\n"
        f"  Baseline:  {baseline:.4f}s\n"
        f"  LMCache:   {lmcache:.4f}s\n"
        f"  Diff:      {diff}"
    )

l_ttft, b_ttft, l_round, b_round, l_warm, b_warm = sys.argv[1:]
print("============================================")
print("=== Performance Summary ===")
print("============================================")
print(format_comparison("query_ttft_per_prompt", l_ttft, b_ttft))
print()
print(format_comparison("query_round_time_per_prompt", l_round, b_round))
print()
print(format_comparison("warmup_round_time_per_prompt", l_warm, b_warm))
print()
print(f"{'Metric':<35} {'Baseline':>12} {'LMCache':>12} {'Diff':>10}")
print("-" * 70)
for name, baseline_value, lmcache_value in (
    ("query_ttft_per_prompt", b_ttft, l_ttft),
    ("query_round_time_per_prompt", b_round, l_round),
    ("warmup_round_time_per_prompt", b_warm, l_warm),
):
    baseline = safe_float(baseline_value)
    lmcache = safe_float(lmcache_value)
    if baseline is None or lmcache is None:
        print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
    else:
        pct = ((lmcache - baseline) / baseline) * 100 if baseline > 0 else float("nan")
        pct_text = f"{pct:+.1f}%" if math.isfinite(pct) else "N/A"
        print(f"{name:<35} {baseline:>12.4f} {lmcache:>12.4f} {pct_text:>10}")
print()
PY
}

verify_thresholds() {
    get_result_paths

    echo "=== Verifying LMCache performance vs baseline ==="
    echo "Max allowed TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}%"
    echo "Max allowed query round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}%"
    echo

    if [ ! -f "$LMCache_RESULT" ] || [ ! -f "$BASELINE_RESULT" ]; then
        echo "Cannot verify thresholds: result file(s) missing"
        save_diagnostic_summary "FAIL"
        return 1
    fi

    local lmcache_query_ttft lmcache_query_round_time
    local baseline_query_ttft baseline_query_round_time

    lmcache_query_ttft="$(extract_json_field "$LMCache_RESULT" "query_ttft_per_prompt")"
    lmcache_query_round_time="$(extract_json_field "$LMCache_RESULT" "query_round_time_per_prompt")"
    baseline_query_ttft="$(extract_json_field "$BASELINE_RESULT" "query_ttft_per_prompt")"
    baseline_query_round_time="$(extract_json_field "$BASELINE_RESULT" "query_round_time_per_prompt")"

    local verify_status=0
    "${PYTHON_EXEC}" - \
        "$lmcache_query_ttft" "$baseline_query_ttft" \
        "$lmcache_query_round_time" "$baseline_query_round_time" \
        "$MAX_TTFT_SLOWDOWN_PCT" "$MAX_ROUND_TIME_SLOWDOWN_PCT" <<'PY' || verify_status=$?
import math
import sys

def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def check_metric(name, lmcache_value, baseline_value, threshold):
    lmcache = safe_float(lmcache_value)
    baseline = safe_float(baseline_value)
    if lmcache is None or baseline is None or baseline <= 0:
        print(
            f"{name}: unable to compare "
            f"(lmcache={lmcache_value}, baseline={baseline_value}) -- FAIL"
        )
        return False

    pct = ((lmcache - baseline) / baseline) * 100
    if pct < 0:
        label = f"{abs(pct):.1f}% faster"
    elif pct > 0:
        label = f"{pct:.1f}% slower"
    else:
        label = "no change"

    threshold_label = (
        f"need >= {abs(threshold):.0f}% faster"
        if threshold < 0
        else f"max {threshold}% slower"
    )
    passed = pct <= threshold
    status = "PASS" if passed else "FAIL"
    print(
        f"{name}: {lmcache:.4f}s vs baseline {baseline:.4f}s "
        f"({label}, {threshold_label}) -- {status}"
    )
    return passed

l_ttft, b_ttft, l_round, b_round, ttft_threshold, round_threshold = sys.argv[1:]
try:
    ttft_threshold = float(ttft_threshold)
    round_threshold = float(round_threshold)
except ValueError:
    print("Invalid threshold configuration -- FAIL")
    sys.exit(1)

passed_ttft = check_metric(
    "query_ttft_per_prompt", l_ttft, b_ttft, ttft_threshold
)
passed_round = check_metric(
    "query_round_time_per_prompt", l_round, b_round, round_threshold
)

if passed_ttft and passed_round:
    print("\nAll thresholds passed")
    sys.exit(0)

print("\nThreshold verification FAILED")
sys.exit(1)
PY

    if [ "$verify_status" -eq 0 ]; then
        save_diagnostic_summary "PASS"
        return 0
    fi

    save_diagnostic_summary "FAIL"
    return 1
}

# Run benchmark against baseline.
echo "============================================"
echo "=== Benchmark: Baseline ${ENGINE_NAME:-vllm} (without LMCache) ==="
echo "============================================"
run_long_doc_qa "$ENGINE_BASELINE_PORT" "$LONG_DOC_QA_DIR/baseline_result.json" "baseline"

# Run benchmark against the inference engine with LMCache.
echo "============================================"
echo "=== Benchmark: ${ENGINE_NAME:-vllm} with LMCache ==="
echo "============================================"
run_long_doc_qa "$ENGINE_PORT" "$LONG_DOC_QA_DIR/lmcache_result.json" "lmcache"

echo "============================================"
echo "=== Comparing Results ==="
echo "============================================"
if ! compare_results; then
    echo "Comparison failed"
    save_diagnostic_summary "FAIL"
    exit 1
fi

echo "============================================"
echo "=== Verifying Performance Thresholds ==="
echo "============================================"
if ! verify_thresholds; then
    echo "Threshold verification failed"
    exit 1
fi

echo "============================================"
echo "=== Long Doc QA test completed ==="
echo "============================================"
'''

path = Path("/mnt/data/run-long-doc-qa-fixed.sh")
path.write_text(script, encoding="utf-8")
path.chmod(0o755)

# Basic structural validation: balanced here-doc terminators and shell syntax if bash is available.
import subprocess, os
result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
print(f"Created: {path}")
print(f"Size: {path.stat().st_size} bytes")
print(f"bash -n validation: {'PASS' if result.returncode == 0 else 'FAIL'}")
if result.returncode != 0:
    print(result.stderr)
