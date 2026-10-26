#!/usr/bin/env bash
# Run long_doc_qa against baseline and LMCache-enabled inference servers.
set -Ee
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

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

MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:--60}"
MAX_ROUND_TIME_SLOWDOWN_PCT="${MAX_ROUND_TIME_SLOWDOWN_PCT:--15}"

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

    local checkout_sha timestamp
    checkout_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
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
    variance = (
        sum((x - mean) ** 2 for x in values) / n
        if n > 1 else 0.0
    )

    def percentile(p):
        if n == 1:
            return values[0]
        idx = (n - 1) * p / 100.0
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        weight = idx - low
        return values[low] * (1 - weight) + values[high] * weight

    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": values[0],
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": values[-1],
    }


def normalize_key(value):
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_successful(value, column_name):
    value = str(value or "").strip().lower()

    if value in {
        "true", "1", "yes", "y", "success", "successful", "ok"
    }:
        return True

    if value in {
        "false", "0", "no", "n", "failed", "failure", "error", ""
    }:
        return False

    if normalize_key(column_name) in {
        "status", "status_code", "http_status"
    }:
        try:
            code = int(value)
            return 200 <= code < 300
        except ValueError:
            return value in {"completed", "done"}

    return False


def parse_round_csv(csv_path):
    empty = {
        "present": False,
        "total_samples": 0,
        "successful_samples": 0,
        "ttft_stats": None,
    }

    if not os.path.isfile(csv_path):
        return empty

    success_aliases = {
        "success", "successful", "is_success", "is_successful",
        "request_success", "status", "status_code", "http_status",
    }
    ttft_aliases = {
        "ttft", "ttft_s", "ttft_seconds",
        "time_to_first_token", "time_to_first_token_s",
        "time_to_first_token_seconds",
    }

    total = 0
    successful_count = 0
    successful_ttfts = []

    with open(
        csv_path, "r", encoding="utf-8-sig",
        errors="replace", newline=""
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return {**empty, "present": True}

        headers = {
            normalize_key(header): header
            for header in reader.fieldnames
        }

        success_header = next(
            (headers[key] for key in success_aliases if key in headers),
            None,
        )
        ttft_header = next(
            (headers[key] for key in ttft_aliases if key in headers),
            None,
        )

        for row in reader:
            if not row or not any(
                str(value or "").strip() for value in row.values()
            ):
                continue

            total += 1

            if success_header is None:
                continue

            if not is_successful(
                row.get(success_header), success_header
            ):
                continue

            # Count successful requests even when TTFT is missing/invalid.
            successful_count += 1

            if ttft_header is None:
                continue

            try:
                ttft = float(str(row.get(ttft_header) or "").strip())
            except (TypeError, ValueError):
                continue

            if math.isfinite(ttft) and ttft >= 0:
                successful_ttfts.append(ttft)

        fieldnames = reader.fieldnames

    if total and success_header is None:
        print(
            f"WARNING: No recognized success column in {csv_path}; "
            f"headers={fieldnames}",
            file=sys.stderr,
        )

    return {
        "present": True,
        "total_samples": total,
        "successful_samples": successful_count,
        "ttft_stats": calc_stats(successful_ttfts),
    }


def load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, TypeError):
        return None


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
    "warmup_round": parse_round_csv(
        os.path.join(baseline_dir, "warmup_round.csv")
    ),
    "query_round": parse_round_csv(
        os.path.join(baseline_dir, "query_round.csv")
    ),
    "aggregate": baseline_json,
}

lmcache_phase = {
    "phase": "lmcache",
    "port": os.environ.get("ENGINE_PORT", "8000"),
    "warmup_round": parse_round_csv(
        os.path.join(lmcache_dir, "warmup_round.csv")
    ),
    "query_round": parse_round_csv(
        os.path.join(lmcache_dir, "query_round.csv")
    ),
    "aggregate": lmcache_json,
}

max_ttft_slowdown = float(
    os.environ.get("MAX_TTFT_SLOWDOWN_PCT", "-60")
)
max_round_time_slowdown = float(
    os.environ.get("MAX_ROUND_TIME_SLOWDOWN_PCT", "-15")
)


def compare_metric(name, lmcache_value, baseline_value, threshold):
    try:
        lmcache_value = float(lmcache_value)
        baseline_value = float(baseline_value)
    except (ValueError, TypeError):
        return {
            "name": name,
            "baseline": None,
            "lmcache": None,
            "diff_percent": None,
            "threshold_percent": threshold,
            "passed": False,
        }

    if not (
        math.isfinite(lmcache_value)
        and math.isfinite(baseline_value)
    ):
        diff_pct = None
        passed = False
    elif baseline_value > 0:
        diff_pct = (
            (lmcache_value - baseline_value) / baseline_value
        ) * 100.0
        passed = diff_pct <= threshold
    else:
        diff_pct = None
        passed = False

    return {
        "name": name,
        "baseline": round(baseline_value, 4),
        "lmcache": round(lmcache_value, 4),
        "diff_percent": round(diff_pct, 2) if diff_pct is not None else None,
        "threshold_percent": threshold,
        "passed": passed,
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
    "attempt": int(os.environ.get("DIAG_ATTEMPT", "0")),
    "checkout_sha": os.environ.get("DIAG_CHECKOUT_SHA", "unknown"),
    "timestamp": os.environ.get("DIAG_TIMESTAMP", ""),
    "workload_settings": {
        "engine": os.environ.get("ENGINE_NAME", "vllm"),
        "model": os.environ.get("MODEL", ""),
        "document_length": int(os.environ.get("DOCUMENT_LENGTH", "10000")),
        "num_documents": int(os.environ.get("NUM_DOCUMENTS", "30")),
        "output_len": int(os.environ.get("OUTPUT_LEN", "200")),
        "repeat_count": int(os.environ.get("REPEAT_COUNT", "2")),
        "repeat_mode": os.environ.get("REPEAT_MODE", "tile"),
        "shuffle_seed": int(os.environ.get("SHUFFLE_SEED", "0")),
        "max_inflight_requests": int(
            os.environ.get("MAX_INFLIGHT_REQUESTS", "5")
        ),
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
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except OSError as exc:
        print(
            f"Failed to write diagnostic summary to {out_file}: {exc}",
            file=sys.stderr,
        )

print()
print("============================================")
print("=== Long Doc QA Diagnostic Summary ===")
print("============================================")
print(f"Verdict: {summary['verdict']}")
print(f"Attempt: {summary['attempt']}")
print(f"Timestamp: {summary['timestamp']}")
print(f"Checkout SHA: {summary['checkout_sha']}")
print()

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
                f"mean={stats['mean']:.4f}s, "
                f"p50={stats['p50']:.4f}s, "
                f"p95={stats['p95']:.4f}s, "
                f"p99={stats['p99']:.4f}s, "
                f"min={stats['min']:.4f}s, "
                f"max={stats['max']:.4f}s"
            )
        else:
            print(
                f"    {round_name} TTFT: 0 valid TTFT values "
                f"(total={info['total_samples']}, "
                f"successful={info['successful_samples']})"
            )
    print()

print("Comparisons:")
for key in ("query_ttft_per_prompt", "query_round_time_per_prompt"):
    item = summary["comparison"][key]
    if item["baseline"] is None or item["lmcache"] is None:
        print(f"  {key}: missing or invalid data -- FAIL")
        continue

    diff = item["diff_percent"]
    if diff is None:
        print(f"  {key}: invalid baseline value -- FAIL")
        continue

    description = (
        f"{abs(diff):.1f}% faster"
        if diff < 0 else f"{diff:.1f}% slower"
    )
    threshold = item["threshold_percent"]
    threshold_description = (
        f"need >= {abs(threshold):.0f}% faster"
        if threshold < 0 else f"max {threshold}% slower"
    )
    status = "PASS" if item["passed"] else "FAIL"
    print(
        f"  {key}: {item['lmcache']:.4f}s vs baseline "
        f"{item['baseline']:.4f}s ({description}, "
        f"{threshold_description}) -- {status}"
    )

print("============================================")
print()
PY

    DIAGNOSTIC_SAVED=1

    if [[ "${BUILDKITE_RETRY_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
        mkdir -p "$attempt_dir"

        [ ! -f "$summary_file" ] ||
            cp "$summary_file" "$attempt_dir/diagnostic_summary.json" 2>/dev/null || true

        for phase_dir in "$BASELINE_DIR" "$LMCACHE_OUT_DIR"; do
            [ -d "$phase_dir" ] || continue
            local destination="$attempt_dir/$(basename "$phase_dir")"
            mkdir -p "$destination"
            cp -r "$phase_dir"/. "$destination"/ 2>/dev/null || true
        done
    fi

    local diag_log="/tmp/build_${BUILD_ID}_attempt_${attempt}_long_doc_qa_diagnostic.log"
    cp "$summary_file" "$diag_log" 2>/dev/null || true
    cp "$summary_file" "/tmp/build_${BUILD_ID}_long_doc_qa_diagnostic.log" 2>/dev/null || true

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "$summary_file" 2>/dev/null || true
        buildkite-agent artifact upload "$BASELINE_DIR/*" 2>/dev/null || true
        buildkite-agent artifact upload "$LMCACHE_OUT_DIR/*" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT

    if [ "$rc" -ne 0 ] && [ "$CURRENT_VERDICT" != "FAIL" ]; then
        save_diagnostic_summary "FAIL" || true
    fi

    exit "$rc"
}
trap on_exit EXIT

echo "=== Long Doc QA Test ==="
echo "Model: $MODEL"
echo "Engine Port (with LMCache): $ENGINE_PORT"
echo "${ENGINE_NAME:-vllm} Baseline Port: $ENGINE_BASELINE_PORT"
echo "Document length: $DOCUMENT_LENGTH"
echo "Number of documents: $NUM_DOCUMENTS"
echo "Output length: $OUTPUT_LEN"
echo "Results dir: $LONG_DOC_QA_DIR"
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
        "$PYTHON_EXEC"
        "$LMCACHE_DIR/benchmarks/long_doc_qa/long_doc_qa.py"
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

    (
        cd "$phase_dir"
        "${benchmark_command[@]}" 2>>"$output_file" | tee "$phase_result_file"
    )

    "$PYTHON_EXEC" "$COMMON_WORKLOAD_DIR/long_doc_qa_diagnostics.py" \
        --phase "$description" \
        --engine "${INFERENCE_ENGINE:-${ENGINE_NAME:-vllm}}" \
        --results-dir "$phase_dir" \
        --command "${benchmark_command[@]}" \
        | tee "$phase_dir/diagnostics.txt"

    cp -f "$phase_result_file" \
        "$LONG_DOC_QA_DIR/${description}_result.json" 2>/dev/null || true
    cp -f "$output_file" \
        "$LONG_DOC_QA_DIR/${description}_output.txt" 2>/dev/null || true

    if [[ -n "$result_file" && "$result_file" != "$phase_result_file" ]]; then
        cp -f "$phase_result_file" "$result_file" 2>/dev/null || true
    fi

    echo "$description benchmark completed"
    echo
}

extract_json_field() {
    local json_file="$1"
    local field="$2"

    if [[ ! -f "$json_file" ]]; then
        echo null
        return
    fi

    "$PYTHON_EXEC" - "$json_file" "$field" <<'PY'
import json
import sys

path, field = sys.argv[1:3]

try:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    data = json.loads(lines[-1]) if lines else {}
    value = data.get(field)
    print(value if value is not None else "null")
except (OSError, ValueError, TypeError):
    print("null")
PY
}

compare_results() {
    local lmcache_result="$LMCACHE_OUT_DIR/lmcache_result.json"
    local baseline_result="$BASELINE_DIR/baseline_result.json"

    if [[ ! -f "$lmcache_result" && -f "$LONG_DOC_QA_DIR/lmcache_result.json" ]]; then
        lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    fi
    if [[ ! -f "$baseline_result" && -f "$LONG_DOC_QA_DIR/baseline_result.json" ]]; then
        baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"
    fi

    echo "=== Comparing benchmark results ==="

    if [[ ! -f "$lmcache_result" || ! -f "$baseline_result" ]]; then
        echo "Result files not found"
        return 1
    fi

    "$PYTHON_EXEC" - "$lmcache_result" "$baseline_result" <<'PY'
import json
import sys

def read_result(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return json.loads(lines[-1]) if lines else {}

def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

lmcache = read_result(sys.argv[1])
baseline = read_result(sys.argv[2])

metrics = (
    "query_ttft_per_prompt",
    "query_round_time_per_prompt",
    "warmup_round_time_per_prompt",
)

print()
print(f"{'Metric':<35} {'Baseline':>12} {'LMCache':>12} {'Diff':>10}")
print("-" * 72)

for metric in metrics:
    b = number(baseline.get(metric))
    l = number(lmcache.get(metric))

    if b is None or l is None:
        print(f"{metric:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
        continue

    diff = ((l - b) / b) * 100 if b > 0 else None
    diff_text = f"{diff:+.1f}%" if diff is not None else "N/A"
    print(f"{metric:<35} {b:>12.4f} {l:>12.4f} {diff_text:>10}")
print()
PY
}

verify_thresholds() {
    local lmcache_result="$LMCACHE_OUT_DIR/lmcache_result.json"
    local baseline_result="$BASELINE_DIR/baseline_result.json"

    [[ -f "$lmcache_result" ]] ||
        lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    [[ -f "$baseline_result" ]] ||
        baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"

    echo "=== Verifying LMCache performance vs baseline ==="

    if [[ ! -f "$lmcache_result" || ! -f "$baseline_result" ]]; then
        echo "Result files not found -- FAIL"
        save_diagnostic_summary "FAIL"
        return 1
    fi

    if "$PYTHON_EXEC" - \
        "$lmcache_result" "$baseline_result" \
        "$MAX_TTFT_SLOWDOWN_PCT" "$MAX_ROUND_TIME_SLOWDOWN_PCT" <<'PY'
import json
import math
import sys

lmcache_path, baseline_path, ttft_limit, round_limit = sys.argv[1:]
limits = {
    "query_ttft_per_prompt": float(ttft_limit),
    "query_round_time_per_prompt": float(round_limit),
}

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return json.loads(lines[-1]) if lines else {}

def value(data, key):
    try:
        result = float(data[key])
        return result if math.isfinite(result) else None
    except (KeyError, TypeError, ValueError):
        return None

lmcache = load(lmcache_path)
baseline = load(baseline_path)
failed = False

for metric, threshold in limits.items():
    l = value(lmcache, metric)
    b = value(baseline, metric)

    if l is None or b is None or b <= 0:
        print(f"{metric}: missing or invalid values -- FAIL")
        failed = True
        continue

    diff = ((l - b) / b) * 100
    faster_or_slower = (
        f"{abs(diff):.1f}% faster"
        if diff < 0 else f"{diff:.1f}% slower"
    )
    requirement = (
        f"need >= {abs(threshold):.0f}% faster"
        if threshold < 0 else f"max {threshold:.0f}% slower"
    )
    passed = diff <= threshold
    print(
        f"{metric}: LMCache={l:.4f}s, baseline={b:.4f}s "
        f"({faster_or_slower}; {requirement}) -- "
        f"{'PASS' if passed else 'FAIL'}"
    )
    failed |= not passed

sys.exit(1 if failed else 0)
PY
    then
        save_diagnostic_summary "PASS"
    else
        save_diagnostic_summary "FAIL"
        return 1
    fi
}

echo "=== Baseline benchmark ==="
run_long_doc_qa \
    "$ENGINE_BASELINE_PORT" \
    "$LONG_DOC_QA_DIR/baseline_result.json" \
    "baseline"

echo "=== LMCache benchmark ==="
run_long_doc_qa \
    "$ENGINE_PORT" \
    "$LONG_DOC_QA_DIR/lmcache_result.json" \
    "lmcache"

echo "=== Comparing results ==="
if ! compare_results; then
    echo "Comparison failed"
    save_diagnostic_summary "FAIL"
    exit 1
fi

echo "=== Verifying performance thresholds ==="
if ! verify_thresholds; then
    echo "Threshold verification failed"
    exit 1
fi

echo "=== Long Doc QA test completed ==="
