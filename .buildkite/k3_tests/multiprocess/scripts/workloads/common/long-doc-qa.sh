#!/usr/bin/env bash
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

# Relative performance thresholds (compared against baseline run in same job)
# Negative values mean LMCache must be *faster* than baseline by at least that %.
# Recent CI runs show ~77-84% TTFT improvement and ~27-40% round-time improvement,
# so requiring 60% and 15% respectively leaves comfortable headroom.
MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:--60}"
MAX_ROUND_TIME_SLOWDOWN_PCT="${MAX_ROUND_TIME_SLOWDOWN_PCT:--15}"

# Resolve paths before each benchmark changes its working directory.
# Canonicalize paths safely supporting paths with spaces.
# This runs before save_diagnostic_summary/on_exit are defined or trapped, so
# LMCACHE_DIR and LONG_DOC_QA_DIR are always absolute when the trap can fire.
LMCACHE_DIR="$(cd "$LMCACHE_DIR" 2>/dev/null && pwd || echo "$LMCACHE_DIR")"
mkdir -p "$RESULTS_DIR/long_doc_qa"
LONG_DOC_QA_DIR="$(cd "$RESULTS_DIR/long_doc_qa" && pwd)"
BASELINE_DIR="$LONG_DOC_QA_DIR/baseline"
LMCACHE_OUT_DIR="$LONG_DOC_QA_DIR/lmcache"

mkdir -p "$LONG_DOC_QA_DIR" "$BASELINE_DIR" "$LMCACHE_OUT_DIR"

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
import json
import os
import sys
import math
import csv

def calc_stats(values):
    if not values:
        return None
    n = len(values)
    s_val = sorted(values)
    mean = sum(s_val) / n
    variance = sum((x - mean) ** 2 for x in s_val) / n if n > 1 else 0.0
    std = math.sqrt(variance)

    def percentile(p):
        if n == 1:
            return s_val[0]
        idx = (n - 1) * (p / 100.0)
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        weight = idx - low
        return s_val[low] * (1.0 - weight) + s_val[high] * weight

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": s_val[0],
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": s_val[-1],
    }

def parse_round_csv(csv_path):
    if not os.path.isfile(csv_path):
        return {
            "present": False,
            "total_samples": 0,
            "successful_samples": 0,
            "ttft_stats": None,
        }
    successful_ttfts = []
    total = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            success = row.get("success", "").strip().lower() in ("true", "1")
            ttft_str = row.get("ttft", "")
            if success and ttft_str:
                try:
                    successful_ttfts.append(float(ttft_str))
                except ValueError:
                    pass
    return {
        "present": True,
        "total_samples": total,
        "successful_samples": len(successful_ttfts),
        "ttft_stats": calc_stats(successful_ttfts),
    }

def load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])
    except Exception:
        return None

base_dir = os.environ.get("LONG_DOC_QA_DIR", "")
baseline_dir = os.path.join(base_dir, "baseline")
lmcache_dir = os.path.join(base_dir, "lmcache")

baseline_json = load_json(os.path.join(baseline_dir, "baseline_result.json")) or load_json(os.path.join(base_dir, "baseline_result.json"))
lmcache_json = load_json(os.path.join(lmcache_dir, "lmcache_result.json")) or load_json(os.path.join(base_dir, "lmcache_result.json"))

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

max_ttft_slowdown = float(os.environ.get("MAX_TTFT_SLOWDOWN_PCT", "-60"))
max_round_time_slowdown = float(os.environ.get("MAX_ROUND_TIME_SLOWDOWN_PCT", "-15"))

def compare_metric(name, l_val, b_val, threshold):
    try:
        l = float(l_val)
        b = float(b_val)
    except (ValueError, TypeError):
        return {
            "name": name,
            "baseline": None,
            "lmcache": None,
            "diff_percent": None,
            "threshold_percent": threshold,
            "passed": False,
        }
    if b > 0:
        diff_pct = ((l - b) / b) * 100.0
    else:
        diff_pct = 0.0
    passed = diff_pct <= threshold
    return {
        "name": name,
        "baseline": round(b, 4),
        "lmcache": round(l, 4),
        "diff_percent": round(diff_pct, 2),
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
        max_ttft_slowdown
    ),
    "query_round_time_per_prompt": compare_metric(
        "query_round_time_per_prompt",
        l_agg.get("query_round_time_per_prompt"),
        b_agg.get("query_round_time_per_prompt"),
        max_round_time_slowdown
    ),
    "warmup_round_time_per_prompt": compare_metric(
        "warmup_round_time_per_prompt",
        l_agg.get("warmup_round_time_per_prompt"),
        b_agg.get("warmup_round_time_per_prompt"),
        0.0
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
        "max_inflight_requests": int(os.environ.get("MAX_INFLIGHT_REQUESTS", "5")),
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
    except Exception as e:
        print(f"Failed to write diagnostic summary to {out_file}: {e}", file=sys.stderr)

# Print human-readable summary
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
for phase_name in ["baseline", "lmcache"]:
    ph = summary["phases"][phase_name]
    agg = ph["aggregate"] or {}
    print(f"  [{phase_name.upper()}] (port {ph['port']})")
    print(f"    Aggregates: query_ttft={agg.get('query_ttft_per_prompt')}, query_round_time={agg.get('query_round_time_per_prompt')}, warmup_round_time={agg.get('warmup_round_time_per_prompt')}")
    for rnd in ["warmup_round", "query_round"]:
        r_info = ph[rnd]
        if r_info["present"]:
            st = r_info["ttft_stats"]
            if st:
                print(f"    {rnd} TTFT (N={st['count']}): mean={st['mean']:.4f}s, p50={st['p50']:.4f}s, p95={st['p95']:.4f}s, p99={st['p99']:.4f}s, min={st['min']:.4f}s, max={st['max']:.4f}s")
            else:
                print(f"    {rnd} TTFT: 0 valid samples (total={r_info['total_samples']})")
        else:
            print(f"    {rnd} CSV: not found")
    print()

print("Comparisons:")
for c_key in ["query_ttft_per_prompt", "query_round_time_per_prompt"]:
    c = summary["comparison"][c_key]
    if c["baseline"] is not None and c["lmcache"] is not None:
        diff_str = f"{abs(c['diff_percent']):.1f}% faster" if c['diff_percent'] < 0 else f"{c['diff_percent']:.1f}% slower"
        thresh_str = f"need >= {abs(c['threshold_percent']):.0f}% faster" if c['threshold_percent'] < 0 else f"max {c['threshold_percent']}% slower"
        status_str = "PASS" if c["passed"] else "FAIL"
        print(f"  {c_key}: {c['lmcache']:.4f}s vs baseline {c['baseline']:.4f}s ({diff_str}, {thresh_str}) -- {status_str}")
    else:
        print(f"  {c_key}: missing data -- FAIL")
print("============================================")
print()
PY

    DIAGNOSTIC_SAVED=1

    # Preserve attempt-specific copy without duplicating *.csv files under LONG_DOC_QA_DIR
    # on standard initial runs (where BUILDKITE_RETRY_COUNT is 0 or unset).
    # Only archive to attempt_$attempt when this is an actual retry (> 0).
    if [[ "${BUILDKITE_RETRY_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
        mkdir -p "$attempt_dir"
        if [ -f "$summary_file" ]; then
            cp "$summary_file" "$attempt_dir/diagnostic_summary.json" 2>/dev/null || true
        fi
        if [ -d "$LONG_DOC_QA_DIR/baseline" ]; then
            mkdir -p "$attempt_dir/baseline"
            cp -r "$LONG_DOC_QA_DIR/baseline"/* "$attempt_dir/baseline/" 2>/dev/null || true
        fi
        if [ -d "$LONG_DOC_QA_DIR/lmcache" ]; then
            mkdir -p "$attempt_dir/lmcache"
            cp -r "$LONG_DOC_QA_DIR/lmcache"/* "$attempt_dir/lmcache/" 2>/dev/null || true
        fi
    fi

    # Copy to log files in /tmp so cleanup.sh copies them and Buildkite collects them as *.log artifacts
    local diag_log="/tmp/build_${BUILD_ID}_attempt_${attempt}_long_doc_qa_diagnostic.log"
    cp "$summary_file" "$diag_log" 2>/dev/null || true
    cp "$summary_file" "/tmp/build_${BUILD_ID}_long_doc_qa_diagnostic.log" 2>/dev/null || true

    # Upload artifacts directly if buildkite-agent is available
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "$summary_file" 2>/dev/null || true
        buildkite-agent artifact upload "$LONG_DOC_QA_DIR/baseline/*" 2>/dev/null || true
        buildkite-agent artifact upload "$LONG_DOC_QA_DIR/lmcache/*" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "${CURRENT_VERDICT:-}" != "FAIL" ]; then
        save_diagnostic_summary "FAIL"
    fi
}
trap on_exit EXIT

echo "=== Long Doc QA Test ==="
echo "Model: $MODEL"
echo "Engine Port (with LMCache): $ENGINE_PORT"
echo "${ENGINE_NAME} Baseline Port (without LMCache): $ENGINE_BASELINE_PORT"
echo "Document length: $DOCUMENT_LENGTH"
echo "Number of documents: $NUM_DOCUMENTS"
echo "Output length: $OUTPUT_LEN"
echo "Results dir: $LONG_DOC_QA_DIR"
echo ""
echo "Performance thresholds (relative to baseline, negative = must be faster):"
echo "  Max TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}% (LMCache must be >= $(echo "$MAX_TTFT_SLOWDOWN_PCT" | tr -d '-')% faster)"
echo "  Max round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}% (LMCache must be >= $(echo "$MAX_ROUND_TIME_SLOWDOWN_PCT" | tr -d '-')% faster)"
echo ""

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

    # Execute benchmark inside phase_dir so warmup_round.csv and query_round.csv
    # are written to the phase's dedicated directory and do not overwrite each other.
    (
        cd "$phase_dir" && "${benchmark_command[@]}" 2>> "$output_file" | tee "$phase_result_file"
    )

    "${PYTHON_EXEC}" "$COMMON_WORKLOAD_DIR/long_doc_qa_diagnostics.py" \
        --phase "$description" \
        --engine "${INFERENCE_ENGINE:-${ENGINE_NAME:-vllm}}" \
        --results-dir "$phase_dir" \
        --command "${benchmark_command[@]}" | tee "$phase_dir/diagnostics.txt"

    # Maintain compatibility root copies for compare_results and artifact upload
    cp -f "$phase_result_file" "$LONG_DOC_QA_DIR/${description}_result.json" 2>/dev/null || true
    cp -f "$output_file" "$LONG_DOC_QA_DIR/${description}_output.txt" 2>/dev/null || true
    if [[ -n "$result_file" && "$result_file" != "$phase_result_file" ]]; then
        cp -f "$phase_result_file" "$result_file" 2>/dev/null || true
    fi

    echo "$description benchmark completed"
    echo ""
}

extract_json_field() {
    local json_file="$1"
    local field="$2"
    local json_line
    json_line=$(tail -n 1 "$json_file")
    "${PYTHON_EXEC}" -c "
import json, sys
try:
    data = json.loads('''$json_line''')
    value = data.get('$field', 'null')
    print(value if value is not None else 'null')
except json.JSONDecodeError:
    print('null')
"
}

compare_results() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache/lmcache_result.json"
    local baseline_result="$LONG_DOC_QA_DIR/baseline/baseline_result.json"

    if [ ! -f "$lmcache_result" ] && [ -f "$LONG_DOC_QA_DIR/lmcache_result.json" ]; then
        lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    fi
    if [ ! -f "$baseline_result" ] && [ -f "$LONG_DOC_QA_DIR/baseline_result.json" ]; then
        baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"
    fi

    echo "=== Comparing benchmark results ==="

    if [ ! -f "$lmcache_result" ] || [ ! -f "$baseline_result" ]; then
        echo "Result files not found"
        return 1
    fi

    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    lmcache_warmup_round_time=$(extract_json_field "$lmcache_result" "warmup_round_time_per_prompt")

    baseline_query_ttft=$(extract_json_field "$baseline_result" "query_ttft_per_prompt")
    baseline_query_round_time=$(extract_json_field "$baseline_result" "query_round_time_per_prompt")
    baseline_warmup_round_time=$(extract_json_field "$baseline_result" "warmup_round_time_per_prompt")

    echo ""
    echo "============================================"
    echo "=== Performance Summary ==="
    echo "============================================"

    "${PYTHON_EXEC}" << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def format_comparison(name, lmcache_val, baseline_val):
    lmcache = safe_float(lmcache_val)
    baseline = safe_float(baseline_val)
    if lmcache is None or baseline is None:
        return f"{name}: Unable to compare (invalid values)"
    if baseline > 0:
        diff_pct = ((lmcache - baseline) / baseline) * 100
        diff_str = f"{abs(diff_pct):.2f}% {'faster' if diff_pct < 0 else 'slower'}"
    else:
        diff_str = "N/A"
    return f"{name}:\n  Baseline:  {baseline:.4f}s\n  LMCache:   {lmcache:.4f}s\n  Diff:      {diff_str}"

print(format_comparison("query_ttft_per_prompt", "$lmcache_query_ttft", "$baseline_query_ttft"))
print()
print(format_comparison("query_round_time_per_prompt", "$lmcache_query_round_time", "$baseline_query_round_time"))
print()
print(format_comparison("warmup_round_time_per_prompt", "$lmcache_warmup_round_time", "$baseline_warmup_round_time"))
print()

# Summary table
print(f"{'Metric':<35} {'Baseline':>12} {'LMCache':>12} {'Diff':>10}")
print("-" * 70)
metrics = [
    ("query_ttft_per_prompt", "$baseline_query_ttft", "$lmcache_query_ttft"),
    ("query_round_time_per_prompt", "$baseline_query_round_time", "$lmcache_query_round_time"),
    ("warmup_round_time_per_prompt", "$baseline_warmup_round_time", "$lmcache_warmup_round_time"),
]
for name, bval, lval in metrics:
    b = safe_float(bval)
    l = safe_float(lval)
    if b is not None and l is not None:
        diff_pct = ((l - b) / b) * 100 if b > 0 else 0
        print(f"{name:<35} {b:>12.4f} {l:>12.4f} {diff_pct:>+9.1f}%")
    else:
        print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
print()
EOF

    return 0
}

verify_thresholds() {
    local lmcache_result="$LONG_DOC_QA_DIR/lmcache/lmcache_result.json"
    local baseline_result="$LONG_DOC_QA_DIR/baseline/baseline_result.json"

    if [ ! -f "$lmcache_result" ] && [ -f "$LONG_DOC_QA_DIR/lmcache_result.json" ]; then
        lmcache_result="$LONG_DOC_QA_DIR/lmcache_result.json"
    fi
    if [ ! -f "$baseline_result" ] && [ -f "$LONG_DOC_QA_DIR/baseline_result.json" ]; then
        baseline_result="$LONG_DOC_QA_DIR/baseline_result.json"
    fi

    echo "=== Verifying LMCache performance vs baseline ==="
    echo "Max allowed TTFT slowdown: ${MAX_TTFT_SLOWDOWN_PCT}%"
    echo "Max allowed query round time slowdown: ${MAX_ROUND_TIME_SLOWDOWN_PCT}%"
    echo ""

    lmcache_query_ttft=$(extract_json_field "$lmcache_result" "query_ttft_per_prompt")
    lmcache_query_round_time=$(extract_json_field "$lmcache_result" "query_round_time_per_prompt")
    baseline_query_ttft=$(extract_json_field "$baseline_result" "query_ttft_per_prompt")
    baseline_query_round_time=$(extract_json_field "$baseline_result" "query_round_time_per_prompt")

    "${PYTHON_EXEC}" << EOF
import sys

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def check_metric(name, lmcache_val, baseline_val, max_slowdown_pct):
    lmc = safe_float(lmcache_val)
    base = safe_float(baseline_val)
    if lmc is None or base is None or base <= 0:
        print(f"{name}: unable to compare (lmcache={lmcache_val}, baseline={baseline_val}) -- FAIL")
        return False
    pct = ((lmc - base) / base) * 100
    label = f"{abs(pct):.1f}% faster" if pct < 0 else f"{pct:.1f}% slower"
    if max_slowdown_pct < 0:
        threshold_label = f"need >= {abs(max_slowdown_pct):.0f}% faster"
    else:
        threshold_label = f"max {max_slowdown_pct}% slower"
    if pct <= max_slowdown_pct:
        print(f"{name}: {lmc:.4f}s vs baseline {base:.4f}s ({label}, {threshold_label}) -- PASS")
        return True
    else:
        print(f"{name}: {lmc:.4f}s vs baseline {base:.4f}s ({label}, {threshold_label}) -- FAIL")
        return False

failed = False
if not check_metric("query_ttft_per_prompt",
                     "$lmcache_query_ttft", "$baseline_query_ttft",
                     float("$MAX_TTFT_SLOWDOWN_PCT")):
    failed = True
if not check_metric("query_round_time_per_prompt",
                     "$lmcache_query_round_time", "$baseline_query_round_time",
                     float("$MAX_ROUND_TIME_SLOWDOWN_PCT")):
    failed = True

if failed:
    print("\nThreshold verification FAILED")
    sys.exit(1)
else:
    print("\nAll thresholds passed")
    sys.exit(0)
EOF
    local verify_status=$?

    if [ "$verify_status" -eq 0 ]; then
        save_diagnostic_summary "PASS"
        return 0
    else
        save_diagnostic_summary "FAIL"
        return 1
    fi
}

# Run benchmark against baseline
echo "============================================"
echo "=== Benchmark: Baseline ${ENGINE_NAME} (without LMCache) ==="
echo "============================================"
run_long_doc_qa "$ENGINE_BASELINE_PORT" "$LONG_DOC_QA_DIR/baseline_result.json" "baseline"

# Run benchmark against the inference engine with LMCache.
echo "============================================"
echo "=== Benchmark: ${ENGINE_NAME} with LMCache ==="
echo "============================================"
run_long_doc_qa "$ENGINE_PORT" "$LONG_DOC_QA_DIR/lmcache_result.json" "lmcache"

# Compare
echo "============================================"
echo "=== Comparing Results ==="
echo "============================================"
if ! compare_results; then
    echo "Comparison failed"
    save_diagnostic_summary "FAIL"
    exit 1
fi

# Verify thresholds
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