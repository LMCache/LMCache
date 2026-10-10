#!/usr/bin/env bash
# Verify a long_doc_qa L2 run: performance thresholds, then L2 data flow and
# the MP observability surface from one /metrics snapshot.
#
# long-doc-qa-l2.sh runs this after its benchmark phases, while the L2 server
# is still alive. Every check runs even when an earlier one fails, so a
# threshold failure (a known intermittent verdict) still leaves the metrics
# evidence behind instead of exiting before it is collected. The exit status
# is non-zero if any check failed; no threshold or assertion is relaxed.
#
# Inputs (env):
#   L2_RESULTS_DIR         holds baseline_result.json and l2_result.json
#   METRICS_HTTP_PORT      LMCache HTTP server port serving /metrics (8080)
#   MIN_L2_SPEEDUP, MIN_L2_TTFT_SPEEDUP, MAX_WARMUP_OVERHEAD
#   METRICS_FETCH_ATTEMPTS, METRICS_FETCH_INTERVAL   bounded scrape retry (5, 2s)
#   LMCACHE_L2_LOG         optional; tailed when /metrics is unreachable
# Outputs (in L2_RESULTS_DIR):
#   prometheus_metrics.txt     the snapshot (empty when the scrape failed)
#   verification_summary.txt   one key=value line per check, plus overall
#
# Deliberately no ``set -e``: each step records its status and the summary
# at the end decides the exit code.
set -u
set -o pipefail

L2_RESULTS_DIR="${L2_RESULTS_DIR:?L2_RESULTS_DIR is required}"
METRICS_HTTP_PORT="${METRICS_HTTP_PORT:-8080}"
MIN_L2_SPEEDUP="${MIN_L2_SPEEDUP:-1.3}"
MIN_L2_TTFT_SPEEDUP="${MIN_L2_TTFT_SPEEDUP:-1.5}"
MAX_WARMUP_OVERHEAD="${MAX_WARMUP_OVERHEAD:-1.2}"
METRICS_FETCH_ATTEMPTS="${METRICS_FETCH_ATTEMPTS:-5}"
METRICS_FETCH_INTERVAL="${METRICS_FETCH_INTERVAL:-2}"
LMCACHE_L2_LOG="${LMCACHE_L2_LOG:-}"

L2_METRICS_FILE="$L2_RESULTS_DIR/prometheus_metrics.txt"
L2_SUMMARY_FILE="$L2_RESULTS_DIR/verification_summary.txt"

PERF_STATUS="FAIL"
SNAPSHOT_STATUS="FAIL"
DATA_FLOW_STATUS="SKIPPED"
OBSERVABILITY_STATUS="SKIPPED"
SNAPSHOT_DETAIL=""

extract_json_field() {
    local json_file="$1"
    local field="$2"
    tail -n 1 "$json_file" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    v = data.get('$field')
    print(v if v is not None else 'null')
except Exception:
    print('null')
"
}

# ---------------------------------------------------------------------------
# Step 3: Verify thresholds
# ---------------------------------------------------------------------------

echo "============================================"
echo "=== Verifying L2 Performance ==="
echo "============================================"

baseline_query_ttft=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "query_ttft_per_prompt")
baseline_query_round_time=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "query_round_time_per_prompt")
baseline_warmup_round_time=$(extract_json_field "$L2_RESULTS_DIR/baseline_result.json" "warmup_round_time_per_prompt")

l2_query_ttft=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "query_ttft_per_prompt")
l2_query_round_time=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "query_round_time_per_prompt")
l2_warmup_round_time=$(extract_json_field "$L2_RESULTS_DIR/l2_result.json" "warmup_round_time_per_prompt")

python3 << EOF
import sys

def sf(val):
    try: return float(val)
    except: return None

bqt  = sf("$baseline_query_ttft")
bqrt = sf("$baseline_query_round_time")
bwrt = sf("$baseline_warmup_round_time")
lqt  = sf("$l2_query_ttft")
lqrt = sf("$l2_query_round_time")
lwrt = sf("$l2_warmup_round_time")

min_spd  = float("$MIN_L2_SPEEDUP")
min_ttft = float("$MIN_L2_TTFT_SPEEDUP")
max_oh   = float("$MAX_WARMUP_OVERHEAD")

failed = False

print("=" * 60)
print("L2 Performance Summary")
print("=" * 60)
print(f"{'Metric':<35} {'Baseline':>12} {'L2':>12}")
print("-" * 60)
for name, bv, lv in [
    ("query_ttft_per_prompt (s)", bqt, lqt),
    ("query_round_time_per_prompt (s)", bqrt, lqrt),
    ("warmup_round_time_per_prompt (s)", bwrt, lwrt),
]:
    bs = f"{bv:.4f}" if bv else "N/A"
    ls = f"{lv:.4f}" if lv else "N/A"
    print(f"{name:<35} {bs:>12} {ls:>12}")

print()
print("=" * 60)
print("Threshold Verification")
print("=" * 60)

# 1. L2 query round-time speedup
if lqrt and bqrt and lqrt > 0:
    s = bqrt / lqrt
    ok = s >= min_spd
    print(f"[{'PASS' if ok else 'FAIL'}] L2 query speedup: {s:.2f}x (need >= {min_spd}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute L2 query speedup"); failed = True

# 2. L2 TTFT speedup
if lqt and bqt and lqt > 0:
    s = bqt / lqt
    ok = s >= min_ttft
    print(f"[{'PASS' if ok else 'FAIL'}] L2 TTFT speedup: {s:.2f}x (need >= {min_ttft}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute L2 TTFT speedup"); failed = True

# 3. Warmup overhead
if lwrt and bwrt and bwrt > 0:
    o = lwrt / bwrt
    ok = o <= max_oh
    print(f"[{'PASS' if ok else 'FAIL'}] Warmup overhead: {o:.2f}x (need <= {max_oh}x)")
    if not ok: failed = True
else:
    print("[FAIL] Cannot compute warmup overhead"); failed = True

print()
if failed:
    print("[FAIL] L2 performance verification FAILED")
    sys.exit(1)
else:
    print("[PASS] All L2 performance thresholds passed")
EOF
if [ $? -eq 0 ]; then
    PERF_STATUS="PASS"
else
    echo "Continuing to collect L2 metrics despite the performance failure."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Verify L2 data flow via Prometheus metrics
# ---------------------------------------------------------------------------

echo "============================================"
echo "=== Verifying L2 Data Flow (Metrics) ==="
echo "============================================"

# Retry briefly: when LMCache is relaunched on the same port as the
# previous instance, the Prometheus socket can take a moment to come
# back up, and a single-shot curl loses the metrics check silently.
> "$L2_METRICS_FILE"
attempt=0
while [ "$attempt" -lt "$METRICS_FETCH_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    if curl -sf "http://localhost:${METRICS_HTTP_PORT}/metrics" \
            > "$L2_METRICS_FILE" 2>/dev/null && [ -s "$L2_METRICS_FILE" ]; then
        SNAPSHOT_STATUS="PASS"
        break
    fi
    if [ "$attempt" -lt "$METRICS_FETCH_ATTEMPTS" ]; then
        sleep "$METRICS_FETCH_INTERVAL"
    fi
done

if [ "$SNAPSHOT_STATUS" != "PASS" ]; then
    SNAPSHOT_DETAIL="could not fetch /metrics from port ${METRICS_HTTP_PORT} after ${attempt} attempt(s)"
    echo "FAIL: could not fetch /metrics from LMCache HTTP server (port $METRICS_HTTP_PORT)."
    echo "       /metrics being unreachable means we cannot verify the L2"
    echo "       data flow or the observability surface; failing the test"
    echo "       rather than silently skipping."
    echo ""
    if [ -n "$LMCACHE_L2_LOG" ]; then
        echo "--- LMCache L2 server log (last 50 lines) ---"
        tail -50 "$LMCACHE_L2_LOG" 2>&1 || true
        echo ""
    fi
    echo "--- Listening sockets on port ${METRICS_HTTP_PORT} ---"
    (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) \
        | awk -v p=":${METRICS_HTTP_PORT}" '$0 ~ p'
else
    python3 -c "
import sys

with open('$L2_METRICS_FILE') as f:
    metrics_text = f.read()

def get_counter(name):
    for line in metrics_text.splitlines():
        if line.startswith(name + ' ') or line.startswith(name + '{'):
            return float(line.rsplit(' ', 1)[-1])
    return 0.0

# L1 metrics
l1_write_keys = get_counter('lmcache_mp_l1_write_chunks_total')

# L2 metrics
store_keys = get_counter('lmcache_mp_l2_store_submitted_objects_chunks_total')
store_succeeded = get_counter('lmcache_mp_l2_store_completed_objects_chunks_total')
prefetch_lookups = get_counter('lmcache_mp_l2_prefetch_lookup_requests_total')
l2_hit_keys = get_counter('lmcache_mp_lookup_hit_l2_keys_total')
prefetch_loaded = get_counter('lmcache_mp_l2_prefetch_load_completed_chunks_total')

print('=' * 60)
print('Data Flow Metrics')
print('=' * 60)
print(f'  L1 write keys:               {l1_write_keys:.0f}')
print(f'  L2 store keys submitted:     {store_keys:.0f}')
print(f'  L2 store keys succeeded:     {store_succeeded:.0f}')
print(f'  L2 prefetch lookups:         {prefetch_lookups:.0f}')
print(f'  L2 lookup hit keys:          {l2_hit_keys:.0f}')
print(f'  L2 prefetch keys loaded:     {prefetch_loaded:.0f}')
print()

failed = False

def check(cond, pass_msg, fail_msg):
    global failed
    if cond:
        print(f'[PASS] {pass_msg}')
    else:
        print(f'[FAIL] {fail_msg}')
        failed = True

# 1. L1 store activity (warmup writes KV to L1 before L2 store)
check(l1_write_keys > 0,
      f'L1 store: {l1_write_keys:.0f} keys written',
      'No keys written to L1 (expected > 0 from warmup)')

# 2. L2 store submitted and completed
check(store_keys > 0,
      f'L2 store: {store_keys:.0f} keys submitted',
      'No keys submitted to L2 store')
check(store_succeeded > 0,
      f'L2 store: {store_succeeded:.0f} keys succeeded',
      'No keys successfully stored to L2')

# 3. L2 prefetch submitted and completed (query round: L1 cold, L2 has data)
check(prefetch_lookups > 0,
      f'L2 prefetch: {prefetch_lookups:.0f} lookup requests',
      'No prefetch lookups (expected > 0 from query round)')
check(l2_hit_keys > 0,
      f'L2 prefetch: {l2_hit_keys:.0f} hit keys served from L2',
      'No lookup hit keys served from L2')
check(prefetch_loaded > 0,
      f'L2 prefetch: {prefetch_loaded:.0f} keys loaded',
      'No keys loaded from L2')

print()
if failed:
    print('[FAIL] Data flow verification FAILED')
    sys.exit(1)
else:
    print('[PASS] All data flow checks passed')
"
    if [ $? -eq 0 ]; then
        DATA_FLOW_STATUS="PASS"
    else
        DATA_FLOW_STATUS="FAIL"
    fi

    # -----------------------------------------------------------------------
    # Step 5: Verify the rest of the MP observability surface
    # -----------------------------------------------------------------------
    # The data-flow block above is L2-focused.  This block goes wider — it
    # asserts that every metric we publish from MP mode actually advances
    # during the run.  ``--metrics-sample-rate 1.0`` was set on the relaunch
    # so the histograms record on every event (the default 0.01 would leave
    # them empty in this short workload and flake the assertions).

    echo ""
    echo "============================================"
    echo "=== Verifying full MP observability surface ==="
    echo "============================================"

    python3 - "$L2_METRICS_FILE" <<'PYEOF'
import re
import sys

with open(sys.argv[1]) as f:
    text = f.read()


def counter_total(name: str) -> float:
    """Sum a counter across all label combinations."""
    total = 0.0
    pat = re.compile(rf"^{re.escape(name)}(\{{[^}}]*\}})?\s+([0-9eE+\-.]+)\s*$", re.M)
    for _, value in pat.findall(text):
        try:
            total += float(value)
        except ValueError:
            pass
    return total


def histogram_count(base_name: str) -> float:
    """Sum the ``_count`` series across all label combinations.

    Non-zero means the histogram observed at least one sample.

    The OTel→Prometheus bridge appends the OTel ``unit`` to the metric
    name (e.g. unit ``GB/s`` → ``GB_per_second``), so the actual series
    looks like ``<base>_GB_per_second_count``.  Match that as a suffix
    so this works whether or not the unit is present.
    """
    pat = re.compile(
        rf"^{re.escape(base_name)}(?:_[A-Za-z_]+)?_count(?:\{{[^}}]*\}})?\s+"
        rf"([0-9eE+\-.]+)\s*$",
        re.M,
    )
    return sum(float(v) for v in pat.findall(text))


def has_label(base_name: str, label: str) -> bool:
    """Check that at least one sample of `base_name` carries the named label.

    Tolerates the OTel unit suffix that Prometheus appends to histograms.
    """
    pat = re.compile(
        rf"^{re.escape(base_name)}(?:_[A-Za-z_]+)?(?:_count|_sum|_bucket)?"
        rf"\{{[^}}]*\b{re.escape(label)}=",
        re.M,
    )
    return bool(pat.search(text))


# (kind, metric_name, optional_label_to_assert_present_or_None)
checks = [
    # ── Newer counters (with label dimensions) ─────────────────────
    ("counter", "lmcache_mp_l2_store_completed_requests_total", "l2_name"),
    ("counter", "lmcache_mp_l2_load_completed_requests_total", "l2_name"),
    ("counter", "lmcache_mp_lookup_requested_tokens_total", "model_name"),
    ("counter", "lmcache_mp_lookup_hit_tokens_total", "model_name"),
    ("counter", "lmcache_mp_num_chunks_loaded_total", "worker_id"),
    # ── Histograms.  The OTel→Prometheus bridge appends the OTel
    # ``unit`` to the series name, so a histogram declared with
    # ``unit="GB/s"`` actually reports as
    # ``<name>_GB_per_second_count`` / ``..._sum`` / ``..._bucket``.
    # Match by base name and let the helper tolerate the unit suffix.
    ("hist", "lmcache_mp_l0_l1_store_throughput", None),
    ("hist", "lmcache_mp_l0_l1_load_throughput", None),
    ("hist", "lmcache_mp_l2_store_throughput", "l2_name"),
    ("hist", "lmcache_mp_l2_load_throughput", "l2_name"),
]

failed = False
for kind, name, label in checks:
    if kind == "counter":
        value = counter_total(name)
        ok = value > 0
        detail = f"total={value:.0f}"
    else:
        value = histogram_count(name)
        ok = value > 0
        detail = f"_count={value:.0f}"

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        failed = True
        continue

    if label is not None:
        if has_label(name, label):
            print(f"       └─ label '{label}' present")
        else:
            print(f"[FAIL] {name}: expected label '{label}' is missing")
            failed = True

print()
if failed:
    print("[FAIL] Observability metric verification FAILED")
    print("       (some metric did not advance, or its label dimension is missing)")
    sys.exit(1)
print("[PASS] All observability metrics populated.")
PYEOF
    if [ $? -eq 0 ]; then
        OBSERVABILITY_STATUS="PASS"
    else
        OBSERVABILITY_STATUS="FAIL"
    fi
fi

# ---------------------------------------------------------------------------
# Summary: every outcome, then one exit status
# ---------------------------------------------------------------------------

OVERALL_STATUS="PASS"
for status in "$PERF_STATUS" "$SNAPSHOT_STATUS" "$DATA_FLOW_STATUS" "$OBSERVABILITY_STATUS"; do
    if [ "$status" != "PASS" ]; then
        OVERALL_STATUS="FAIL"
    fi
done

snapshot_line="metrics_snapshot=${SNAPSHOT_STATUS}"
if [ -n "$SNAPSHOT_DETAIL" ]; then
    snapshot_line="${snapshot_line} (${SNAPSHOT_DETAIL})"
fi
{
    echo "performance=${PERF_STATUS}"
    echo "$snapshot_line"
    echo "data_flow=${DATA_FLOW_STATUS}"
    echo "observability=${OBSERVABILITY_STATUS}"
    echo "overall=${OVERALL_STATUS}"
} > "$L2_SUMMARY_FILE"

echo ""
echo "============================================"
echo "=== L2 Verification Summary ==="
echo "============================================"
cat "$L2_SUMMARY_FILE"
echo "Snapshot: $L2_METRICS_FILE"
echo "Summary:  $L2_SUMMARY_FILE"

if [ "$OVERALL_STATUS" != "PASS" ]; then
    exit 1
fi
