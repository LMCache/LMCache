#!/usr/bin/env bash
# Preemption correctness test using lm_eval (gsm8k).
#
# Purpose: verify that LMCache handles vLLM preemption correctly. When
# GPU_MEMORY_UTILIZATION is low (e.g. 0.5), the KV cache pool is small and
# vLLM is forced to preempt (swap out) in-flight requests under concurrent
# load. This test checks that:
#   1. Both lm_eval runs complete without crashing.
#   2. The gsm8k score drift between the two runs is within SCORE_TOLERANCE
#      (default 0.05, i.e. 5 percentage point) -- LMCache's preemption-resume
#      path must not corrupt KV and skew the score.
#   3. Preemption actually occurred during the runs (non-vacuous: proves the
#      low GPU_MEMORY_UTILIZATION env var actually triggered the codepath).
#   4. The score stays above SCORE_MIN (default 0.70) -- a correctness floor
#      that catches catastrophic regressions.
#
# Flow:
#   1. Run lm_eval (gsm8k) against vLLM+LMCache (populates LMCache).
#   2. Run lm_eval again (cache-hit run, may trigger more preemptions).
#   3. Assert score drift <= SCORE_TOLERANCE.
#   4. Assert preemption was observed in the vLLM log.
#   5. Assert both scores >= SCORE_MIN.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-300}"
# Max absolute difference allowed between the two runs' gsm8k scores.
# 0.05 = 5 percentage point on the [0, 1] exact_match scale.
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.05}"
# Minimum acceptable gsm8k score for either run (correctness floor).
SCORE_MIN="${SCORE_MIN:-0.80}"
# GPU memory utilization fraction passed to vLLM. Set low (e.g. via
# GPU_MEMORY_UTILIZATION=0.5 in pipeline.yml) to force the KV cache pool
# small enough that preemption is triggered under concurrent lm_eval load.
# launch-processes.sh passes this to vLLM via --gpu-memory-utilization.
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
# vLLM server log, scanned to confirm preemption actually occurred.
VLLM_LOG="${VLLM_LOG:-/tmp/build_${BUILD_ID}_vllm.log}"

PREEMPTION_DIR="$RESULTS_DIR/lm_eval_preemption"
FIRST_RUN_DIR="$PREEMPTION_DIR/first_run"
SECOND_RUN_DIR="$PREEMPTION_DIR/second_run"

echo "=== LM-Eval Preemption Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Score tolerance: $SCORE_TOLERANCE"
echo "Score minimum: $SCORE_MIN"
echo "Results dir: $PREEMPTION_DIR"
echo ""

mkdir -p "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

# Run one lm_eval gsm8k pass against a vLLM OpenAI-compatible server.
#
# Globals (read):
#   MODEL          - HuggingFace model id, echoed to lm_eval's model_args.
#   NUM_CONCURRENT - number of in-flight requests lm_eval issues.
#   LIMIT          - number of gsm8k samples to evaluate.
# Arguments:
#   $1 output_dir - directory lm_eval writes results_*.json / samples_*.jsonl to.
#   $2 run_name   - human-readable label used only in progress log lines.
# Outputs:
#   Writes lm_eval result and per-sample files under output_dir; prints progress
#   to stdout.
# Returns:
#   lm_eval's exit status (non-zero if the evaluation run fails). Propagated to
#   the caller via ``set -e``.
run_lm_eval() {
    local output_dir="$1"
    local run_name="$2"

    echo "=== Running lm_eval ($run_name) on port $VLLM_PORT ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    echo "$run_name completed"
    echo ""
}

# Count how many times the vLLM log mentions preemption events so far.
#
# Globals (read):
#   VLLM_LOG - path to the vLLM server log file.
# Arguments:
#   none.
# Outputs:
#   The integer count of "preempted" log lines to stdout (0 if the log file
#   does not exist yet).
count_preemptions() {
    [ -f "$VLLM_LOG" ] || { echo 0; return; }
    local count
    count=$(grep -c "<preempted>" "$VLLM_LOG" 2>/dev/null || true)
    echo "${count:-0}"
}

# ── 1. First run -- populates LMCache, may trigger preemptions ───
echo "============================================"
echo "=== First lm_eval run (cache population) ==="
echo "============================================"
preemptions_before=$(count_preemptions)
run_lm_eval "$FIRST_RUN_DIR" "first_run"

# ── 2. Second run -- cache-hit path, may also trigger preemptions ─
echo "============================================"
echo "=== Second lm_eval run (cache hit) ==="
echo "============================================"
run_lm_eval "$SECOND_RUN_DIR" "second_run"
preemptions_after=$(count_preemptions)

# ── 3. Assert correctness ────────────────────────────────────────
echo "============================================"
echo "=== Verifying preemption correctness ==="
echo "============================================"
echo "vLLM preemptions logged: before=${preemptions_before}, after=${preemptions_after}"

python3 - "$FIRST_RUN_DIR" "$SECOND_RUN_DIR" \
    "$SCORE_TOLERANCE" "$SCORE_MIN" "$preemptions_before" "$preemptions_after" <<'PYEOF'
import glob
import json
import os
import sys

first_dir, second_dir, tol_s, score_min_s, before_s, after_s = sys.argv[1:7]
tol = float(tol_s)
score_min = float(score_min_s)
preemptions_before = int(before_s)
preemptions_after = int(after_s)


def gsm8k_score_and_stderr(results_dir: str) -> tuple[float, float]:
    """Return the gsm8k (exact_match, stderr) from an lm_eval results directory.

    Prefers the strict-match variant; falls back to any non-stderr
    ``exact_match`` metric key (paired with its ``exact_match_stderr`` twin).

    Args:
        results_dir: Directory passed to ``lm_eval --output_path``. Searched
            recursively for the newest ``results_*.json`` (lm_eval nests it
            under a per-model subdirectory and stamps the filename with a
            timestamp).

    Returns:
        ``(score, stderr)``: the gsm8k ``exact_match`` accuracy in
        ``[0.0, 1.0]`` and its reported sampling stderr (0.0 if absent).

    Raises:
        SystemExit: If no ``results_*.json`` exists under ``results_dir`` or the
            newest one contains no ``exact_match`` metric for the gsm8k task.
    """
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    metrics = data["results"]["gsm8k"]
    preferred = "exact_match,strict-match"
    if preferred in metrics:
        stderr = float(metrics.get("exact_match_stderr,strict-match", 0.0))
        return float(metrics[preferred]), stderr
    for key, value in metrics.items():
        if key.startswith("exact_match,") and "stderr" not in key:
            variant = key.split(",", 1)[1]
            stderr = float(metrics.get(f"exact_match_stderr,{variant}", 0.0))
            return float(value), stderr
    raise SystemExit(f"No exact_match metric in {latest}: {sorted(metrics)}")


s_first, e_first = gsm8k_score_and_stderr(first_dir)
s_second, e_second = gsm8k_score_and_stderr(second_dir)

print(f"  First run  gsm8k exact_match = {s_first:.4f} +/- {e_first:.4f}")
print(f"  Second run gsm8k exact_match = {s_second:.4f} +/- {e_second:.4f}")
print(f"  tolerance = {tol}")
print(f"  score_min = {score_min}")

failures = []
# Score drift: a broken LMCache preemption-resume path would corrupt KV and
# skew results between runs.
if abs(s_first - s_second) > tol:
    failures.append(
        f"score drift between runs: |{s_first:.4f} - {s_second:.4f}| = "
        f"{abs(s_first - s_second):.4f} > {tol}"
    )
# Score floor: catastrophic regression check.
for label, score in [("first_run", s_first), ("second_run", s_second)]:
    if score < score_min:
        failures.append(
            f"{label} score {score:.4f} < score_min {score_min}"
        )
# Non-vacuous: preemption must have actually occurred during the test runs.
if preemptions_after <= preemptions_before:
    failures.append(
        "vLLM logged no preemption events during the test runs "
        f"(before={preemptions_before}, after={preemptions_after}); "
        "check GPU_MEMORY_UTILIZATION is set low enough to trigger preemption"
    )

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: score drift {abs(s_first - s_second):.4f} <= {tol}; "
    f"both scores >= {score_min}; "
    f"preemptions observed: {preemptions_after - preemptions_before}."
)
PYEOF

echo ""
echo "============================================"
echo "=== LM-Eval preemption test passed ==="
echo "============================================"
