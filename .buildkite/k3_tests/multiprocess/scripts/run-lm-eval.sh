#!/usr/bin/env bash
# Run lm_eval workload test against vLLM server.
# Sends the same requests twice to test LMCache caching behavior.
# Adapted from the old Docker-based run-lm-eval.sh -- no venv setup needed
# (setup-env.sh + extras already installed by run.sh).
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
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

# Output directories
LM_EVAL_DIR="$RESULTS_DIR/lm_eval"
FIRST_RUN_DIR="$LM_EVAL_DIR/first_run"
SECOND_RUN_DIR="$LM_EVAL_DIR/second_run"

echo "=== LM-Eval Workload Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Results dir: $LM_EVAL_DIR"
echo ""

mkdir -p "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

run_lm_eval() {
    local run_name="$1"
    local output_dir="$2"

    echo "=== Running lm_eval ($run_name) ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'

    echo "$run_name completed"
    echo ""
}

# Extract only the raw model responses ("resps") keyed by doc_id from an
# lm_eval samples jsonl, emitting one normalized JSON object per line sorted by
# doc_id. We deliberately ignore "filtered_resps", "exact_match", "arguments",
# and other lm_eval bookkeeping: those reflect post-processing filters and
# metric computation that can change across lm_eval / dependency versions even
# when the model output is byte-identical. The cache-consistency invariant we
# care about is that a cache hit reproduces the same model generation, so we
# compare "resps" alone.
extract_resps() {
    local samples_file="$1"
    local out_file="$2"

    python3 - "$samples_file" > "$out_file" <<'PY'
import json
import sys

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append((obj["doc_id"], obj.get("resps")))

rows.sort(key=lambda row: row[0])
for doc_id, resps in rows:
    print(json.dumps({"doc_id": doc_id, "resps": resps}, sort_keys=True, ensure_ascii=False))
PY
}

# Copy the per-sample jsonl outputs from both runs into the workspace root so
# Buildkite collects them as downloadable artifacts (see artifact_paths in
# pipeline.yml). Called after both evals regardless of whether verification
# passes, so the raw lm_eval outputs are always available for debugging a
# mismatch. Files are renamed to stable, run-prefixed names because the source
# filenames embed a timestamp.
collect_samples_artifacts() {
    local first_dir="$1"
    local second_dir="$2"

    echo "=== Collecting lm_eval samples as Buildkite artifacts ==="

    local first_samples second_samples
    first_samples=$(find "$first_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    second_samples=$(find "$second_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)

    if [ -n "$first_samples" ]; then
        cp "$first_samples" "${REPO_ROOT}/lm_eval_first_run_samples.jsonl"
        echo "Collected: lm_eval_first_run_samples.jsonl (from $first_samples)"
    else
        echo "WARNING: no first-run samples_gsm8k_*.jsonl found in $first_dir"
    fi

    if [ -n "$second_samples" ]; then
        cp "$second_samples" "${REPO_ROOT}/lm_eval_second_run_samples.jsonl"
        echo "Collected: lm_eval_second_run_samples.jsonl (from $second_samples)"
    else
        echo "WARNING: no second-run samples_gsm8k_*.jsonl found in $second_dir"
    fi
    echo ""
}

verify_samples_match() {
    local first_dir="$1"
    local second_dir="$2"

    echo "=== Verifying model responses (resps) match ==="

    first_samples=$(find "$first_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    second_samples=$(find "$second_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)

    if [ -z "$first_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in first run directory: $first_dir"
        find "$first_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    if [ -z "$second_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in second run directory: $second_dir"
        find "$second_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    echo "First run samples: $first_samples"
    echo "Second run samples: $second_samples"

    first_resps=$(mktemp)
    second_resps=$(mktemp)

    extract_resps "$first_samples" "$first_resps"
    extract_resps "$second_samples" "$second_resps"

    if diff -q "$first_resps" "$second_resps" > /dev/null 2>&1; then
        echo "Model responses are identical!"
        rm -f "$first_resps" "$second_resps"
        return 0
    else
        echo "Model responses differ!"
        echo ""
        echo "=== Diff (first 50 lines) ==="
        diff "$first_resps" "$second_resps" | head -50 || true
        rm -f "$first_resps" "$second_resps"
        return 1
    fi
}

# First run -- populates cache
echo "============================================"
echo "=== First lm_eval run (cache population) ==="
echo "============================================"
run_lm_eval "first_run" "$FIRST_RUN_DIR"

# Second run -- should use cached results
echo "============================================"
echo "=== Second lm_eval run (cache hit) ==="
echo "============================================"
run_lm_eval "second_run" "$SECOND_RUN_DIR"

# Collect raw outputs as artifacts before verifying, so they are downloadable
# from Buildkite whether or not the consistency check passes.
collect_samples_artifacts "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

# Verify consistency
echo "============================================"
echo "=== Verifying output consistency ==="
echo "============================================"
if ! verify_samples_match "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"; then
    echo "Verification failed: model responses do not match"
    exit 1
fi

echo "============================================"
echo "=== LM-Eval workload test completed ==="
echo "============================================"
