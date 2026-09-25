#!/usr/bin/env bash
# ShareGPT differential correctness for the MP connector under vLLM preemption.
#
# By default this runs the two rungs that matter for a green build:
#
#   ref   plain vLLM, low concurrency   the ground truth: no preemption, no LMCache
#   D     LMCache,    high concurrency  preemption, and KV loaded back on resume
#
# D must reproduce ref's answers exactly, and must show requests actually
# resuming with a load (asserted from the connector's <resume-load> log, because
# vLLM's prefix-cache counters exclude requests with num_preemptions > 0).
#
# The concurrencies come from the KV pool, which the pipeline pins with
# NUM_GPU_BLOCKS_OVERRIDE so the pressure does not depend on the GPU model:
# preemption is guaranteed when the requests in flight cannot all be resident,
# and impossible when the concurrent prompts plus their max_tokens fit. Both are
# asserted from vllm:num_preemptions_total, so a mis-sized pool fails the run
# instead of quietly making it vacuous.
#
# DEBUGGING A FAILURE
# -------------------
# Re-run with PREEMPT_FULL_LADDER=1 to add three intermediate rungs that
# isolate which variable broke. Each adds one thing to the one below it:
#
#   A   plain vLLM, high concurrency   + preemption, still no LMCache
#   B   LMCache,    low concurrency    + the connector on a cold cache (writes only)
#   C   LMCache,    low concurrency    + the connector on a warm cache (reads)
#
#   * A fails      -> not LMCache. Either vLLM's preempt-and-recompute is lossy,
#                     or the exact-match oracle is invalid: check that
#                     VLLM_BATCH_INVARIANT=1, ENFORCE_EAGER and a pinned
#                     attention backend are in effect and that the model is one
#                     whose kernels vLLM covers (RMSNorm families).
#   * A passes, B fails -> storing KV perturbs generation.
#   * B passes, C fails -> what was stored reads back wrong.
#   * C passes, D fails -> specific to preemption/resume.
#
# The answers are kept as files and compared by request id, so they can be read
# by hand. With real ShareGPT traffic a cross-request KV mix-up is visible: an
# answer about one topic continues into another.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
CORRECTNESS_DIR="${REPO_ROOT}/.buildkite/correctness"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
OUT="${RESULTS_DIR}/preemption_correctness"

# ShareGPT must already be on the runner (same expectation as
# .buildkite/scripts/vllm-correctness.sh).
SHAREGPT_PATH="${SHAREGPT_PATH:-$HOME/correctness/.ShareGPT_V3_unfiltered_cleaned_split.json}"

NUM_REQUESTS="${PREEMPT_NUM_REQUESTS:-100}"
HOT_CONCURRENCY="${PREEMPT_HOT_CONCURRENCY:-40}"
REF_CONCURRENCY="${PREEMPT_REF_CONCURRENCY:-3}"
# Add the intermediate rungs that isolate a failure (see DEBUGGING above).
FULL_LADDER="${PREEMPT_FULL_LADDER:-0}"

VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"

SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -s "$SHAREGPT_PATH" ]; then
    # Pre-seeding $SHAREGPT_PATH on the agent skips this 670 MB download.
    echo "[INFO] ShareGPT not found at $SHAREGPT_PATH, downloading..."
    mkdir -p "$(dirname "$SHAREGPT_PATH")"
    if ! wget -q "$SHAREGPT_URL" -O "$SHAREGPT_PATH"; then
        rm -f "$SHAREGPT_PATH"
        echo "[ERROR] could not download the ShareGPT dataset"
        exit 1
    fi
fi

mkdir -p "$OUT"
echo "=== Preemption correctness (ShareGPT) ==="
echo "Model:          $MODEL"
echo "vLLM + LMCache: http://127.0.0.1:${VLLM_PORT}"
echo "vLLM baseline:  http://127.0.0.1:${VLLM_BASELINE_PORT}"
echo "Results:        $OUT"

# The converter keys its output by conversation id, so duplicate ids in the
# slice collapse. Take N from the dataset, never from the slice.
jq ".[0:${NUM_REQUESTS}]" "$SHAREGPT_PATH" > "$OUT/sharegpt_slice.json"
python3 "${CORRECTNESS_DIR}/sharegpt2openai.py" \
    -i "$OUT/sharegpt_slice.json" -o "$OUT/dataset.json"
EXPECTED=$(jq 'length' "$OUT/dataset.json")
echo "Requests:       $EXPECTED"

preemptions() {  # preemptions <port>
    curl -s "http://127.0.0.1:$1/metrics" \
        | awk '/^vllm:num_preemptions_total\{/ {printf "%.0f", $2}'
}

run() {  # run <port> <concurrency> <outfile>
    rm -f "$OUT/$3" "$OUT/${3%.txt}_length.txt"
    python3 "${CORRECTNESS_DIR}/async_request.py" \
        --model "$MODEL" \
        --endpoint "http://127.0.0.1:$1/v1/chat/completions" \
        --dataset_file "$OUT/dataset.json" \
        --request-number "$EXPECTED" \
        --max-concurrency "$2" \
        --output-file "$OUT/$3" > "$OUT/${3%.txt}_client.log" 2>&1
}

FAILED=0
REF_COUNT=0

# rung <label> <port> <concurrency> <outfile> <expect-preemption: yes|no>
rung() {
    local label="$1" port="$2" conc="$3" out="$4" expect="$5"
    local before after got
    before=$(preemptions "$port")
    run "$port" "$conc" "$out"
    after=$(preemptions "$port")
    got=$(( after - before ))
    echo "--- $label: concurrency $conc, $got preemptions ---"

    if [ "$expect" = "yes" ] && [ "$got" -eq 0 ]; then
        echo "[FAIL] $label preempted 0 times: the KV pool is too large for this"
        echo "       workload, so this run does not test preemption at all."
        FAILED=1
    fi
    if [ "$expect" = "no" ] && [ "$got" -ne 0 ]; then
        echo "[FAIL] $label preempted $got times but must not preempt; lower"
        echo "       PREEMPT_REF_CONCURRENCY or raise NUM_GPU_BLOCKS_OVERRIDE."
        FAILED=1
    fi

    local answers
    answers=$(grep -c '^chatcmpl-' "$OUT/$out" || true)

    # The reference defines how many answers a healthy run produces. It can be
    # fewer than the dataset: a conversation whose prompt plus max_tokens
    # exceeds --max-model-len is rejected, identically on every server, so it
    # simply drops out of the comparison.
    if [ "$out" = "ref.txt" ]; then
        REF_COUNT="$answers"
        echo "reference answered $REF_COUNT of $EXPECTED requests"
        if [ "$REF_COUNT" -eq 0 ]; then
            echo "[FAIL] the reference answered nothing; the run is void"
            FAILED=1
        elif [ "$REF_COUNT" -lt "$EXPECTED" ]; then
            echo "[INFO] $(( EXPECTED - REF_COUNT )) prompts do not fit in"
            echo "       --max-model-len and were rejected by every server"
        fi
        return
    fi

    if [ "$answers" -ne "$REF_COUNT" ]; then
        echo "[FAIL] $label produced $answers answers, reference produced $REF_COUNT"
        FAILED=1
    fi

    local cmp different only_ref only_this
    cmp=$(python3 "${CORRECTNESS_DIR}/compare_files.py" \
        --file1 "$OUT/ref.txt" --file2 "$OUT/$out")
    echo "$cmp" | sed -n '2,4p'
    different=$(echo "$cmp" | grep '^Different IDs:' | grep -oE '[0-9]+' || echo 0)
    only_ref=$(echo "$cmp" | awk '/^—— Only in File 1 ——$/,/^—— Only in File 2 ——$/' \
        | grep -cE '^chatcmpl-' || true)
    only_this=$(echo "$cmp" | awk '/^—— Only in File 2 ——$/,0' \
        | grep -cE '^chatcmpl-' || true)
    if [ "$different" -ne 0 ]; then
        echo "[FAIL] $label: $different answers differ from the reference"
        echo "$cmp" | sed -n '/—— Different IDs ——/,/—— Only in File 1 ——/p' | head -20
        FAILED=1
    fi
    if [ "$only_ref" -ne 0 ] || [ "$only_this" -ne 0 ]; then
        echo "[FAIL] $label answered a different set of requests than the"
        echo "       reference (only-in-ref=$only_ref only-in-$out=$only_this)"
        FAILED=1
    fi
}

rung "ref  (plain vLLM, no preemption)" "$VLLM_BASELINE_PORT" "$REF_CONCURRENCY" ref.txt no

if [ "$FULL_LADDER" = "1" ] || [ "$FULL_LADDER" = "true" ]; then
    rung "A    (plain vLLM, preemption)"    "$VLLM_BASELINE_PORT" "$HOT_CONCURRENCY" preempt.txt      yes
    rung "B    (LMCache, cold, no preempt)" "$VLLM_PORT"          "$REF_CONCURRENCY" lmcache_cold.txt no
    rung "C    (LMCache, warm, no preempt)" "$VLLM_PORT"          "$REF_CONCURRENCY" lmcache_warm.txt no
fi

# Cold cache unless the full ladder warmed it, which makes the resume-load
# assertion below unambiguous: with nothing pre-populated, the only KV a
# request can hit is KV it stored itself before being preempted.
rung "D    (LMCache, preemption)"       "$VLLM_PORT"          "$HOT_CONCURRENCY" lmcache_preempt.txt yes

# Rung D must actually have resumed requests that read their KV back, or it
# proves only that the connector did not corrupt anything. vLLM cannot report
# this (its prefix-cache counters exclude requests with num_preemptions > 0),
# so the connector logs it and we aggregate per request: the scheduler polls
# get_num_new_matched_tokens repeatedly per admission.
grep -o "<resume-load> req=[^ ]* apc=[0-9]* lmcache=[0-9]* load=[0-9]*" \
    "$VLLM_LOG" > "$OUT/resume_load.txt" || true
RESUMED=$(awk '{split($2,r,"=");split($5,c,"=");if(c[2]>mx[r[2]])mx[r[2]]=c[2]}
               END{for(i in mx) if(mx[i]>0) n++; print n+0}' "$OUT/resume_load.txt")
LOADED=$(awk '{split($2,r,"=");split($5,c,"=");if(c[2]>mx[r[2]])mx[r[2]]=c[2]}
              END{for(i in mx) s+=mx[i]; print s+0}' "$OUT/resume_load.txt")
echo "--- resume-load: $RESUMED requests recovered $LOADED prompt tokens from LMCache ---"
if [ "$RESUMED" -eq 0 ]; then
    echo "[FAIL] no request loaded KV from LMCache after being preempted, so the"
    echo "       agreement above does not exercise the resume path."
    FAILED=1
fi

echo
echo "Answers kept under $OUT (ref.txt, preempt.txt, lmcache_*.txt)."
if [ "$FAILED" -ne 0 ]; then
    echo "[FAIL] preemption correctness FAILED"
    exit 1
fi
echo "[PASS] every rung reproduces the reference answers"
