#!/usr/bin/env bash
# Preemption correctness ladder (T0-T3 of docs/design/integration/vllm/
# mp_preemption_correctness.md) against the two vLLM servers launched by
# launch-processes.sh.
#
# Requires the servers to have been launched with NUM_GPU_BLOCKS_OVERRIDE and
# MAX_NUM_SEQS (see pipeline.yml) so that the workload below overflows the KV
# pool by construction and vLLM preempts.  Preemption is asserted from vLLM's
# /metrics counter; outputs are compared as token ids against the baseline with
# a top-2 logprob near-tie classifier; the warm replay must be served from
# LMCache.  See preemption_correctness.py for the details.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

# Workload shape.  With NUM_GPU_BLOCKS_OVERRIDE=2048 and a 16-token block the
# pool holds 32k tokens; 64 requests of up to 1024 prompt + 1024 output tokens
# demand ~130k, so preemption is guaranteed by arithmetic.
PREEMPT_NUM_REQUESTS="${PREEMPT_NUM_REQUESTS:-64}"
PREEMPT_HOT_CONCURRENCY="${PREEMPT_HOT_CONCURRENCY:-64}"
PREEMPT_REPLAY_CONCURRENCY="${PREEMPT_REPLAY_CONCURRENCY:-4}"
PREEMPT_PROMPT_MIN="${PREEMPT_PROMPT_MIN:-512}"
PREEMPT_PROMPT_MAX="${PREEMPT_PROMPT_MAX:-1024}"
PREEMPT_MAX_TOKENS="${PREEMPT_MAX_TOKENS:-1024}"
PREEMPT_REPEATS="${PREEMPT_REPEATS:-2}"
PREEMPT_MAX_SLOWDOWN_PERCENT="${PREEMPT_MAX_SLOWDOWN_PERCENT:-}"

echo "=== Preemption correctness ladder ==="
echo "Model: $MODEL"
echo "vLLM + LMCache: http://127.0.0.1:${VLLM_PORT}"
echo "vLLM baseline:  http://127.0.0.1:${VLLM_BASELINE_PORT}"
echo "Results dir: $RESULTS_DIR/preemption_correctness"
mkdir -p "$RESULTS_DIR/preemption_correctness"

extra_args=()
if [ -n "$PREEMPT_MAX_SLOWDOWN_PERCENT" ]; then
    extra_args+=(--max-slowdown-percent "$PREEMPT_MAX_SLOWDOWN_PERCENT")
fi

python3 "${SCRIPT_DIR}/preemption_correctness.py" \
    --baseline-url "http://127.0.0.1:${VLLM_BASELINE_PORT}" \
    --lmcache-url "http://127.0.0.1:${VLLM_PORT}" \
    --model "$MODEL" \
    --num-requests "$PREEMPT_NUM_REQUESTS" \
    --hot-concurrency "$PREEMPT_HOT_CONCURRENCY" \
    --replay-concurrency "$PREEMPT_REPLAY_CONCURRENCY" \
    --prompt-min "$PREEMPT_PROMPT_MIN" \
    --prompt-max "$PREEMPT_PROMPT_MAX" \
    --max-tokens "$PREEMPT_MAX_TOKENS" \
    --repeats "$PREEMPT_REPEATS" \
    --output-dir "$RESULTS_DIR/preemption_correctness" \
    "${extra_args[@]}"
