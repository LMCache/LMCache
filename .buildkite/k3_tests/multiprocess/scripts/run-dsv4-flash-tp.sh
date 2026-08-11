#!/usr/bin/env bash
# L1 + slot-compression KV correctness for DeepSeek-V4-Flash (sparse-MLA +
# fp8) served with tensor parallelism.
#
# Why this test exists:
#   DeepSeek-V4-Flash interleaves several KV cache groups with different block
#   geometries (tokens_per_block 256/64/8/4): compressed fp8 MLA latents and
#   float32 sparse-attention indexer caches that pack multiple logical tokens
#   into one physical slot (MLAAttentionSpec.compress_ratio > 1). LMCache's
#   hybrid-group registration + slot-compression store/retrieve path
#   (docs/design/integration/vllm/hybrid-kv-cache-groups.md) serves these
#   groups per-block-size; this test pins that path end-to-end against the
#   real model with the L1 (CPU pool) tier enabled. L1 is the only storage
#   tier configured here, so a served retrieve proves the L1 path.
#
# Hardware requirement:
#   Hopper (SM90), datacenter Blackwell (SM100), or workstation Blackwell
#   (SM120, e.g. RTX PRO 6000). DeepSeek-V4-Flash routes its fp8 block-scaled
#   linears, MoE experts, hyper-connection prenorm GEMM and attention o_proj
#   einsum through DeepGEMM, so the arch must have DeepGEMM kernels.
#
#   On SM120 the vLLM wheel's *bundled* DeepGEMM does not, which is why this
#   script provisions one (see "Provision DeepGEMM" below). That is a
#   workaround for an upstream regression, not a permanent requirement:
#
#     vllm#47304 (07-02) pinned deepseek-ai/DeepGEMM@a6b593d2, whose
#       csrc/apis/layout.hpp dispatches on `arch_major == 10 or == 12`.
#       SM120 worked.
#     vllm#50000 (07-30) repointed the pin to vllm-project/DeepGEMM@f5a76426,
#       a fork branch based off DeepGEMM main. SM120 only ever lived on
#       nv_dev, so the `== 12` branches silently vanished and weight loading
#       began aborting at the fallthrough
#       DG_HOST_UNREACHABLE("Unknown SF transformation"),
#       csrc/apis/layout.hpp:60.
#     vllm#51003 (08-04) rebased a CUDA FP8 header fix onto that same
#       SM120-less base (e21c821f), which is the pin as of writing. Note
#       tools/install_deepgemm.sh still carries the stale comment
#       "targeting nv-dev branch due to sm120 support" above that SHA.
#
#   Delete this workaround once vLLM's pin carries SM120 again; the guard
#   below then no-ops on its own, since the bundled DeepGEMM will work.
#
#   Disabling DeepGEMM instead (VLLM_USE_DEEP_GEMM=0) does not work on SM120.
#   Only the fp8 linear and FP4 MoE call sites consult is_deep_gemm_supported();
#   measured on an SM120 node, those do demote (CutlassFp8BlockScaledMMKernel,
#   MARLIN) and the run then dies in mhc_pre_broadcast_tilelang, which calls
#   DeepGEMM unconditionally. Patching that gate exposes the next wall --
#   torch.ops._C.cutlass_scaled_mm has no SM120 block-scaled kernel in the
#   wheel -- and behind it DSv4's per-layer _o_proj -> deep_gemm_fp8_o_proj,
#   which has no non-DeepGEMM implementation on the CUDA path. (The ROCm and
#   XPU model modules define alternative _o_proj implementations, but
#   _select_dsv4_attn_cls only ever returns CUDA classes here.)
#
# This test is self-contained: it launches its own LMCache server + a TP=N
# vLLM instead of using launch-processes.sh / wait-for-servers.sh, since it
# needs tensor parallelism and the model's dedicated launch flags
# (docs/source/recipes/deepseek_v4_flash.rst). PIDs are written to the shared
# PID_FILE so the dispatcher's cleanup.sh trap still tears everything down.
#
# Flow (TP=4 by default):
#   1. Launch the LMCache server with an explicit L1 pool + vLLM (dev mode,
#      so /reset_prefix_cache is available).
#   2. vLLM run: send one long deterministic (greedy) completion request; vLLM
#      computes it from scratch and populates LMCache through the
#      slot-compression store path. Capture output A.
#   3. Reset vLLM's *local* prefix cache (APC) via POST /reset_prefix_cache,
#      leaving LMCache intact. Unlike the Kimi-Linear test, no vLLM restart is
#      needed: DeepSeek-V4-Flash keeps no Mamba state, all its cached state is
#      paged KV that the APC reset evicts. This saves a second multi-minute
#      160GB weight load.
#   4. LMCache retrieve run: send the identical request; vLLM's APC misses, so
#      the prefix KV -- MLA latents and indexer caches -- must be served by
#      LMCache. Capture output B.
#   5. Assert A == B. A broken slot-compression restore corrupts the restored
#      latents/indexer state and diverges the greedy decode. Both runs happen
#      in one vLLM process (same tuned kernels), and the retrieved KV is
#      byte-reloaded rather than recomputed, so a correct run is deterministic
#      and the outputs match exactly.
#   6. Assert LMCache actually served retrieves in run 2 (non-vacuous).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"

# LMCache chunk size must be a multiple of every KV cache group's
# tokens_per_block; DeepSeek-V4-Flash's groups use 256/64/8/4, so 256 is the
# smallest valid chunk size.
CHUNK_SIZE="${CHUNK_SIZE:-256}"
# One long prompt's KV is small (compressed MLA latents), but the pool is
# lazily allocated, so a generous default costs nothing up front.
L1_SIZE_GB="${L1_SIZE_GB:-40}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Readiness timeout for the vLLM launch. Owned by the test (a 160GB fp8
# TP-shard load is slow, and the first CI run also downloads the weights);
# deliberately does NOT reuse MAX_WAIT_SECONDS, which run-single-test.sh
# pre-exports to 300s -- that would shadow the value here.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-2700}"
# DeepSeek-V4-Flash has multiple KV cache groups with different block
# geometries. Keep per-group registration explicit instead of depending on the
# LMCache server default, which flipped off in #3869/#4437.
SEPARATE_OBJECT_GROUPS="${SEPARATE_OBJECT_GROUPS:-1}"
SEPARATE_OBJECT_GROUPS_ARG=""
if [ "$SEPARATE_OBJECT_GROUPS" = "1" ] || [ "$SEPARATE_OBJECT_GROUPS" = "true" ]; then
    SEPARATE_OBJECT_GROUPS_ARG="--separate-object-groups"
fi

# Tokens to generate per request. Greedy (temperature 0); a divergence in the
# restored KV shows up within the first few tokens, but a longer generation
# makes an accidental match astronomically unlikely.
MAX_TOKENS="${MAX_TOKENS:-128}"
# Seconds to let async LMCache stores drain before the retrieve run.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"

# vllm-project/DeepGEMM@codex/cuda129-fp8-include-5f33a180 (2fd67329).
#
# Chosen because it is a strict descendant of vLLM's current pin: `git compare
# f5a76426...5f33a180` reports ahead_by 96, behind_by 0, so nothing the pinned
# revision has is lost. It carries the byte-identical one-line commit
# "[Build] Include CUDA FP8 type in MQA layout header" that produced the pin
# (e21c821f), on top of nv_dev+situ, which has the SM120 kernels.
#
# Not a minimal delta, though: those 96 commits touch 70 files and include
# SM90 (kv_block=32, next_n=4) and SM100 (paged indexer, MQA logits sync)
# changes plus an upstream-main sync. That is acceptable here because this
# install only happens on SM120, where the alternative is not booting at all --
# but it is why the install is guarded by arch rather than applied everywhere.
#
# Its lineage also discharges the TODO vLLM left in deepgemm.cmake,
# "switch to nv_dev branch after it support situ": nv_dev+situ is exactly that.
DEEPGEMM_SM120_REF="${DEEPGEMM_SM120_REF:-2fd67329ec2942f65ba35d561256ab6ed3b903cb}"
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/vllm-project/DeepGEMM.git}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/dsv4_flash_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_vllm_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

echo "=== DeepSeek-V4-Flash L1 slot-compression correctness test ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | vLLM port: $VLLM_PORT | TP=$TENSOR_PARALLEL_SIZE"
echo "Chunk size: $CHUNK_SIZE | L1 pool: ${L1_SIZE_GB}GB"
echo "Results dir: $TP_DIR"
echo ""

# Send one greedy completion request and write the generated text to a file.
# Uses only the Python stdlib so no extra client dependency is required.
send_completion() {
    local out_file="$1"
    local run_name="$2"
    echo "=== Sending completion ($run_name) on port $VLLM_PORT ==="
    python3 - "$VLLM_PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS" "$out_file" <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, out_file = sys.argv[1:6]
prompt = open(prompt_file).read()
body = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "seed": 0,
    }
).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.load(resp)
text = data["choices"][0]["text"]
with open(out_file, "w") as f:
    f.write(text)
print(f"  generated {len(text)} chars")
PYEOF
    echo "$run_name completed"
    echo ""
}

# Reset vLLM's local prefix cache (APC) while preserving LMCache.
# reset_external defaults to false -> only vLLM's APC is cleared. Requires
# VLLM_SERVER_DEV_MODE=1 on the vLLM launch.
reset_vllm_prefix_cache() {
    echo "=== Resetting vLLM local prefix cache (LMCache preserved) ==="
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")
    if [ "$code" != "200" ]; then
        echo "Failed to reset prefix cache (HTTP $code). Is VLLM_SERVER_DEV_MODE=1?"
        return 1
    fi
    echo "vLLM prefix cache reset."
    echo ""
}

# Count completed LMCache retrieves in the server log (proves run 2 was served
# by LMCache, so the comparison can't pass vacuously by recomputing).
count_retrieves() {
    # NB: ``grep -c`` prints 0 *and* exits 1 on no match, so guard the file
    # existence and use ``|| true`` (not ``|| echo 0``) to avoid emitting "0\n0".
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

# ── 0. Provision DeepGEMM (SM120 only) ──────────────────────
# vLLM's _import_deep_gemm() prefers a `deep_gemm` in site-packages over the
# copy bundled in the wheel, so installing one overrides the arch-incomplete
# bundled build without rebuilding vLLM. Build steps mirror vLLM's own
# tools/install_deepgemm.sh. No-op on SM90/SM100, where the bundled copy is
# correct and must be left alone.
provision_deepgemm_sm120() {
    local arch_major
    arch_major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 \
        | tr -d ' ' | cut -d. -f1)
    if [ "$arch_major" != "12" ]; then
        echo "=== SM${arch_major}0: using vLLM's bundled DeepGEMM ==="
        return 0
    fi

    if python3 -c "import deep_gemm" 2>/dev/null; then
        echo "=== SM120: deep_gemm already present in site-packages ==="
        return 0
    fi

    echo "=== SM120: building DeepGEMM @ ${DEEPGEMM_SM120_REF:0:12} ==="
    local build_dir build_log
    build_dir="$(mktemp -d)"
    build_log="$TP_DIR/deepgemm_build.log"
    if ! {
        git clone --recursive --shallow-submodules \
            "$DEEPGEMM_REPO" "$build_dir/deepgemm" &&
        cd "$build_dir/deepgemm" &&
        git checkout "$DEEPGEMM_SM120_REF" &&
        python3 setup.py bdist_wheel &&
        { command -v uv >/dev/null 2>&1 && uv pip install dist/*.whl \
            || python3 -m pip install dist/*.whl; }
    } > "$build_log" 2>&1; then
        cd "$REPO_ROOT"
        echo "FAILED to build DeepGEMM. Tail of $build_log:"
        tail -40 "$build_log"
        rm -rf "$build_dir"
        return 1
    fi
    cd "$REPO_ROOT"
    rm -rf "$build_dir"
    echo "DeepGEMM installed: $(python3 -c 'import deep_gemm; print(deep_gemm.__file__)')"
}
provision_deepgemm_sm120

# ── 1. Launch LMCache MP server with an explicit L1 pool ────
echo "=== Launching LMCache MP server (port $LMCACHE_PORT, L1 ${L1_SIZE_GB}GB) ==="
lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers 4 \
    ${SEPARATE_OBJECT_GROUPS_ARG} \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Build a long, deterministic prompt ───────────────────
# A ~7-8k word document (spanning many 256-token LMCache chunks, so several
# slot-compressed blocks per group are stored) built by repeating a fixed
# paragraph, so the input is identical on every run.
#
# The prompt deliberately ends MID-SENTENCE: the retrieve run recomputes the
# few prompt-tail tokens beyond the last full chunk with a different prefill
# kernel shape than the cold run, which perturbs the logits slightly. An
# open-ended ending (e.g. "Summarize the text above:") puts the first
# generated token at a near-tie (measured top-2 logprob gap 0.0000 between
# ' The' and ' \n'), so that jitter flips the greedy argmax and the byte
# comparison flakes. Ending inside the 80x-repeated sentence pins every
# generated token to the memorized continuation (measured min top-2 gap 6.75
# across 128 tokens), making the comparison robust.
python3 - "$PROMPT_FILE" <<'PYEOF'
import sys

prompt_file = sys.argv[1]
paragraph = (
    "The poll() system call waits for one of a set of file descriptors to "
    "become ready to perform I/O. The set of file descriptors to be monitored "
    "is specified in the fds argument, which is an array of pollfd structures. "
    "The caller should specify the number of items in the fds array in nfds. "
    "The timeout argument specifies the number of milliseconds that poll() "
    "should block waiting for a file descriptor to become ready. The call will "
    "block until either a file descriptor becomes ready, the call is "
    "interrupted by a signal handler, or the timeout expires. "
)
# ~97 words/paragraph * 80 ~= 7.8k words.
with open(prompt_file, "w") as f:
    f.write(paragraph * 80)
    f.write("The poll() system call waits for one of a set of file")
PYEOF
echo "Prompt built ($(wc -w < "$PROMPT_FILE") words)."
echo ""

# ── 3. Launch vLLM (dev mode for /reset_prefix_cache) ───────
echo "=== Launching vLLM ($MODEL, TP=$TENSOR_PARALLEL_SIZE, port $VLLM_PORT) ==="
echo "Log: $VLLM_LOG"
# Save and unset VLLM_PORT: vLLM's internal get_open_port() would otherwise
# collide with the serving port for torch.distributed.
saved_port="$VLLM_PORT"
unset VLLM_PORT

# Launch flags per docs/source/recipes/deepseek_v4_flash.rst:
# --kv-cache-dtype fp8_ds_mla and --tokenizer-mode deepseek_v4 are required
# for this model; --enable-expert-parallel distributes the MoE experts across
# the TP ranks.
# --enforce-eager: capturing FULL decode CUDA graphs crashes in vLLM's
# custom all-reduce kernel at TP=4 ("illegal memory access",
# custom_all_reduce.cuh) on current nightlies; a correctness test needs no
# CUDA graphs, and eager also skips several minutes of capture/compile.
VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-expert-parallel \
    --kv-cache-dtype fp8_ds_mla \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --max-model-len auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$saved_port" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 120}}" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
export VLLM_PORT="$saved_port"
echo "vLLM started (PID=$VLLM_PID)"

if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$VLLM_LOG"; then
    echo "vLLM failed to start."
    exit 1
fi
echo ""

# ── 4. vLLM run: compute from scratch, populating LMCache ───
send_completion "$OUT_A" "vLLM run"

echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
sleep "$STORE_DRAIN_SECONDS"
retrieves_before=$(count_retrieves)

# ── 5. Invalidate vLLM's local prefix cache (keep LMCache) ──
reset_vllm_prefix_cache

# ── 6. Retrieve run: vLLM APC misses -> LMCache serves the KV ─
send_completion "$OUT_B" "LMCache retrieve run"
retrieves_after=$(count_retrieves)

# ── 7. Compare outputs and verify LMCache was actually used ──
echo "============================================"
echo "=== Verifying L1 slot-compression correctness ==="
echo "============================================"
echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

failed=0

if cmp -s "$OUT_A" "$OUT_B"; then
    echo "PASS: vLLM-run and LMCache-retrieve outputs are identical."
else
    echo "FAILED: outputs differ between the cold run and the LMCache-served run."
    echo "--- vLLM run (first 400 chars) ---"
    head -c 400 "$OUT_A"; echo
    echo "--- LMCache retrieve run (first 400 chars) ---"
    head -c 400 "$OUT_B"; echo
    failed=1
fi

if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "FAILED: LMCache served no retrieves during the retrieve run "
    echo "        (before=${retrieves_before}, after=${retrieves_after}); the "
    echo "        comparison would be vacuous."
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo ""
echo "============================================"
echo "=== DeepSeek-V4-Flash L1 test passed ==="
echo "  outputs identical; LMCache served $((retrieves_after - retrieves_before)) retrieves."
echo "============================================"
