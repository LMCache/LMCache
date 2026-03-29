#!/bin/bash
# xPyD run script — supports arbitrary numbers of prefillers (P) and decoders (D).
#
# Configuration via environment variables:
#   SHUFFLE_NUM_DOCUMENTS   shuffle_doc_qa --num-documents   (default: 3)
#   SHUFFLE_DOCUMENT_LENGTH shuffle_doc_qa --document-length (default: 3000)
#   SHUFFLE_OUTPUT_LEN      shuffle_doc_qa --output-len      (default: 200)
#   SERVICE_PORT      port for the final exposed service          (default: 10001)
#   PREFILLER_PORT    comma-separated vLLM ports for prefillers (default: 8100)
#   DECODER_PORT      comma-separated vLLM ports for decoders   (default: 8200)
#   TENSOR_PARALLEL   tensor-parallel size per vLLM instance     (default: 1)
#   DEFAULT_VENV_DIR  image / prefiller venv root (default: /opt/venv) — matches setup-blend-env.sh
#   TEST_VENV_DIR     wheel / decoder + proxy + benchmark venv (default: /workspace/.venv)
#   Legacy: DEFAULT_VENV, TEST_VENV still set *_DIR when *_DIR is unset.
#
# GPU assignment: each instance uses TENSOR_PARALLEL consecutive GPUs.
#   prefillers: GPUs 0..P*TP-1, decoders: GPUs P*TP..P*TP+D*TP-1.
# All instances share a single LMCache blend server on port 6566.

set -euo pipefail
set -x

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
WORK_LOG="/tmp/build_${BUILD_ID}_blend.log"
# Blend server, vLLM prefiller/decoder, and proxy stdout/stderr (main script uses WORK_LOG via tee).
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"
# Benchmark wall-clock limit (seconds). Exit 124 from `timeout` => failure. Default stays under blend pipeline 90m.
BENCHMARK_TIMEOUT_SEC="${BENCHMARK_TIMEOUT_SEC:-4800}"

: > "${WORK_LOG}"
: > "${VLLM_LOG}"

BGPIDS=()

collect_artifact() {
  echo "[INFO] Collecting logs into ${ARTIFACT}"
  cat "${WORK_LOG}" "${VLLM_LOG}" > "${ARTIFACT}" 2>/dev/null || true
}

finalize() {
  local rc=$?
  echo ""
  echo "[INFO] Shutting down all processes..."
  for pid in "${BGPIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  collect_artifact
  exit "$rc"
}

trap finalize EXIT INT TERM

exec > >(tee -a "${WORK_LOG}") 2>&1

wait_for_vllm_api() {
  local port=$1
  for i in {1..200}; do
    if curl --fail --silent "http://localhost:${port}/v1/models" > /dev/null; then
      echo "vLLM API on ${port} is awake!"
      return
    fi
    echo "Waiting for vLLM API on port ${port} to be ready... ($i/200)"
    sleep 2
  done
  echo "ERROR: vLLM API on port ${port} did not become ready in time."
  exit 1
}

# Scan every /tmp/build_${BUILD_ID}_*.log for severe/error lines (same family as other k3 build logs).
# Strip bash xtrace lines first: `set -x` tees "+ grep ... \\berror\\b ..." into the same log and
# would false-positive this check.
check_build_logs_for_errors() {
  local -a logs=()
  local f
  shopt -s nullglob
  logs=(/tmp/build_"${BUILD_ID}"_*.log)
  shopt -u nullglob
  if [[ ${#logs[@]} -eq 0 ]]; then
    echo "[WARN] No /tmp/build_${BUILD_ID}_*.log files found for error scan"
    return 0
  fi
  for f in "${logs[@]}"; do
    if grep -v '^+ ' "$f" 2>/dev/null | grep -iE '\berror\b|traceback|fatal' >/dev/null 2>&1; then
      echo "[FAIL] Found error/traceback/fatal pattern in: $f"
      echo "--- matching lines (first 80) ---"
      grep -v '^+ ' "$f" 2>/dev/null | grep -inE '\berror\b|traceback|fatal' | head -80 || true
      exit 1
    fi
  done
  echo "[PASS] No error/traceback/fatal pattern in build logs: ${logs[*]}"
}

export PYTHONUNBUFFERED=1

MODEL="${MODEL:-/data0/weishu-model/gpt-oss-20b}"
LMCACHE_MP_PORT=6566

SERVICE_PORT="${SERVICE_PORT:-10001}"
PREFILLER_PORT="${PREFILLER_PORT:-8100}"
DECODER_PORT="${DECODER_PORT:-8200}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
L2_FILE_PATH="${L2_FILE_PATH:-/mnt/}"
L2_POOL_SIZE="${L2_POOL_SIZE:-10}"
L2_SIZE_GB="${L2_SIZE_GB:-10}"

# Same layout as .buildkite/k3_harness/setup-blend-env.sh: DEFAULT_VENV_BIN=/opt/venv/bin, TEST_VENV_BIN=/workspace/.venv/bin
DEFAULT_VENV_DIR="${DEFAULT_VENV_DIR:-${DEFAULT_VENV:-/opt/venv}}"
DEFAULT_VENV_DIR="${DEFAULT_VENV_DIR%/}"
TEST_VENV_DIR="${TEST_VENV_DIR:-${TEST_VENV:-/workspace/.venv}}"
TEST_VENV_DIR="${TEST_VENV_DIR%/}"
DEFAULT_VENV_BIN="${DEFAULT_VENV_DIR}/bin"
TEST_VENV_BIN="${TEST_VENV_DIR}/bin"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-${DEFAULT_VENV_BIN}/python}"
TEST_PYTHON="${TEST_PYTHON:-${TEST_VENV_BIN}/python}"
# shuffle_doc_qa benchmark (repo-root cwd; see blend/run.sh)
SHUFFLE_NUM_DOCUMENTS="${SHUFFLE_NUM_DOCUMENTS:-3}"
SHUFFLE_DOCUMENT_LENGTH="${SHUFFLE_DOCUMENT_LENGTH:-3000}"
SHUFFLE_OUTPUT_LEN="${SHUFFLE_OUTPUT_LEN:-200}"
PREFILLER_VLLM_BIN="${PREFILLER_VLLM_BIN:-${DEFAULT_VENV_BIN}/vllm}"
DECODER_VLLM_BIN="${DECODER_VLLM_BIN:-${TEST_VENV_BIN}/vllm}"
IFS=',' read -ra PREFILLER_PORTS <<< "$PREFILLER_PORT"
IFS=',' read -ra DECODER_PORTS <<< "$DECODER_PORT"

NUM_PREFILLERS=${#PREFILLER_PORTS[@]}
NUM_DECODERS=${#DECODER_PORTS[@]}

echo "Configuration: ${NUM_PREFILLERS}P${NUM_DECODERS}D (TP=${TENSOR_PARALLEL})"
echo "  Prefiller ports: ${PREFILLER_PORTS[*]}"
echo "  Decoder ports:   ${DECODER_PORTS[*]}"
echo "  GPUs per instance: ${TENSOR_PARALLEL}"
echo "  Default venv dir: ${DEFAULT_VENV_DIR} (prefiller / blend server python: ${DEFAULT_PYTHON})"
echo "  Test venv dir:    ${TEST_VENV_DIR} (decoder / proxy / benchmark python: ${TEST_PYTHON})"
echo "  Prefiller vLLM:   ${PREFILLER_VLLM_BIN}"
echo "  Decoder vLLM:     ${DECODER_VLLM_BIN}"


export MAX_MODEL_LEN
export LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2025.1.0/host/linux-desktop-glibc_2_11_3-x64/:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# 1. Start the LMCache blend server
# ---------------------------------------------------------------------------

"${DEFAULT_PYTHON}" -m lmcache.v1.multiprocess.blend_server_v2 \
  --max-workers 1 \
  --port "${LMCACHE_MP_PORT}" \
  --l1-size 70 \
  --eviction-policy LRU \
  --chunk-size 1024 \
  --l1-align-bytes 16777216 \
  >>"${VLLM_LOG}" 2>&1 &
BGPIDS+=($!)

sleep 10
# ---------------------------------------------------------------------------
# 2. Start prefiller vLLM instances (GPUs 0..P-1, LMCacheMPCBConnector)
# ---------------------------------------------------------------------------
GPU_IDX=0
for port in "${PREFILLER_PORTS[@]}"; do
  GPU_END=$((GPU_IDX + TENSOR_PARALLEL - 1))
  CUDA_DEVS=$(seq -s, "$GPU_IDX" "$GPU_END")
  echo "Starting prefiller on GPUs ${CUDA_DEVS}, port ${port}"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVS \
    LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi \
    LMCACHE_REQUEST_TELEMETRY_ENDPOINT="http://localhost:5768/api/v1/telemetry" \
    VLLM_USE_FLASHINFER_MOE_FP8=0 \
    "${PREFILLER_VLLM_BIN}" serve  --model "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --port "$port" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-transfer-config \
      "{\"kv_connector\":\"LMCacheMPCBConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_MP_PORT}}}" \
    >>"${VLLM_LOG}" 2>&1 &
  BGPIDS+=($!)
  GPU_IDX=$((GPU_IDX + TENSOR_PARALLEL))
done


# ---------------------------------------------------------------------------
# 3. Start decoder vLLM instances (GPUs P..P+D-1, LMCacheMPConnector)
# ---------------------------------------------------------------------------
for port in "${DECODER_PORTS[@]}"; do
  GPU_END=$((GPU_IDX + TENSOR_PARALLEL - 1))
  CUDA_DEVS=$(seq -s, "$GPU_IDX" "$GPU_END")
  echo "Starting decoder on GPUs ${CUDA_DEVS}, port ${port}"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVS \
    VLLM_USE_FLASHINFER_MOE_FP8=0 \
    "${DECODER_VLLM_BIN}" serve  --model "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --port "$port" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-transfer-config \
      "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_MP_PORT}}}" \
    >>"${VLLM_LOG}" 2>&1 &
  BGPIDS+=($!)
  GPU_IDX=$((GPU_IDX + TENSOR_PARALLEL))
done

# ---------------------------------------------------------------------------
# 4. Wait for all vLLM instances to be ready
# ---------------------------------------------------------------------------
for port in "${PREFILLER_PORTS[@]}"; do
  wait_for_vllm_api "$port"
done
for port in "${DECODER_PORTS[@]}"; do
  wait_for_vllm_api "$port"
done

# ---------------------------------------------------------------------------
# 5. Start the CacheBlend proxy
# ---------------------------------------------------------------------------


"${TEST_PYTHON}" proxy.py \
  --port "$SERVICE_PORT" \
  --prefiller-host localhost --prefiller-port "$PREFILLER_PORT" \
  --decoder-host localhost --decoder-port "$DECODER_PORT" \
  --telemetry-port 5768 >>"${VLLM_LOG}" 2>&1 &
BGPIDS+=($!)

# ---------------------------------------------------------------------------
# 6. Benchmark (with timeout) + log error gate
# ---------------------------------------------------------------------------
if ! timeout "${BENCHMARK_TIMEOUT_SEC}" \
  "${TEST_PYTHON}" benchmarks/multi_doc_qa/shuffle_doc_qa.py \
  --num-documents "${SHUFFLE_NUM_DOCUMENTS}" \
  --document-length "${SHUFFLE_DOCUMENT_LENGTH}" \
  --output-len "${SHUFFLE_OUTPUT_LEN}"; then
  rc=$?
  if [[ "$rc" -eq 124 ]]; then
    echo "[FAIL] shuffle_doc_qa exceeded BENCHMARK_TIMEOUT_SEC=${BENCHMARK_TIMEOUT_SEC}s"
  else
    echo "[FAIL] shuffle_doc_qa exited with code ${rc}"
  fi
  exit 1
fi

check_build_logs_for_errors

echo "[PASS] Blend integration test completed successfully."
exit 0