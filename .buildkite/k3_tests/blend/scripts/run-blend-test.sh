#!/bin/bash
# xPyD run script — supports arbitrary numbers of prefillers (P) and decoders (D).
#
# Configuration via environment variables:
#   SHUFFLE_NUM_DOCUMENTS   shuffle_doc_qa --num-documents   (default: 3)
#   SHUFFLE_DOCUMENT_LENGTH shuffle_doc_qa --document-length (default: 3000)
#   SHUFFLE_OUTPUT_LEN      shuffle_doc_qa --output-len      (default: 200)
#   LMCACHE_SERVER_ENTRYPOINT lmcache server entrypoint: cli|legacy (default: cli)
#   LMCACHE_L1_SIZE_GB       LMCache server L1 size in GB       (default: 70)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-400}"

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
# Write logs under REPO_ROOT so they are visible on the host immediately via the bind mount.
# /tmp logs would be invisible until docker cp runs after the container exits.
LOG_DIR="${REPO_ROOT}/logs_${BUILD_ID}"
mkdir -p "${LOG_DIR}"
WORK_LOG="${LOG_DIR}/build_${BUILD_ID}_blend.log"
# Proxy stdout/stderr. Blend server/prefiller/decoder each get their own _blend_server/_prefiller_PORT/_decoder_PORT logs.
VLLM_LOG="${LOG_DIR}/build_${BUILD_ID}_proxy.log"
BLEND_SERVER_LOG="${LOG_DIR}/build_${BUILD_ID}_blend_server.log"
BENCHMARK_LOG="${LOG_DIR}/build_${BUILD_ID}_benchmark.log"
VERSIONS_LOG="${LOG_DIR}/versions.txt"
NVIDIA_SMI_LOG="${LOG_DIR}/nvidia-smi.txt"
# Benchmark wall-clock limit (seconds). Exit 124 from `timeout` => failure. Default stays under blend pipeline 90m.
BENCHMARK_TIMEOUT_SEC="${BENCHMARK_TIMEOUT_SEC:-4800}"

: > "${WORK_LOG}"
: > "${VLLM_LOG}"
: > "${BLEND_SERVER_LOG}"
: > "${BENCHMARK_LOG}"
: > "${VERSIONS_LOG}"
: > "${NVIDIA_SMI_LOG}"

declare -A RESERVED_PORTS=()

reserve_port() {
  local requested_port="$1"
  local label="$2"
  local next_probe="${requested_port}"
  local chosen

  while true; do
    chosen="$(find_free_port "${next_probe}")"
    if [[ -z "${RESERVED_PORTS[$chosen]+x}" ]]; then
      RESERVED_PORTS["$chosen"]=1
      if [[ "${chosen}" != "${requested_port}" ]]; then
        echo "[INFO] ${label}: requested ${requested_port}, using free port ${chosen}" >&2
      else
        echo "[INFO] ${label}: using requested free port ${chosen}" >&2
      fi
      echo "${chosen}"
      return 0
    fi
    next_probe=$((chosen + 1))
  done
}

resolve_port_csv() {
  local label="$1"
  local csv="$2"
  local -a requested=()
  local -a resolved=()
  local port
  local idx=0

  IFS=',' read -ra requested <<< "${csv}"
  for port in "${requested[@]}"; do
    port="${port//[[:space:]]/}"
    if [[ -z "${port}" ]]; then
      echo "ERROR: Empty port in ${label}: '${csv}'" >&2
      exit 1
    fi
    resolved+=("$(reserve_port "${port}" "${label}[${idx}]")")
    idx=$((idx + 1))
  done

  local joined
  joined="$(IFS=','; echo "${resolved[*]}")"
  echo "${joined}"
}

finalize() {
  local rc=$?
  echo ""
  echo "[INFO] Shutting down all processes..."
  cleanup_pids
  echo "[INFO] Logs: ${LOG_DIR}/"
  exit "$rc"
}

trap finalize EXIT INT TERM

exec > >(tee -a "${WORK_LOG}") 2>&1

check_build_logs_for_errors() {
  local -a logs=()
  local f
  local sanitized
  local fatal_pattern
  # Scan only infrastructure/runtime failure signatures. Benchmark logs include
  # arbitrary model generations, so a broad ``error`` grep can self-fail a
  # successful E2E when the model emits text like "formatting error".
  fatal_pattern='Traceback|\bfatal\b|CUDA error|NCCL.*(error|fail)'
  fatal_pattern+='|ZMQ.*timeout|HTTP/1\.1" 5|status_code=5'
  fatal_pattern+='|Internal server error|EngineDeadError|engine process failed'
  fatal_pattern+='|benchmark.*timeout|timed out waiting for telemetry'
  fatal_pattern+='|request.*exception|RuntimeError|process died unexpectedly'
  fatal_pattern+='|exited with code [1-9]'
  shopt -s nullglob
  logs=("${LOG_DIR}"/build_"${BUILD_ID}"_*.log)
  shopt -u nullglob
  if [[ ${#logs[@]} -eq 0 ]]; then
    echo "[WARN] No build logs found in ${LOG_DIR}/ for error scan"
    return 0
  fi
  for f in "${logs[@]}"; do
    sanitized="$(mktemp)"
    grep -h -v '^+ ' "$f" 2>/dev/null >"${sanitized}" || true
    if grep -iE "${fatal_pattern}" "${sanitized}" >/dev/null 2>&1; then
      echo "[FAIL] Found fatal/runtime pattern in: $f"
      echo "--- matching lines (first 80) ---"
      grep -inE "${fatal_pattern}" "${sanitized}" | head -80 || true
      echo "--- context windows (first 8 matches, +/- 12 lines) ---"
      grep -inE "${fatal_pattern}" "${sanitized}" | head -8 | cut -d: -f1 | while read -r line_no; do
        start=$(( line_no > 12 ? line_no - 12 : 1 ))
        end=$(( line_no + 12 ))
        echo "--- ${f}:${start}-${end} ---"
        sed -n "${start},${end}p" "${sanitized}" || true
      done
      rm -f "${sanitized}"
      exit 1
    fi
    rm -f "${sanitized}"
  done
  echo "[PASS] No fatal/runtime pattern in build logs: ${logs[*]}"
}

export PYTHONUNBUFFERED=1

MODEL="${MODEL:-openai/gpt-oss-20b}"
LMCACHE_MP_PORT_REQUESTED="${LMCACHE_MP_PORT:-6566}"
SERVICE_PORT_REQUESTED="${SERVICE_PORT:-10001}"
PREFILLER_PORT_REQUESTED="${PREFILLER_PORT:-8100}"
DECODER_PORT_REQUESTED="${DECODER_PORT:-8200}"
TELEMETRY_PORT_REQUESTED="${TELEMETRY_PORT:-5768}"
LMCACHE_HTTP_PORT_REQUESTED="${LMCACHE_HTTP_PORT:-8080}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
L2_FILE_PATH="${L2_FILE_PATH:-/mnt/}"
L2_POOL_SIZE="${L2_POOL_SIZE:-10}"
L2_SIZE_GB="${L2_SIZE_GB:-10}"
LMCACHE_SERVER_ENTRYPOINT="${LMCACHE_SERVER_ENTRYPOINT:-cli}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-70}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
LMCACHE_L1_ALIGN_BYTES="${LMCACHE_L1_ALIGN_BYTES:-16777216}"

# The blend E2E launches all vLLM instances inside one worker/container.  Some
# runtime environments (including Modal's container networking) expose a host
# address that NCCL can initialize with, but Gloo then fails while creating
# vLLM's CPU-side model-parallel group.  Pin vLLM/Gloo to loopback by default;
# callers can still override these if they intentionally run across nodes.
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

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
SHUFFLE_DOCUMENT_LENGTH="${SHUFFLE_DOCUMENT_LENGTH:-1000}"
SHUFFLE_OUTPUT_LEN="${SHUFFLE_OUTPUT_LEN:-200}"
PREFILLER_VLLM_BIN="${PREFILLER_VLLM_BIN:-${DEFAULT_VENV_BIN}/vllm}"
DECODER_VLLM_BIN="${DECODER_VLLM_BIN:-${TEST_VENV_BIN}/vllm}"
LMCACHE_MP_PORT="$(reserve_port "${LMCACHE_MP_PORT_REQUESTED}" "blend_server")"
LMCACHE_HTTP_PORT="$(reserve_port "${LMCACHE_HTTP_PORT_REQUESTED}" "blend_http")"
TELEMETRY_PORT="$(reserve_port "${TELEMETRY_PORT_REQUESTED}" "telemetry_server")"
SERVICE_PORT="$(reserve_port "${SERVICE_PORT_REQUESTED}" "proxy_service")"
PREFILLER_PORT="$(resolve_port_csv "prefiller" "${PREFILLER_PORT_REQUESTED}")"
DECODER_PORT="$(resolve_port_csv "decoder" "${DECODER_PORT_REQUESTED}")"
IFS=',' read -ra PREFILLER_PORTS <<< "$PREFILLER_PORT"
IFS=',' read -ra DECODER_PORTS <<< "$DECODER_PORT"
export SERVICE_PORT

NUM_PREFILLERS=${#PREFILLER_PORTS[@]}
NUM_DECODERS=${#DECODER_PORTS[@]}

echo "Configuration: ${NUM_PREFILLERS}P${NUM_DECODERS}D (TP=${TENSOR_PARALLEL})"
echo "  Prefiller ports: ${PREFILLER_PORTS[*]}"
echo "  Decoder ports:   ${DECODER_PORTS[*]}"
echo "  Service port:    ${SERVICE_PORT}"
echo "  Telemetry port:  ${TELEMETRY_PORT}"
echo "  Blend MP port:   ${LMCACHE_MP_PORT}"
echo "  Blend HTTP port: ${LMCACHE_HTTP_PORT}"
echo "  LMCache server:  ${LMCACHE_SERVER_ENTRYPOINT} (engine-type=blend)"
echo "  GPUs per instance: ${TENSOR_PARALLEL}"
echo "  Default venv dir: ${DEFAULT_VENV_DIR} (prefiller vLLM: image-built)"
echo "  Test venv dir:    ${TEST_VENV_DIR} (blend server / decoder vLLM / proxy / benchmark: nightly)"
echo "  Prefiller vLLM:   ${PREFILLER_VLLM_BIN}"
echo "  Decoder vLLM:     ${DECODER_VLLM_BIN}"


export MAX_MODEL_LEN
export LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2025.1.0/host/linux-desktop-glibc_2_11_3-x64/:${LD_LIBRARY_PATH:-}

{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "model=${MODEL}"
  echo "python_default=${DEFAULT_PYTHON}"
  echo "python_test=${TEST_PYTHON}"
  "${TEST_PYTHON}" - <<'PY' || true
import importlib.metadata as md
for pkg in ("lmcache", "vllm", "torch"):
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}=unavailable ({exc})")
PY
} >"${VERSIONS_LOG}" 2>&1

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${NVIDIA_SMI_LOG}" 2>&1 || true
fi

CACHEBLEND_KV_TRANSFER_CONFIG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_MP_PORT},\"lmcache.mp.cacheblend\":true}}"
echo "  KV config:        ${CACHEBLEND_KV_TRANSFER_CONFIG}"

# ---------------------------------------------------------------------------
# 1. Start the LMCache blend server
# ---------------------------------------------------------------------------

if [[ "${LMCACHE_SERVER_ENTRYPOINT}" == "cli" ]]; then
  "${TEST_VENV_BIN}/lmcache" server \
    --engine-type blend \
    --host localhost \
    --port "${LMCACHE_MP_PORT}" \
    --http-host 0.0.0.0 \
    --http-port "${LMCACHE_HTTP_PORT}" \
    --max-workers 1 \
    --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
    --eviction-policy LRU \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --l1-align-bytes "${LMCACHE_L1_ALIGN_BYTES}" \
    >>"${BLEND_SERVER_LOG}" 2>&1 &
elif [[ "${LMCACHE_SERVER_ENTRYPOINT}" == "legacy" ]]; then
  "${TEST_PYTHON}" -m lmcache.v1.multiprocess.blend_server_v2 \
    --max-workers 1 \
    --port "${LMCACHE_MP_PORT}" \
    --l1-size "${LMCACHE_L1_SIZE_GB}" \
    --eviction-policy LRU \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --l1-align-bytes "${LMCACHE_L1_ALIGN_BYTES}" \
    >>"${BLEND_SERVER_LOG}" 2>&1 &
else
  echo "ERROR: LMCACHE_SERVER_ENTRYPOINT must be cli or legacy, got ${LMCACHE_SERVER_ENTRYPOINT}" >&2
  exit 1
fi
TRACKED_PIDS+=($!)

sleep 10
if command -v curl >/dev/null 2>&1; then
  curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" >/dev/null 2>&1 || true
  curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/status" >>"${BLEND_SERVER_LOG}" 2>&1 || true
fi
# ---------------------------------------------------------------------------
# 2. Start prefiller vLLM instances (GPUs 0..P-1, CacheBlend-enabled LMCacheMPConnector)
# ---------------------------------------------------------------------------
GPU_IDX=0
for port in "${PREFILLER_PORTS[@]}"; do
  GPU_END=$((GPU_IDX + TENSOR_PARALLEL - 1))
  CUDA_DEVS=$(seq -s, "$GPU_IDX" "$GPU_END")
  PREFILLER_LOG="${LOG_DIR}/build_${BUILD_ID}_prefiller_${port}.log"
  : > "${PREFILLER_LOG}"
  echo "Starting prefiller on GPUs ${CUDA_DEVS}, port ${port}"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVS \
    LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi \
    LMCACHE_REQUEST_TELEMETRY_ENDPOINT="http://localhost:${TELEMETRY_PORT}/api/v1/telemetry" \
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
      "${CACHEBLEND_KV_TRANSFER_CONFIG}" \
    >>"${PREFILLER_LOG}" 2>&1 &
  TRACKED_PIDS+=($!)
  GPU_IDX=$((GPU_IDX + TENSOR_PARALLEL))
done


# ---------------------------------------------------------------------------
# 3. Start decoder vLLM instances (GPUs P..P+D-1, LMCacheMPConnector)
# ---------------------------------------------------------------------------
for port in "${DECODER_PORTS[@]}"; do
  GPU_END=$((GPU_IDX + TENSOR_PARALLEL - 1))
  CUDA_DEVS=$(seq -s, "$GPU_IDX" "$GPU_END")
  DECODER_LOG="${LOG_DIR}/build_${BUILD_ID}_decoder_${port}.log"
  : > "${DECODER_LOG}"
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
      "${CACHEBLEND_KV_TRANSFER_CONFIG}" \
    >>"${DECODER_LOG}" 2>&1 &
  TRACKED_PIDS+=($!)
  GPU_IDX=$((GPU_IDX + TENSOR_PARALLEL))
done

# ---------------------------------------------------------------------------
# 4. Wait for all vLLM instances to be ready
# ---------------------------------------------------------------------------
for port in "${PREFILLER_PORTS[@]}"; do
  PREFILLER_LOG="${LOG_DIR}/build_${BUILD_ID}_prefiller_${port}.log"
  if ! wait_for_server "$port" "$SERVER_WAIT_TIMEOUT" "$PREFILLER_LOG"; then
    echo "ERROR: Prefiller vLLM on port ${port} did not become ready."
    exit 1
  fi
done
for port in "${DECODER_PORTS[@]}"; do
  DECODER_LOG="${LOG_DIR}/build_${BUILD_ID}_decoder_${port}.log"
  if ! wait_for_server "$port" "$SERVER_WAIT_TIMEOUT" "$DECODER_LOG"; then
    echo "ERROR: Decoder vLLM on port ${port} did not become ready."
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 5. Start the CacheBlend proxy
# ---------------------------------------------------------------------------


"${TEST_PYTHON}" "${SCRIPT_DIR}/proxy.py" \
  --port "$SERVICE_PORT" \
  --prefiller-host localhost --prefiller-port "$PREFILLER_PORT" \
  --decoder-host localhost --decoder-port "$DECODER_PORT" \
  --telemetry-port "$TELEMETRY_PORT" >>"${VLLM_LOG}" 2>&1 &
TRACKED_PIDS+=($!)

# ---------------------------------------------------------------------------
# 6. Benchmark (with timeout) + log error gate
# ---------------------------------------------------------------------------
set +e
timeout "${BENCHMARK_TIMEOUT_SEC}" \
  "${TEST_PYTHON}" benchmarks/multi_doc_qa/shuffle_doc_qa.py \
  --num-documents "${SHUFFLE_NUM_DOCUMENTS}" \
  --document-length "${SHUFFLE_DOCUMENT_LENGTH}" \
  --output-len "${SHUFFLE_OUTPUT_LEN}" \
  2>&1 | tee -a "${BENCHMARK_LOG}"
benchmark_rc=${PIPESTATUS[0]}
set -e
if [[ "$benchmark_rc" -ne 0 ]]; then
  if [[ "$benchmark_rc" -eq 124 ]]; then
    echo "[FAIL] shuffle_doc_qa exceeded BENCHMARK_TIMEOUT_SEC=${BENCHMARK_TIMEOUT_SEC}s"
  else
    echo "[FAIL] shuffle_doc_qa exited with code ${benchmark_rc}"
  fi
  exit 1
fi

echo "[PASS] shuffle_doc_qa benchmark exited 0"

if command -v curl >/dev/null 2>&1; then
  curl -sf "http://localhost:${LMCACHE_HTTP_PORT}/status" >"${LOG_DIR}/lmcache-status-final.json" 2>/dev/null || true
fi

check_build_logs_for_errors
"${SCRIPT_DIR}/validate-blend-logs.sh" "${LOG_DIR}" "${BUILD_ID}"

echo "[PASS] Blend integration test completed successfully with CacheBlend V2 evidence."
exit 0
