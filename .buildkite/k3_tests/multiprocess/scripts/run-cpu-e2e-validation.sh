#!/usr/bin/env bash
set -euo pipefail

echo "Build ID: ${BUILDKITE_BUILD_ID:-local}"
echo "Python: $(python3 --version 2>&1 || true)"
echo "uv: $(uv --version 2>&1 || true)"

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
VENV_DIR=".venv-${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache_cpu_validation.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm_cpu_validation.log"
LMCACHE_PID=""
VLLM_PID=""
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-2}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-30}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-120}"
# Set LMCACHE_SHM_NAME="" to use pickle transport; unset/default uses shm transport
LMCACHE_SHM_NAME="${LMCACHE_SHM_NAME-__default__}"

# Directory to collect artifacts before workspace is deleted
ARTIFACT_DIR="/tmp/build_${BUILD_ID}_artifacts"
mkdir -p "${ARTIFACT_DIR}"

upload_artifacts() {
  # Copy logs to artifact dir (which survives workspace deletion)
  cp -f "${LMCACHE_LOG}" "${ARTIFACT_DIR}/lmcache_cpu_validation.log" 2>/dev/null || true
  cp -f "${VLLM_LOG}" "${ARTIFACT_DIR}/vllm_cpu_validation.log" 2>/dev/null || true

  if [ -n "${BUILDKITE_BUILD_ID:-}" ] && command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent artifact upload "${ARTIFACT_DIR}/*.log" || true
  fi
}

cleanup_workspace() {
  if [ -n "${BUILDKITE_BUILD_ID:-}" ]; then
    export TARGET="$PWD"
    case "$TARGET" in
      ""|"/"|"/usr"|"/var"|"/etc"|"/bin"|"/sbin"|"/opt"|"/home"|"/tmp")
        echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
        return 1
        ;;
    esac
    if [ "$TARGET" = "$HOME" ]; then
      echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
      return 1
    fi
    if [ ! -d "$TARGET/.git" ] || [ ! -f "$TARGET/pyproject.toml" ]; then
      echo "❌ Refusing to delete unexpected workspace path: $TARGET"
      return 1
    fi
    echo "Deleting current workspace $TARGET"
    cd /
    if command -v sudo >/dev/null 2>&1; then
      sudo rm -rf "$TARGET"
    else
      rm -rf "$TARGET"
    fi
  fi
}

print_failure_logs() {
  echo "=== LMCache Server Log (${LMCACHE_LOG}) ==="
  if [ -f "${LMCACHE_LOG}" ]; then
    tail -n 200 "${LMCACHE_LOG}" || true
  else
    echo "Log not found"
  fi
  echo "=== vLLM Log (${VLLM_LOG}) ==="
  if [ -f "${VLLM_LOG}" ]; then
    tail -n 200 "${VLLM_LOG}" || true
  else
    echo "Log not found"
  fi
}

cleanup_processes() {
  set +e
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [ -n "${LMCACHE_PID}" ] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    echo "Stopping LMCache server (PID=${LMCACHE_PID})"
    kill "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
  fi
  set -e
}

wait_for_endpoint_contains() {
  local url="$1"
  local timeout="$2"
  local expected="$3"
  local label="$4"
  local response

  for _ in $(seq 1 "${timeout}"); do
    if response="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [ -z "${expected}" ] || echo "${response}" | grep -q "${expected}"; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "❌ ${label} did not become ready within ${timeout}s"
  return 1
}

# Scrape a Prometheus counter value from LMCache /metrics endpoint
scrape_metric() {
  local metric_name="$1"
  python3 - <<EOF
import sys, urllib.request
url = "http://localhost:${LMCACHE_HTTP_PORT}/metrics"
try:
    body = urllib.request.urlopen(url, timeout=10).read().decode()
except Exception as e:
    print(f"ERROR fetching {url}: {e}", file=sys.stderr)
    print("0")
    sys.exit(0)
total = 0.0
for line in body.splitlines():
    if line.startswith("#"):
        continue
    if not line.startswith("${metric_name}"):
        continue
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        continue
    try:
        total += float(parts[1])
    except ValueError:
        continue
print(int(total))
EOF
}

# Send a completion request and print the text output
send_completion() {
  local prompt_file="$1"
  local max_tokens="${2:-50}"
  local prompt
  prompt="$(cat "${prompt_file}")"
  local response
  response="$(curl -fsS "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json, sys
prompt = open('${prompt_file}').read()
print(json.dumps({
    'model': 'facebook/opt-125m',
    'prompt': prompt,
    'max_tokens': ${max_tokens},
    'temperature': 0
}))
")")"
  echo "${response}" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"
}

start_vllm() {
  echo "Starting vLLM server..."
  VLLM_TARGET_DEVICE=cpu vllm serve facebook/opt-125m \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM server started (PID=${VLLM_PID})"
  sleep 1
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "❌ vLLM server exited immediately after startup. See ${VLLM_LOG} for details"
    return 1
  fi
  echo "Waiting for vLLM readiness at http://localhost:${VLLM_PORT}/v1/models (timeout: ${VLLM_READY_TIMEOUT}s)"
  if ! wait_for_endpoint_contains "http://localhost:${VLLM_PORT}/v1/models" "${VLLM_READY_TIMEOUT}" "facebook/opt-125m" "vLLM server"; then
    return 1
  fi
  echo "✅ vLLM server is ready"
}

stop_vllm() {
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR
  echo "❌ CPU install validation failed (exit code: ${exit_code})"
  set +e
  print_failure_logs
  cleanup_processes
  upload_artifacts
  cleanup_workspace || echo "❌ Workspace cleanup failed"
  set -e
  exit "$exit_code"
}

trap on_error ERR

echo "=== CPU Install Validation (Phase 1) ==="
echo "Creating virtual environment with uv at ${VENV_DIR}"
uv venv --python 3.12 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo "✅ Virtual environment ready"

echo "Upgrading pip/setuptools/wheel"
uv pip install --upgrade pip setuptools wheel
echo "✅ Upgraded pip/setuptools/wheel"

echo "Installing build dependencies from requirements/build.txt"
uv pip install -r requirements/build.txt
echo "✅ Installed requirements/build.txt"

echo "Installing common dependencies from requirements/common.txt"
uv pip install -r requirements/common.txt
echo "✅ Installed requirements/common.txt"

echo "Installing vLLM CPU build"
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
echo "✅ vLLM CPU install completed"

echo "Installing LMCache in editable mode with NO_GPU_EXT=1"
NO_GPU_EXT=1 uv pip install -e . --no-build-isolation
echo "✅ LMCache install completed"

echo "Freezing installed package versions"
uv pip freeze

echo "Validating imports"
python -c "import lmcache; import vllm; print('✅ Imports OK')"

echo "Printing package versions"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import lmcache; print('lmcache:', lmcache.__version__)"

echo "✅ CPU install validation passed"

echo "=== CPU E2E Validation (Phase 2) ==="

echo "[Phase 2 / Step 1] Installing numpy<2 for scipy/vLLM compatibility"
uv pip install "numpy<2"
echo "✅ numpy<2 installed"

echo "[Phase 2 / Step 2] Downloading facebook/opt-125m model (cache-aware)"
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')"; then
  echo "❌ Failed to download/cache facebook/opt-125m"
  false
fi
echo "✅ Model download/check complete"

echo "[Phase 2 / Step 3] Starting LMCache server"
echo "LMCache log: ${LMCACHE_LOG}"
# Build lmcache server args
LMCACHE_ARGS=(
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}"
  --eviction-policy "${LMCACHE_EVICTION_POLICY}"
  --chunk-size "${LMCACHE_CHUNK_SIZE}"
)
if [ "${LMCACHE_SHM_NAME}" = "__default__" ]; then
  echo "Transport mode: shared memory (shm)"
  EXPECTED_TRANSPORT="shm"
else
  echo "Transport mode: pickle (--shm-name '${LMCACHE_SHM_NAME}')"
  LMCACHE_ARGS+=(--shm-name "${LMCACHE_SHM_NAME}")
  EXPECTED_TRANSPORT="pickle"
fi

lmcache server "${LMCACHE_ARGS[@]}" \
  >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
echo "LMCache server started (PID=${LMCACHE_PID})"
sleep 1
if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
  echo "❌ LMCache server exited immediately after startup. See ${LMCACHE_LOG} for details"
  false
fi

echo "Waiting for LMCache healthcheck at http://localhost:${LMCACHE_HTTP_PORT}/healthcheck (timeout: ${LMCACHE_HEALTHCHECK_TIMEOUT}s)"
if ! wait_for_endpoint_contains "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" "${LMCACHE_HEALTHCHECK_TIMEOUT}" "" "LMCache server"; then
  false
fi
echo "✅ LMCache server is healthy"

echo "[Phase 2 / Step 4] Installing libnuma and starting vLLM server"
apt-get update && apt-get install -y --no-install-recommends libnuma1
export VLLM_TARGET_DEVICE=cpu
start_vllm

echo "[Phase 2 / Step 5] Sending E2E test request"
completion_response="$(curl -fsS "http://localhost:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"facebook/opt-125m","prompt":"Hello","max_tokens":5}')"
echo "Completion response: ${completion_response}"
if ! echo "${completion_response}" | grep -q '"choices"'; then
  echo "❌ E2E request response failed structural validation"
  false
fi
if ! echo "${completion_response}" | grep -q "facebook/opt-125m"; then
  echo "❌ E2E request response missing expected model"
  false
fi
echo "✅ E2E request validation passed"

# Verify transport mode (logged after vLLM connects to LMCache server)
echo "[Phase 2 / Step 5.5] Verifying transport mode: expecting '${EXPECTED_TRANSPORT}'"
if [ "${EXPECTED_TRANSPORT}" = "shm" ]; then
  if ! grep -q "Using shm" "${LMCACHE_LOG}" 2>/dev/null; then
    echo "❌ Expected shm transport but 'Using shm' not found in log"
    tail -50 "${LMCACHE_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: shm"
elif [ "${EXPECTED_TRANSPORT}" = "pickle" ]; then
  if ! grep -q "Using pickle" "${LMCACHE_LOG}" 2>/dev/null; then
    echo "❌ Expected pickle transport but 'Using pickle' not found in log"
    tail -50 "${LMCACHE_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: pickle"
fi

echo "[Phase 2 / Step 6] Cleaning up Phase 2 vLLM"
stop_vllm
echo "✅ Phase 2 cleanup completed"

echo "✅ CPU E2E validation passed"

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Cache Hit Validation
# ═══════════════════════════════════════════════════════════════════
# Scenario:
#   - LMCache server stays running the entire time
#   - vLLM instance 1: request A → LMCache store; request A again → LMCache hit
#   - vLLM restart (instance 2): request A → LMCache hit (cross-instance)
#   - All three outputs must be identical (bit-exact with temperature=0)
# ═══════════════════════════════════════════════════════════════════

echo "=== Cache Hit Validation (Phase 3) ==="

# Generate a fixed ~1000 token prompt
PROMPT_FILE="/tmp/build_${BUILD_ID}_phase3_prompt.txt"
python3 -c "
# A diverse, non-repetitive story prompt (~1000 tokens for opt-125m).
# The model will continue the narrative; correctness doesn't matter,
# only that all three runs produce identical output (cache consistency).
story = '''Once upon a time in a small coastal village, there lived an old lighthouse keeper named Thomas. Every evening, he climbed the one hundred and thirty-seven steps to the top of the lighthouse to light the great lamp. The sea was unpredictable in those parts. Ships from distant lands carried spices, silk, and stories of places Thomas had never seen. One stormy night in November, a merchant vessel called the Silver Heron appeared on the horizon, listing dangerously to starboard. Thomas watched through his brass telescope as the waves crashed against its hull. He knew that if the ship did not change course within the next ten minutes, it would strike the jagged rocks known locally as the Devil's Teeth. He grabbed the emergency flare gun from the wooden cabinet and fired three red flares into the sky. The captain of the Silver Heron saw the warning and ordered hard to port. The ship groaned as it turned, barely clearing the outermost rock by twenty meters. The next morning, the captain rowed ashore to thank Thomas personally. He brought a gift: a small wooden box containing a compass that always pointed not north, but toward home. Thomas kept it on his desk for the rest of his days. Years later, when Thomas retired, he passed the compass to his granddaughter Elena, who had inherited his love of the sea. Elena became a marine biologist studying the migration patterns of humpback whales along the Pacific coast. She traveled from Alaska to Mexico following the whale pods, documenting their songs and social behaviors. Her research revealed that whale families maintained bonds across thousands of miles, communicating through low-frequency calls that could travel entire ocean basins. One afternoon while diving near a coral reef off the coast of Baja California, Elena discovered something extraordinary beneath a rocky overhang:'''
print(story, end='')
" > "${PROMPT_FILE}"
echo "Generated prompt file: ${PROMPT_FILE} ($(wc -c < "${PROMPT_FILE}") bytes)"

# Reset metrics to have a clean baseline
echo "[Phase 3 / Step 1] Resetting LMCache metrics"
curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT}/metrics/reset" >/dev/null
echo "✅ Metrics reset"

# Start vLLM instance 1
echo "[Phase 3 / Step 2] Starting vLLM (instance 1)"
start_vllm

# Request A (first time) → should trigger store
echo "[Phase 3 / Step 3] Request A (first) — expecting LMCache store"
L1_WRITE_BEFORE=$(scrape_metric "lmcache_mp_l1_write_chunks_total")
OUTPUT_1=$(send_completion "${PROMPT_FILE}" 200)
echo "Output 1: ${OUTPUT_1}"
sleep 2  # allow async store to complete
L1_WRITE_AFTER=$(scrape_metric "lmcache_mp_l1_write_chunks_total")
STORE_DELTA=$((L1_WRITE_AFTER - L1_WRITE_BEFORE))
echo "L1 write chunks delta: ${STORE_DELTA}"
if [ "${STORE_DELTA}" -lt 1 ]; then
  echo "❌ No L1 write activity after first request (expected store)"
  false
fi
echo "✅ LMCache store verified (${STORE_DELTA} chunks written)"

# Request A (second time, same vLLM instance) → should trigger read/hit
echo "[Phase 3 / Step 4] Request A (second) — expecting LMCache hit"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_2=$(send_completion "${PROMPT_FILE}" 200)
echo "Output 2: ${OUTPUT_2}"
sleep 2
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta: ${READ_DELTA}"
if [ "${READ_DELTA}" -lt 1 ]; then
  echo "❌ No L1 read activity on second request (expected cache hit)"
  false
fi
echo "✅ LMCache hit verified on same instance (${READ_DELTA} chunks read)"

# Restart vLLM
echo "[Phase 3 / Step 5] Restarting vLLM (instance 2)"
stop_vllm
sleep 2
start_vllm

# Request A (third time, new vLLM instance) → should trigger read/hit from LMCache
echo "[Phase 3 / Step 6] Request A (third) — expecting LMCache hit after vLLM restart"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_3=$(send_completion "${PROMPT_FILE}" 200)
echo "Output 3: ${OUTPUT_3}"
sleep 2
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta: ${READ_DELTA}"
if [ "${READ_DELTA}" -lt 1 ]; then
  echo "❌ No L1 read activity after vLLM restart (expected cross-instance cache hit)"
  false
fi
echo "✅ LMCache cross-instance hit verified (${READ_DELTA} chunks read)"

# Verify all three outputs are identical
echo "[Phase 3 / Step 7] Verifying output consistency"
if [ "${OUTPUT_1}" != "${OUTPUT_2}" ]; then
  echo "❌ Output mismatch between request 1 and request 2"
  echo "  Output 1: ${OUTPUT_1}"
  echo "  Output 2: ${OUTPUT_2}"
  false
fi
if [ "${OUTPUT_1}" != "${OUTPUT_3}" ]; then
  echo "❌ Output mismatch between request 1 and request 3 (after vLLM restart)"
  echo "  Output 1: ${OUTPUT_1}"
  echo "  Output 3: ${OUTPUT_3}"
  false
fi
echo "✅ All three outputs are identical — cache does not alter inference results"

# Negative test: a completely different prompt should NOT hit the cache
echo "[Phase 3 / Step 8] Request B (different prompt) — expecting cache MISS"
PROMPT_FILE_B="/tmp/build_${BUILD_ID}_phase3_prompt_b.txt"
python3 -c "
# A completely different prompt that shares no prefix with prompt A
story_b = '''In the year 2147, humanity established its first permanent colony on Mars. The settlement, named Arcadia, housed three thousand researchers and engineers working to terraform the red planet. Chief botanist Dr. Yuki Tanaka spent her days in the greenhouse domes, cultivating genetically modified crops that could thrive in Martian soil. The atmospheric processors hummed day and night, slowly converting carbon dioxide into breathable oxygen. It was tedious work measured in decades, but the colonists were patient. Every morning, Dr. Tanaka checked her instruments and recorded the oxygen levels in her logbook. Today the readings showed'''
print(story_b, end='')
" > "${PROMPT_FILE_B}"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_B=$(send_completion "${PROMPT_FILE_B}" 200)
echo "Output B: ${OUTPUT_B}"
wait_for_metric_change "lmcache_mp_l1_read_chunks_total" "${L1_READ_BEFORE}" 5 || true
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta for new prompt: ${READ_DELTA}"
if [ "${READ_DELTA}" -gt 0 ]; then
  echo "❌ Unexpected cache hit on a completely different prompt — metrics may be unreliable"
  false
fi
echo "✅ Cache miss confirmed for different prompt — metrics are trustworthy"

echo "[Phase 3 / Step 9] Cleaning up"
stop_vllm
cleanup_processes
echo "✅ Phase 3 cleanup completed"

echo ""
echo "=========================================="
echo "✅ All phases passed (Phase 1 + 2 + 3)"
echo "=========================================="

# Upload artifacts BEFORE deleting the workspace
upload_artifacts
cleanup_workspace
