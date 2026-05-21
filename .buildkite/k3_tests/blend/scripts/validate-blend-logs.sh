#!/usr/bin/env bash
# Validate that the Blend Buildkite E2E exercised CacheBlend V2, not just
# ordinary LMCache MP traffic.

set -euo pipefail

LOG_DIR="${1:-}"
BUILD_ID="${2:-}"

if [[ "${LOG_DIR}" == "--self-test" ]]; then
  SELF_TEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-validate-blend-logs.sh"
  if [[ ! -x "${SELF_TEST}" ]]; then
    echo "[FAIL] self-test helper is missing or not executable: ${SELF_TEST}" >&2
    exit 1
  fi
  exec "${SELF_TEST}"
fi

if [[ -z "${LOG_DIR}" ]]; then
  echo "usage: $0 <log-dir> [build-id] | --self-test" >&2
  exit 2
fi

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "[FAIL] log directory does not exist: ${LOG_DIR}" >&2
  exit 1
fi

shopt -s nullglob
LOGS=("${LOG_DIR}"/*.log)
shopt -u nullglob

if [[ ${#LOGS[@]} -eq 0 ]]; then
  echo "[FAIL] no .log files found in ${LOG_DIR}" >&2
  exit 1
fi

count_pattern() {
  local pattern="$1"
  (sanitized_logs | grep -E "${pattern}" || true) | wc -l | tr -d ' '
}

show_matches() {
  local pattern="$1"
  local max_lines="${2:-20}"
  # Report matches from the same sanitized stream used by validators. The
  # main blend log is produced with `set -x`, so raw grep would otherwise
  # self-match shell xtrace lines that merely contain these regex literals.
  sanitized_logs | grep -nE "${pattern}" | head -"${max_lines}" || true
}

sanitized_logs() {
  grep -h '' "${LOGS[@]}" 2>/dev/null \
    | grep -v '^+ ' \
    | grep -v '^++ ' \
    | grep -v '^+++' \
    | grep -viE '^\[PASS\] No error/traceback/fatal pattern in build logs:' \
    | grep -viE '^\[PASS\] No fatal/runtime pattern in build logs:' \
    | grep -viE '^\[PASS\] CacheBlend E2E log validation passed' \
    || true
}

fail_with_matches() {
  local message="$1"
  local pattern="${2:-}"
  echo "[FAIL] ${message}" >&2
  if [[ -n "${pattern}" ]]; then
    show_matches "${pattern}" 80 >&2
  fi
  exit 1
}

# Runtime failures that make the benchmark result non-actionable. Keep this
# stricter than run-blend-test.sh because this script runs after the workload.
FATAL_PATTERN='Traceback|\bfatal\b|CUDA error|NCCL.*(error|fail)|ZMQ.*timeout|HTTP/1\.1" 5|status_code=5|Internal server error|EngineDeadError|engine process failed|benchmark.*timeout|timed out waiting for telemetry|request.*exception|RuntimeError|process died unexpectedly|exited with code [1-9]'
if sanitized_logs | grep -iE "${FATAL_PATTERN}" >/dev/null; then
  fail_with_matches "fatal/runtime error pattern found in logs" "${FATAL_PATTERN}"
fi

# Server/adapter startup proof. Require the production-ish CLI path, not only a
# legacy direct-module server, then require BlendEngineV2 evidence.
if ! grep -hE 'lmcache"? server .*--engine-type blend|lmcache server .*--engine-type blend|LMCache server:.*engine-type=blend' "${LOGS[@]}" >/dev/null 2>&1; then
  fail_with_matches "no evidence that the run used 'lmcache server --engine-type blend'" 'lmcache.*server.*--engine-type blend|LMCache server:.*engine-type=blend'
fi

show_matches 'LMCache cache blend v2 server is running|engine_type.?=.?(blend|BlendEngineV2)|BlendEngineV2' 20 >/tmp/blend_startup_matches.$$ || true
if [[ ! -s /tmp/blend_startup_matches.$$ ]]; then
  rm -f /tmp/blend_startup_matches.$$
  fail_with_matches "no evidence that the LMCache blend server / BlendEngineV2 started" 'LMCache cache blend v2 server is running|engine_type.?=.?(blend|BlendEngineV2)|BlendEngineV2'
fi
rm -f /tmp/blend_startup_matches.$$

if ! grep -hE 'enable_cacheblend=True|enable_cacheblend=true' "${LOGS[@]}" >/dev/null 2>&1; then
  fail_with_matches "vLLM adapter logs never showed enable_cacheblend=True" 'enable_cacheblend=(True|true|False|false)'
fi

# CacheBlend V2 protocol proof. The exact logs can appear either as enum names
# from the adapter or as server-side operation messages.
REGISTER_COUNT="$(count_pattern 'CB_REGISTER_KV_CACHE|Registered CB KV cache')"
STORE_PRE_COUNT="$(count_pattern 'CB_STORE_PRE_COMPUTED|Stored pre-computed doc')"
LOOKUP_COUNT="$(count_pattern 'CB_LOOKUP_PRE_COMPUTED_V2|CB_LOOKUP_PRE_COMPUTED|LMCache MP lookup request:.*request_type=CB_LOOKUP')"
RETRIEVE_COUNT="$(count_pattern 'CB_RETRIEVE_PRE_COMPUTED_V2|Retrieved pre-computed')"
STORE_FINAL_COUNT="$(count_pattern 'CB_STORE_FINAL|Stored final doc|cacheblend_store_final=True|cacheblend_store_final=true')"

[[ "${REGISTER_COUNT}" -gt 0 ]] || fail_with_matches "no CacheBlend KV registration evidence" 'CB_REGISTER_KV_CACHE|Registered CB KV cache'
[[ "${STORE_PRE_COUNT}" -gt 0 ]] || fail_with_matches "no CacheBlend precomputed-store evidence" 'CB_STORE_PRE_COMPUTED|Stored pre-computed doc'
[[ "${LOOKUP_COUNT}" -gt 0 ]] || fail_with_matches "no CacheBlend lookup evidence" 'CB_LOOKUP_PRE_COMPUTED_V2|CB_LOOKUP_PRE_COMPUTED|request_type=CB_LOOKUP'
[[ "${RETRIEVE_COUNT}" -gt 0 ]] || fail_with_matches "no CacheBlend retrieve evidence" 'CB_RETRIEVE_PRE_COMPUTED_V2|Retrieved pre-computed'

# Final-store is workload-dependent. Some derangement workloads retrieve all
# prompt chunks from pre-computed documents, so there may be no newly completed
# prompt chunk to store via CB_STORE_FINAL. Treat final-store as additional
# evidence when present, but do not fail a run that has non-empty retrieve/hit
# evidence plus proxy decode completion and benchmark exit-0 evidence below.

# Non-empty hit/match evidence. This is what separates true CacheBlend reuse
# from merely starting a blend server and sending ordinary MP requests.
HIT_PATTERN='Retrieved pre-computed for [1-9][0-9]* match results|CBMatchResult\(|cb_match_result=\[[^]]|num_lmcache_hit_blocks=[1-9]|lookup_hit_tokens_total[^0-9]*[1-9]|fingerprint_hits_total[^0-9]*[1-9]|storage_hits_total[^0-9]*[1-9]|cacheblend_(match|hit)[^0-9]*[1-9]'
if ! sanitized_logs | grep -E "${HIT_PATTERN}" >/dev/null 2>&1; then
  fail_with_matches "no non-empty CacheBlend match/hit evidence found; run may not prove non-prefix KV reuse" "${HIT_PATTERN}"
fi

# Proxy / telemetry proof. This catches cases where the benchmark completes
# against one side but the prefill->decode handoff did not actually happen.
SAVE_PATTERN='Request [0-9a-f-]+: finished saving KV caches after prefill|All [0-9]+ TP worker\(s\) done for request:|Request [0-9a-f-]+: telemetry wait timed out after [0-9.]+s; fallback enabled; forwarding request to decoder after prefill response'
FORWARD_PATTERN='Request [0-9a-f-]+: forwarding request to decoder'
DECODE_PATTERN='Request [0-9a-f-]+: (streaming )?decode response completed'
if ! sanitized_logs | grep -E "${SAVE_PATTERN}" >/dev/null 2>&1; then
  fail_with_matches "no request-level prefiller/telemetry save evidence found" "${SAVE_PATTERN}"
fi
if ! sanitized_logs | grep -E "${FORWARD_PATTERN}" >/dev/null 2>&1; then
  fail_with_matches "no request-level proxy-to-decoder forwarding evidence found" "${FORWARD_PATTERN}"
fi
if ! sanitized_logs | grep -E "${DECODE_PATTERN}" >/dev/null 2>&1; then
  fail_with_matches "no request-level decoder completion evidence found" "${DECODE_PATTERN}"
fi

if ! sanitized_logs | grep -E '\[PASS\] shuffle_doc_qa benchmark exited 0|benchmark exit(ed)? 0|benchmark.*exit_code=0' >/dev/null 2>&1; then
  fail_with_matches "no benchmark exit-0 evidence found" '\[PASS\] shuffle_doc_qa benchmark exited 0|benchmark exit(ed)? 0|benchmark.*exit_code=0'
fi

cat <<EOF
[PASS] CacheBlend E2E log validation passed${BUILD_ID:+ for ${BUILD_ID}}
  logs: ${LOG_DIR}
  CB register evidence:        ${REGISTER_COUNT}
  CB precomputed-store evidence:${STORE_PRE_COUNT}
  CB lookup evidence:          ${LOOKUP_COUNT}
  CB retrieve evidence:        ${RETRIEVE_COUNT}
  CB final-store evidence:     ${STORE_FINAL_COUNT}
EOF
