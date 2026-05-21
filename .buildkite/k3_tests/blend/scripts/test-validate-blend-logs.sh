#!/usr/bin/env bash
# Synthetic pass/fail fixtures for validate-blend-logs.sh. These keep the
# production E2E validator honest without requiring GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SCRIPT_DIR}/validate-blend-logs.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

write_good_fixture() {
  local dir="$1"
  mkdir -p "${dir}"
  cat >"${dir}/build_fixture_blend.log" <<'LOG'
+ /workspace/.venv/bin/lmcache server --engine-type blend --host localhost --port 6566
Configuration: 1P1D (TP=1)
LMCache server:  cli (engine-type=blend)
[PASS] shuffle_doc_qa benchmark exited 0
LOG
  cat >"${dir}/build_fixture_blend_server.log" <<'LOG'
LMCache cache blend v2 server is running with engine_type=blend
BlendEngineV2 ready
CB_REGISTER_KV_CACHE gpu_id=0
CB_STORE_PRE_COMPUTED doc_id=doc-1
CB_LOOKUP_PRE_COMPUTED_V2 request_id=req-1
CB_RETRIEVE_PRE_COMPUTED_V2 request_id=req-1
CB_STORE_FINAL request_id=req-1
Retrieved pre-computed for 2 match results
LOG
  cat >"${dir}/build_fixture_prefiller_8100.log" <<'LOG'
LMCacheMPConnector initialized enable_cacheblend=True
LOG
  cat >"${dir}/build_fixture_proxy.log" <<'LOG'
Request 123e4567-e89b-12d3-a456-426614174000: finished saving KV caches after prefill
Request 123e4567-e89b-12d3-a456-426614174000: forwarding request to decoder
Request 123e4567-e89b-12d3-a456-426614174000: streaming decode response completed
LOG
  cat >"${dir}/build_fixture_benchmark.log" <<'LOG'
benchmark exit 0
LOG
}

expect_pass() {
  local name="$1"
  local dir="${TMP_ROOT}/${name}"
  write_good_fixture "${dir}"
  "${VALIDATOR}" "${dir}" "${name}" >/tmp/${name}.out
  echo "[PASS] validator accepted ${name}"
}

expect_fail() {
  local name="$1"
  local remove_pattern="$2"
  local dir="${TMP_ROOT}/${name}"
  write_good_fixture "${dir}"
  perl -0pi -e "s/${remove_pattern}//g" "${dir}"/*.log
  if "${VALIDATOR}" "${dir}" "${name}" >/tmp/${name}.out 2>&1; then
    echo "[FAIL] validator unexpectedly accepted ${name}" >&2
    cat /tmp/${name}.out >&2
    exit 1
  fi
  echo "[PASS] validator rejected ${name}"
}

expect_pass positive-cacheblend-v2

fallback_dir="${TMP_ROOT}/positive-telemetry-fallback"
write_good_fixture "${fallback_dir}"
cat >"${fallback_dir}/build_fixture_proxy.log" <<'LOG'
Request 123e4567-e89b-12d3-a456-426614174000: telemetry wait timed out after 120.0s; fallback enabled; forwarding request to decoder after prefill response
Request 123e4567-e89b-12d3-a456-426614174000: forwarding request to decoder
Request 123e4567-e89b-12d3-a456-426614174000: streaming decode response completed
LOG
"${VALIDATOR}" "${fallback_dir}" positive-telemetry-fallback >/tmp/positive-telemetry-fallback.out
echo "[PASS] validator accepted positive-telemetry-fallback"


model_text_dir="${TMP_ROOT}/positive-model-text-error-word"
write_good_fixture "${model_text_dir}"
cat >>"${model_text_dir}/build_fixture_benchmark.log" <<'LOG'
assistant output: This is a formatting error in the generated answer, not infra.
LOG
"${VALIDATOR}" "${model_text_dir}" positive-model-text-error-word >/tmp/positive-model-text-error-word.out
echo "[PASS] validator ignored model-generated error wording"

runtime_error_dir="${TMP_ROOT}/negative-runtime-error"
write_good_fixture "${runtime_error_dir}"
echo 'RuntimeError: engine process failed' >>"${runtime_error_dir}/build_fixture_proxy.log"
if "${VALIDATOR}" "${runtime_error_dir}" negative-runtime-error >/tmp/negative-runtime-error.out 2>&1; then
  echo "[FAIL] validator unexpectedly accepted negative-runtime-error" >&2
  cat /tmp/negative-runtime-error.out >&2
  exit 1
fi
echo "[PASS] validator rejected negative-runtime-error"

expect_fail negative-no-register 'CB_REGISTER_KV_CACHE gpu_id=0\n'
expect_fail negative-ordinary-mp 'CB_STORE_PRE_COMPUTED doc_id=doc-1\nCB_LOOKUP_PRE_COMPUTED_V2 request_id=req-1\nCB_RETRIEVE_PRE_COMPUTED_V2 request_id=req-1\nCB_STORE_FINAL request_id=req-1\nRetrieved pre-computed for 2 match results\n'
expect_fail negative-no-hit 'Retrieved pre-computed for 2 match results\n'

no_retrieve_dir="${TMP_ROOT}/negative-no-retrieve"
write_good_fixture "${no_retrieve_dir}"
perl -0pi -e 's/CB_RETRIEVE_PRE_COMPUTED_V2 request_id=req-1\n//g; s/Retrieved pre-computed for 2 match results\n//g' "${no_retrieve_dir}"/*.log
if "${VALIDATOR}" "${no_retrieve_dir}" negative-no-retrieve >/tmp/negative-no-retrieve.out 2>&1; then
  echo "[FAIL] validator unexpectedly accepted negative-no-retrieve" >&2
  cat /tmp/negative-no-retrieve.out >&2
  exit 1
fi
echo "[PASS] validator rejected negative-no-retrieve"

retrieve_only_dir="${TMP_ROOT}/positive-retrieve-only"
write_good_fixture "${retrieve_only_dir}"
perl -0pi -e 's/CB_STORE_FINAL request_id=req-1\n//g' "${retrieve_only_dir}"/*.log
"${VALIDATOR}" "${retrieve_only_dir}" positive-retrieve-only >/tmp/positive-retrieve-only.out
echo "[PASS] validator accepted positive-retrieve-only"

no_benchmark_dir="${TMP_ROOT}/negative-no-benchmark-exit0"
write_good_fixture "${no_benchmark_dir}"
perl -0pi -e 's/\[PASS\] shuffle_doc_qa benchmark exited 0\n//g; s/benchmark exit 0\n//g' "${no_benchmark_dir}"/*.log
if "${VALIDATOR}" "${no_benchmark_dir}" negative-no-benchmark-exit0 >/tmp/negative-no-benchmark-exit0.out 2>&1; then
  echo "[FAIL] validator unexpectedly accepted negative-no-benchmark-exit0" >&2
  cat /tmp/negative-no-benchmark-exit0.out >&2
  exit 1
fi
echo "[PASS] validator rejected negative-no-benchmark-exit0"

echo "[PASS] synthetic CacheBlend validator fixtures completed"
