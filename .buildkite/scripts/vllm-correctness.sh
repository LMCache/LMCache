#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_ID:?BUILD_ID must be set}"

PORT=8000
MODEL="meta-llama/Llama-3.2-1B-Instruct"

LOGFILE="/tmp/build_${BUILD_ID}_correctness.log"

# Capture *all* stdout/stderr to console + logfile
exec > >(tee -a "$LOGFILE") 2>&1

echo "[INFO] Build ID: $BUILD_ID"
echo "[INFO] Logfile: $LOGFILE"

# ---- Build contexts ----

CONTEXT="$(
  man bash \
  | col -b \
  | tr -s '[:space:]' ' ' \
  | awk '{
      for (i = 1; i <= NF; i++) {
        printf "%s ", $i
        if (++c == 5000) exit
      }
    }'
)"

HALF_CONTEXT="$(
  man bash \
  | col -b \
  | tr -s '[:space:]' ' ' \
  | awk '{
      for (i = 1; i <= NF; i++) {
        printf "%s ", $i
        if (++c == 2500) exit
      }
    }'
)"

# ---- Helper to send a request ----

send_completion() {
  local content="$1"

  curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg model "$MODEL" \
      --arg content "$content" \
      --argjson max_tokens 100 \
      '{
        model: $model,
        temperature: 0,
        max_tokens: $max_tokens,
        messages: [
          { role: "user", content: $content }
        ]
      }'
    )" \
  | jq -r '.choices[0].message.content'
}

# ---- Test flow ----

echo "[STEP 1] Full CONTEXT (initial population)"
RESULT_1="$(send_completion "$CONTEXT")"

echo "[STEP 2] Resetting prefix cache"
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null

echo "[STEP 3] HALF_CONTEXT (APC only)"
send_completion "$HALF_CONTEXT" >/dev/null

echo "[STEP 4] Full CONTEXT again (APC + LMCache)"
RESULT_4="$(send_completion "$CONTEXT")"

echo "[STEP 5] Strict equality check"

if [[ "$RESULT_1" != "$RESULT_4" ]]; then
  echo "[FAIL] Mismatch between initial and cached results"
  echo "----- RESULT 1 -----"
  printf '%s\n' "$RESULT_1"
  echo "----- RESULT 4 -----"
  printf '%s\n' "$RESULT_4"
  exit 1
fi

echo "[PASS] Results are strictly identical"
