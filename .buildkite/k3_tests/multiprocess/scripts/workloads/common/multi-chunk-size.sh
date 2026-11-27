#!/usr/bin/env bash
# Verify that an inference-engine client can select a chunk size larger than
# the LMCache server default and use it for a real L1 store/retrieve round trip.
set -euo pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

SERVER_CHUNK_SIZE="${CHUNK_SIZE:-256}"
PROMPT_TOKENS="${MULTI_CHUNK_PROMPT_TOKENS:-900}"
MAX_TOKENS="${MAX_TOKENS:-1}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
OUT_DIR="${RESULTS_DIR}/multi_chunk_size"
mkdir -p "$OUT_DIR"

if [[ "${LAUNCH_SECOND_ENGINE:-false}" != "true" ]]; then
    echo "This workload requires two LMCache-connected engine instances" >&2
    exit 1
fi

read_metric() {
    local metric_name="$1"
    local model_name="${2:-}"
    curl --noproxy '*' -fsS --max-time 10 \
        "http://127.0.0.1:${LMCACHE_HTTP_PORT}/metrics" \
        | awk -v metric="$metric_name" -v model="$model_name" '
            ($1 == metric || (index($1, metric) == 1 \
                    && substr($1, length(metric) + 1, 1) == "{")) \
                    && (model == "" \
                        || index($1, "model_name=\"" model "\"") > 0) {
                total += $NF
            }
            END { printf "%d", total + 0 }
        '
}

send_completion() {
    local port="$1"
    local model="$2"
    local prompt_tokens="$3"
    local output_file="$4"
    python3 - "$port" "$model" "$prompt_tokens" "$MAX_TOKENS" "$output_file" \
        <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_tokens, max_tokens, output_file = sys.argv[1:6]
prompt = [1000 + index % 100 for index in range(int(prompt_tokens))]
body = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "seed": 0,
    }
).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
with open(output_file, "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
print(result["usage"]["prompt_tokens"])
PYEOF
}

wait_for_metric_delta() {
    local metric_name="$1"
    local before="$2"
    local expected="$3"
    local model_name="${4:-}"
    local deadline=$((SECONDS + 30))
    local current

    while ((SECONDS < deadline)); do
        current="$(read_metric "$metric_name" "$model_name")"
        if ((current - before >= expected)); then
            echo "$((current - before))"
            return 0
        fi
        sleep 1
    done
    current="$(read_metric "$metric_name" "$model_name")"
    echo "$((current - before))"
}

verify_registered_models() {
    local first_model="$1"
    local second_model="$2"
    python3 - "$LMCACHE_HTTP_PORT" "$first_model" "$second_model" <<'PYEOF'
import json
import sys
import urllib.request

port, first_model, second_model = sys.argv[1:4]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=10) as response:
    status = json.load(response)
registered = {
    value["model_name"] for value in status.get("cache_context_meta", {}).values()
}
expected = {first_model, second_model}
if not expected.issubset(registered):
    raise SystemExit(
        f"Expected registrations for {sorted(expected)}, got {sorted(registered)}"
    )
print(f"Registered models: {sorted(registered)}")
PYEOF
}

REQUESTED_METRIC="lmcache_mp_lookup_requested_tokens_total"
HIT_METRIC="lmcache_mp_lookup_hit_tokens_total"
WRITE_METRIC="lmcache_mp_l1_write_chunks_total"
LOADED_METRIC="lmcache_mp_num_chunks_loaded_total"

run_profile() {
    local label="$1"
    local port="$2"
    local model="$3"
    local effective_chunk_size="$4"
    local configured_prompt_tokens="$5"
    local profile_dir="$OUT_DIR/$label"
    local requested_before write_before prompt_tokens expected_tokens
    local expected_chunks server_aligned_tokens requested_delta write_delta
    local hit_before loaded_before warm_prompt_tokens hit_delta loaded_delta

    mkdir -p "$profile_dir"
    if ((effective_chunk_size <= SERVER_CHUNK_SIZE)); then
        echo "Client chunk size must exceed the server minimum" >&2
        return 1
    fi
    if ((configured_prompt_tokens / effective_chunk_size * effective_chunk_size \
            == configured_prompt_tokens / SERVER_CHUNK_SIZE * SERVER_CHUNK_SIZE)); then
        echo "Prompt length does not distinguish chunk sizes for $label" >&2
        return 1
    fi

    echo "=== $label: $model on port $port ==="
    echo "Client minimum / expected effective size: $effective_chunk_size"
    requested_before="$(read_metric "$REQUESTED_METRIC" "$model")"
    write_before="$(read_metric "$WRITE_METRIC")"
    prompt_tokens="$(send_completion \
        "$port" "$model" "$configured_prompt_tokens" "$profile_dir/cold.json")"
    expected_tokens=$((prompt_tokens / effective_chunk_size * effective_chunk_size))
    expected_chunks=$((expected_tokens / effective_chunk_size))
    server_aligned_tokens=$((prompt_tokens / SERVER_CHUNK_SIZE * SERVER_CHUNK_SIZE))
    if ((expected_tokens <= 0 || expected_tokens == server_aligned_tokens)); then
        echo "Prompt token count $prompt_tokens cannot distinguish sizes for $label" >&2
        return 1
    fi
    requested_delta="$(wait_for_metric_delta \
        "$REQUESTED_METRIC" "$requested_before" "$expected_tokens" "$model")"
    if ((requested_delta != expected_tokens)); then
        echo "$label cold lookup requested $requested_delta tokens;" \
            "expected $expected_tokens" >&2
        return 1
    fi

    write_delta="$(wait_for_metric_delta \
        "$WRITE_METRIC" "$write_before" "$expected_chunks")"
    if ((write_delta != expected_chunks)); then
        echo "$label cold store wrote $write_delta chunks; expected $expected_chunks" >&2
        return 1
    fi
    engine_clear_local_cache "$port"

    requested_before="$(read_metric "$REQUESTED_METRIC" "$model")"
    hit_before="$(read_metric "$HIT_METRIC" "$model")"
    loaded_before="$(read_metric "$LOADED_METRIC" "$model")"
    warm_prompt_tokens="$(send_completion \
        "$port" "$model" "$configured_prompt_tokens" "$profile_dir/warm.json")"
    if ((warm_prompt_tokens != prompt_tokens)); then
        echo "$label prompt count changed: cold=$prompt_tokens, warm=$warm_prompt_tokens" \
            >&2
        return 1
    fi
    requested_delta="$(wait_for_metric_delta \
        "$REQUESTED_METRIC" "$requested_before" "$expected_tokens" "$model")"
    hit_delta="$(wait_for_metric_delta \
        "$HIT_METRIC" "$hit_before" "$expected_tokens" "$model")"
    loaded_delta="$(wait_for_metric_delta \
        "$LOADED_METRIC" "$loaded_before" "$expected_chunks" "$model")"

    if ((requested_delta != expected_tokens)); then
        echo "$label warm lookup requested $requested_delta tokens;" \
            "expected $expected_tokens" >&2
        return 1
    fi
    if ((hit_delta != expected_tokens)); then
        echo "$label warm lookup hit $hit_delta tokens; expected $expected_tokens" >&2
        return 1
    fi
    if ((loaded_delta != expected_chunks)); then
        echo "$label warm retrieve loaded $loaded_delta chunks;" \
            "expected $expected_chunks" >&2
        return 1
    fi

    echo "PASS: $label transferred $expected_tokens tokens using chunk size" \
        "$effective_chunk_size."
}

echo "=== Multi-chunk-size integration test ==="
echo "Engine: ${ENGINE_NAME}"
echo "Server minimum: ${SERVER_CHUNK_SIZE}"

verify_registered_models "$MODEL" "$SECOND_ENGINE_MODEL"

run_profile primary "$ENGINE_PORT" "$MODEL" "${CLIENT_CHUNK_SIZE:-512}" \
    "$PROMPT_TOKENS"
run_profile second "$SECOND_ENGINE_PORT" "$SECOND_ENGINE_MODEL" \
    "$SECOND_CLIENT_CHUNK_SIZE" "${SECOND_MULTI_CHUNK_PROMPT_TOKENS:-1100}"
