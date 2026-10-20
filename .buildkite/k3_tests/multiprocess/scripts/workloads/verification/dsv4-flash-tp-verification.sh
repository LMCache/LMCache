#!/usr/bin/env bash
# Shared request and verification helpers for the engine-specific DSV4 tests.

dsv4_build_prompt() {
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
# ~97 words/paragraph * 80 ~= 7.8k words. The incomplete final sentence keeps
# greedy decoding away from a near-tie that can make cross-prefill comparisons
# flaky even when the restored KV is correct.
with open(prompt_file, "w", encoding="utf-8") as output:
    output.write(paragraph * 80)
    output.write("The poll() system call waits for one of a set of file")
PYEOF
    echo "Prompt built ($(wc -w < "$PROMPT_FILE") words)."
    echo ""
}

dsv4_send_completion() {
    local out_file="$1"
    local run_name="$2"

    echo "=== Sending completion ($run_name) on port $ENGINE_PORT ==="
    python3 - "$ENGINE_PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS" \
        "$out_file" "${COMPLETION_TIMEOUT:-600}" <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, out_file, timeout = sys.argv[1:7]
with open(prompt_file, encoding="utf-8") as prompt_input:
    prompt = prompt_input.read()
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
with urllib.request.urlopen(request, timeout=int(timeout)) as response:
    data = json.load(response)
text = data["choices"][0]["text"]
with open(out_file, "w", encoding="utf-8") as output:
    output.write(text)
print(f"  generated {len(text)} chars")
PYEOF
    echo "$run_name completed"
    echo ""
}

dsv4_count_retrieves() {
    local count

    if [[ ! -f "$LMCACHE_LOG" ]]; then
        echo 0
        return 0
    fi
    count=$(grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true)
    echo "${count:-0}"
}

dsv4_run_l1_round_trip() {
    local retrieves_before
    local retrieves_after
    local failed=0

    if ! declare -F engine_clear_local_cache >/dev/null; then
        echo "The ${ENGINE_NAME:-inference engine} adapter cannot clear its local cache." \
            >&2
        return 1
    fi

    dsv4_send_completion "$OUT_A" "cold run"

    echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
    sleep "$STORE_DRAIN_SECONDS"
    retrieves_before=$(dsv4_count_retrieves)

    echo "=== Clearing ${ENGINE_NAME:-inference engine} local prefix cache" \
        "(LMCache preserved) ==="
    engine_clear_local_cache "$ENGINE_PORT"
    echo "Local prefix cache cleared."
    echo ""

    dsv4_send_completion "$OUT_B" "LMCache retrieve run"
    retrieves_after=$(dsv4_count_retrieves)

    echo "============================================"
    echo "=== Verifying DeepSeek-V4-Flash L1 correctness ==="
    echo "============================================"
    echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

    if cmp -s "$OUT_A" "$OUT_B"; then
        echo "PASS: cold-run and LMCache-retrieve outputs are identical."
    else
        echo "FAILED: outputs differ between the cold run and the LMCache-served run."
        echo "--- Cold run (first 400 chars) ---"
        head -c 400 "$OUT_A"
        echo ""
        echo "--- LMCache retrieve run (first 400 chars) ---"
        head -c 400 "$OUT_B"
        echo ""
        failed=1
    fi

    if ((retrieves_after <= retrieves_before)); then
        echo "FAILED: LMCache served no retrieves during the retrieve run"
        echo "        (before=${retrieves_before}, after=${retrieves_after}); the"
        echo "        comparison would be vacuous."
        failed=1
    fi

    if ((failed != 0)); then
        return 1
    fi

    echo ""
    echo "============================================"
    echo "=== DeepSeek-V4-Flash L1 test passed ==="
    echo "  outputs identical; LMCache served" \
        "$((retrieves_after - retrieves_before)) retrieves."
    echo "============================================"
}
