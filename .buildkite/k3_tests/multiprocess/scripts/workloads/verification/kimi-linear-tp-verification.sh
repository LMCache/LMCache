#!/usr/bin/env bash
# Shared request and result checks for Kimi-Linear TP workloads.

kimi_linear_build_prompt() {
    local prompt_file="$1"

    python3 - "$prompt_file" <<'PYEOF'
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
    f.write("\n\nSummarize the manual page text above:")
PYEOF
    echo "Prompt built ($(wc -w < "$prompt_file") words)."
    echo ""
}

kimi_linear_send_completion() {
    local port="$1"
    local model="$2"
    local prompt_file="$3"
    local max_tokens="$4"
    local out_file="$5"
    local run_name="$6"

    echo "=== Sending completion ($run_name) on port $port ==="
    python3 - "$port" "$model" "$prompt_file" "$max_tokens" "$out_file" <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, out_file = sys.argv[1:6]
with open(prompt_file) as f:
    prompt = f.read()
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

kimi_linear_count_retrieves() {
    local lmcache_log="$1"

    if [[ ! -f "$lmcache_log" ]]; then
        echo 0
        return
    fi
    grep -c "Retrieved" "$lmcache_log" 2>/dev/null || true
}

kimi_linear_verify_outputs() {
    local cold_output="$1"
    local cached_output="$2"
    local retrieves_before="$3"
    local retrieves_after="$4"
    local engine_name="$5"
    local failed=0

    echo "============================================"
    echo "=== Verifying cross-TP-rank KV correctness ==="
    echo "============================================"
    echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

    if cmp -s "$cold_output" "$cached_output"; then
        echo "PASS: ${engine_name} cold-run and LMCache-retrieve outputs are identical."
    else
        echo "FAILED: outputs differ between the cold run and the LMCache-served run."
        echo "--- ${engine_name} cold run (first 400 chars) ---"
        head -c 400 "$cold_output"
        echo ""
        echo "--- LMCache retrieve run (first 400 chars) ---"
        head -c 400 "$cached_output"
        echo ""
        failed=1
    fi

    if [[ "$retrieves_after" -le "$retrieves_before" ]]; then
        echo "FAILED: LMCache served no retrieves during the retrieve run "
        echo "        (before=${retrieves_before}, after=${retrieves_after}); the "
        echo "        comparison would be vacuous."
        failed=1
    fi

    if [[ "$failed" -ne 0 ]]; then
        return 1
    fi

    echo ""
    echo "============================================"
    echo "=== Kimi-Linear cross-TP-rank KV test passed ==="
    echo "  outputs identical; LMCache served $((retrieves_after - retrieves_before)) retrieves."
    echo "============================================"
}
