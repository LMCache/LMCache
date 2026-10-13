# SPDX-License-Identifier: Apache-2.0
"""Check cold/warm outputs and LMCache cache statistics in the image."""

# Standard
import json
import time
import urllib.request

base = "http://127.0.0.1:8000"
for attempt in range(450):
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError("vLLM readiness timeout")

payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "The recorded answer is 42. " * 90
            + "\nWhat number is the recorded answer? Reply with only that number.",
        }
    ],
    "max_tokens": 16,
    "temperature": 0,
}
results = []
for label in ("cold", "warm"):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read())
    print(label, json.dumps(data, ensure_ascii=False), flush=True)
    results.append(data)
    time.sleep(3)
cold, warm = results
equal = (
    cold["choices"][0]["message"]["content"] == warm["choices"][0]["message"]["content"]
)
stats = warm.get("kv_transfer_params")
print("OUTPUT_EQUAL", equal, "CACHE_STATS", json.dumps(stats), flush=True)
if not equal:
    raise RuntimeError("Cold and warm outputs differ")
if warm["choices"][0]["message"]["content"].strip() != "42":
    raise RuntimeError("Unexpected model answer")
if stats is None:
    raise RuntimeError("Runtime did not return per-request cache statistics")
if stats["num_lmcache_cached_tokens"] <= 0:
    raise RuntimeError(f"No warm LMCache hit: {stats}")
if cold["kv_transfer_params"]["num_lmcache_cached_tokens"] != 0:
    raise RuntimeError("Cold request unexpectedly hit LMCache")
print("E2E_PASS", flush=True)
