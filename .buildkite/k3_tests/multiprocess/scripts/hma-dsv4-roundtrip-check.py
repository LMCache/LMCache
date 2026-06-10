#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash HMA store/retrieve round-trip on a single request.

Run via ``run-single-test.sh hma_dsv4_roundtrip``, which launches a vLLM
OpenAI server (dummy weights, 4 layers, fp8_ds_mla KV) wired to the
``LMCacheMPConnector`` through the shared ``launch-processes.sh`` flow.

Why dummy weights: the real DeepSeek-V4-Flash fp4 (MXFP4) expert weights do
not load on the current vLLM build (the experts are misdispatched to the fp8
MoE method, so loading KeyErrors on ``routed_experts.w13_weight_scale``).
``--load-format dummy`` skips checkpoint loading, so this check still
exercises the actual ratio-4/128 MLA transfer kernels -- the path this PR
drives -- without needing a fix for that vLLM loader bug.

Why a single greedy request: DeepSeek-V4-Flash's sparse-MLA backends do not
implement batch invariance, so run-to-run bit-exactness is not guaranteed
under concurrency. One request decoded greedily keeps the batch composition
identical across both runs, so the only thing that can change the output is a
corrupted KV transfer.

Flow:
  1. Send one greedy completion; vLLM computes it and stores the prefix KV to
     LMCache.
  2. Reset vLLM's local prefix cache (APC) via POST /reset_prefix_cache
     (reset_external defaults to false, so the LMCache copy is preserved).
  3. Re-send the identical request; vLLM's APC misses, so the prefix KV is
     served by LMCache.
  4. Assert the two completions are byte-identical -- a corrupted KV transfer
     would change the continuation.
  5. Assert LMCache actually served retrieves in run 2 (non-vacuous).
"""

# Standard
import json
import os
import sys
import time
import urllib.request

VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash")
BUILD_ID = os.environ.get("BUILD_ID", f"local_{os.getpid()}")
LMCACHE_LOG = os.environ.get("LMCACHE_LOG", f"/tmp/build_{BUILD_ID}_lmcache.log")
# Seconds to let async LMCache stores drain before the retrieve run.
STORE_DRAIN_SECONDS = int(os.environ.get("STORE_DRAIN_SECONDS", "20"))
# Tokens to generate (greedy, ignore_eos so the count is fixed and non-empty
# even with random dummy logits).
MAX_TOKENS = int(os.environ.get("ROUNDTRIP_MAX_TOKENS", "16"))

# A prompt long enough to span several LMCache chunks (default 256 tokens), so
# run 2 has a real prefix to retrieve rather than recomputing a sub-chunk tail.
_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the kv cache is stored "
    "and retrieved through the multiprocess connector under test. "
)
PROMPT = _SENTENCE * 80


def post_json(path: str, payload: dict) -> dict:
    """POST ``payload`` as JSON to the vLLM server and return the parsed reply.

    Args:
        path: Request path on the vLLM server (e.g. ``/v1/completions``).
        payload: JSON-serializable request body.

    Returns:
        The parsed JSON response body.

    Raises:
        urllib.error.URLError: If the request fails at the transport level.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def generate(label: str) -> str:
    """Send one greedy completion and return its generated text.

    Args:
        label: Human-readable run label used only in progress logging.

    Returns:
        The generated completion text (``choices[0].text``).
    """
    print(f"=== {label}: sending greedy completion ===")
    body = post_json(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
        },
    )
    text = body["choices"][0]["text"]
    print(f"  {label} generated {len(text)} chars")
    return text


def reset_prefix_cache() -> None:
    """Clear vLLM's local APC while preserving the LMCache-managed cache.

    Raises:
        SystemExit: If the server does not acknowledge with HTTP 200 (e.g. the
            dev-mode endpoint is absent because VLLM_SERVER_DEV_MODE was unset).
    """
    print("=== Resetting vLLM local prefix cache (LMCache preserved) ===")
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/reset_prefix_cache", method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise SystemExit(
                f"reset_prefix_cache returned HTTP {resp.status}; "
                "is VLLM_SERVER_DEV_MODE=1?"
            )


def count_retrieves() -> int:
    """Return the number of completed LMCache retrieves recorded in the log.

    Returns:
        Count of ``Retrieved`` lines in the LMCache server log (0 if the log
        file does not exist yet).
    """
    if not os.path.exists(LMCACHE_LOG):
        return 0
    with open(LMCACHE_LOG, errors="ignore") as f:
        return sum(1 for line in f if "Retrieved" in line)


def main() -> int:
    """Run the round-trip and return a process exit code (0 pass, 1 fail)."""
    print("=== DSV4-Flash HMA round-trip (dummy weights, single request) ===")
    print(f"Model: {MODEL}  vLLM port: {VLLM_PORT}  max_tokens: {MAX_TOKENS}")

    text_compute = generate("compute run")

    print(f"Waiting {STORE_DRAIN_SECONDS}s for LMCache stores to drain...")
    time.sleep(STORE_DRAIN_SECONDS)
    retrieves_before = count_retrieves()

    reset_prefix_cache()

    text_retrieve = generate("LMCache retrieve run")
    retrieves_after = count_retrieves()

    print("============================================")
    print(f"LMCache retrieves logged: before={retrieves_before}, "
          f"after={retrieves_after}")

    failures = []
    if text_compute != text_retrieve:
        failures.append(
            "output diverged between compute and LMCache-retrieve runs:\n"
            f"  compute : {text_compute!r}\n"
            f"  retrieve: {text_retrieve!r}"
        )
    if retrieves_after <= retrieves_before:
        failures.append(
            "LMCache served no retrieves during the retrieve run "
            f"(before={retrieves_before}, after={retrieves_after})"
        )

    if failures:
        print("DSV4_HMA_ROUNDTRIP: FAIL")
        for f in failures:
            print(" -", f)
        return 1

    print(
        f"DSV4_HMA_ROUNDTRIP: PASS (identical {len(text_compute)}-char output; "
        f"LMCache served {retrieves_after - retrieves_before} retrieves)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
