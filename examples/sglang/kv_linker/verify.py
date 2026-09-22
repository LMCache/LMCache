# SPDX-License-Identifier: Apache-2.0
"""Verify LMCache restore after clearing only SGLang GPU radix cache."""

# Standard
import argparse
import json
import math
import uuid

# Third Party
import requests


def verify(base_url: str) -> None:
    # Each run gets a fresh salt: old objects cannot make the cold request a hit.
    payload = {
        "text": "Explain why KV caching speeds up language model inference. " * 64,
        "cache_salt": str(uuid.uuid4()),
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        "return_logprob": True,
    }

    def generate():
        response = requests.post(base_url + "/generate", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    cold = generate()
    local = generate()
    response = requests.post(
        base_url + "/flush_cache", params={"timeout": 30}, timeout=40
    )
    response.raise_for_status()
    warm = generate()
    cold_meta, local_meta, warm_meta = (
        cold["meta_info"],
        local["meta_info"],
        warm["meta_info"],
    )
    details = warm_meta.get("cached_tokens_details") or {}
    local_details = local_meta.get("cached_tokens_details") or {}
    assert cold_meta["cached_tokens"] == 0, cold_meta
    assert warm_meta["cached_tokens"] > 0, warm_meta
    # UnifiedCacheLinker reports direct external loads in the host field.
    assert details.get("host", 0) > 0, warm_meta
    assert details.get("device", 0) == 0, warm_meta
    assert local_details.get("device", 0) > 0, local_meta
    assert local_details.get("host", 0) == 0, local_meta
    assert warm_meta["cached_tokens"] == local_meta["cached_tokens"]
    assert cold["text"] == warm["text"], (cold["text"], warm["text"])
    assert local["text"] == warm["text"], (local["text"], warm["text"])
    cold_probs = cold_meta["output_token_logprobs"]
    local_probs = local_meta["output_token_logprobs"]
    warm_probs = warm_meta["output_token_logprobs"]
    assert len(cold_probs) == len(local_probs) == len(warm_probs) > 0
    max_difference = 0.0
    cold_difference = 0.0
    # Match the compute geometry: both references reuse the same prefix.
    # Cold prefill can differ numerically even with SGLang's native cache.
    for cold_token, first, second in zip(
        cold_probs, local_probs, warm_probs, strict=True
    ):
        assert cold_token[1] == first[1] == second[1], (cold_token, first, second)
        assert math.isfinite(first[0]) and math.isfinite(second[0])
        assert math.isfinite(cold_token[0])
        difference = abs(first[0] - second[0])
        assert difference < 0.05, (first, second)
        max_difference = max(max_difference, difference)
        cold_difference = max(cold_difference, abs(cold_token[0] - second[0]))
    print(
        json.dumps(
            {
                "cold_cached_tokens": cold_meta["cached_tokens"],
                "local_cached_tokens_details": local_details,
                "warm_cached_tokens": warm_meta["cached_tokens"],
                "warm_cached_tokens_details": details,
                "matching_output_tokens": len(cold_probs),
                "max_local_vs_external_logprob_difference": max_difference,
                "max_cold_vs_external_logprob_difference": cold_difference,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    verify(parser.parse_args().base_url.rstrip("/"))


if __name__ == "__main__":
    main()
