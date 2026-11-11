# SPDX-License-Identifier: Apache-2.0
"""Compare SGLang DSV4 GPU cache and LMCache results in three stages.

Cold prefill populates both caches. A GPU-cached request supplies the numerical
reference. Flush only the GPU cache before restoring the prefix from LMCache;
require exact output token IDs and selected-token and top-100 logprobs.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import os
import re
import time

# Third Party
from transformers import AutoTokenizer
import requests


@dataclass
class Result:
    """Generation and cache hits, with top-logprob keys encoded as token IDs."""

    prompt_tokens: int
    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    gpu_hit_tokens: int
    lmcache_hit_tokens: int


def verify_results(
    cold: Result,
    gpu: Result,
    restored: Result,
    chunk_size: int,
    sliding_window: int,
    *,
    output_tokens: int = 32,
) -> None:
    """Reject incomplete generations, wrong cache paths, and numerical drift.

    All stages must contain output_tokens generated token IDs and finite
    selected-token and top-100 candidate logprobs at every position. GPU APC
    and LMCache must match exactly, including candidate token identities,
    independently of dictionary insertion order.

    Args:
        cold: Initial request, which must miss both caches.
        gpu: GPU APC reference request, without LMCache retrievals.
        restored: LMCache request after GPU APC reset.
        chunk_size: Positive LMCache chunk size aligned with GPU APC reuse.
        sliding_window: Sliding-window length, or zero for a non-SWA hybrid.
        output_tokens: Required number of generated tokens at every stage.

    Raises:
        ValueError: If hit lengths, output token IDs, selected-token logprobs,
            or top-100 candidate logprobs are invalid or differ.
    """
    for result in (cold, gpu, restored):
        if len(result.token_ids) != output_tokens:
            raise ValueError(f"Expected {output_tokens} output token IDs")
        if len(result.logprobs) != output_tokens or not all(
            math.isfinite(value) for value in result.logprobs
        ):
            raise ValueError("Expected finite selected-token logprobs")
        if len(result.top_logprobs) != output_tokens or any(
            len(candidates) != 100
            or not all(math.isfinite(value) for value in candidates.values())
            for candidates in result.top_logprobs
        ):
            raise ValueError("Expected 100 finite candidate logprobs at every position")
    if not (cold.prompt_tokens == gpu.prompt_tokens == restored.prompt_tokens):
        raise ValueError("Prompt lengths differ")
    # At least the final prompt token must be computed to produce logits.
    expected = (gpu.prompt_tokens - 1) // chunk_size * chunk_size
    if expected <= sliding_window:
        raise ValueError(
            f"Cached prefix must extend beyond the {sliding_window}-token "
            "sliding window"
        )
    if (cold.gpu_hit_tokens, cold.lmcache_hit_tokens) != (0, 0):
        raise ValueError("Initial request was not cold")
    if (gpu.gpu_hit_tokens, gpu.lmcache_hit_tokens) != (expected, 0):
        raise ValueError(f"GPU APC must reuse exactly {expected} tokens alone")
    if (restored.gpu_hit_tokens, restored.lmcache_hit_tokens) != (0, expected):
        raise ValueError(
            f"LMCache must restore exactly {expected} tokens without GPU APC"
        )
    if gpu.token_ids != restored.token_ids:
        raise ValueError("GPU APC and LMCache token IDs differ")
    if gpu.logprobs != restored.logprobs:
        raise ValueError("GPU APC and LMCache selected-token logprobs differ")
    if gpu.top_logprobs != restored.top_logprobs:
        raise ValueError(
            "GPU APC and LMCache top-100 candidate tokens or logprobs differ"
        )


def main(
    workload: str = "hma_dsv4_correctness",
    *,
    transfer_mode: str = "lmcache_driven",
    output_tokens: int = 32,
) -> None:
    """Run the SGLang comparison using the multiprocess harness environment.

    Args:
        workload: Must be hma_dsv4_correctness, using 256-token cache alignment.
        transfer_mode: Must be lmcache_driven; requires multi-group registration.
        output_tokens: Positive number of generated tokens required per stage.

    Raises:
        ValueError: If configuration, cache hits, or generated results are wrong.
        requests.HTTPError: If a SGLang endpoint fails.
        OSError: If server logs or result artifacts cannot be accessed.
    """
    if output_tokens <= 0:
        raise ValueError("Output token count must be positive")
    if workload != "hma_dsv4_correctness" or transfer_mode != "lmcache_driven":
        raise ValueError("SGLang requires the LMCache-driven DSV4 workload")
    model = os.environ["MODEL"]
    chunk_size = int(os.environ["CHUNK_SIZE"])
    # DSV4's heterogeneous logical block sizes align together at 256 tokens.
    if chunk_size != 256:
        raise ValueError("LMCache chunk size must equal the positive cache alignment")
    base_url = f"http://127.0.0.1:{os.environ.get('SGLANG_PORT', '8000')}"
    build_id = os.environ["BUILD_ID"]
    log = Path(os.environ.get("LMCACHE_LOG", f"/tmp/build_{build_id}_lmcache.log"))
    server_log = log.read_text()
    registered = (
        "Registered KV cache for GPU ID" in server_log
        and len(set(re.findall(r"engine_group_idx=(\d+)", server_log))) >= 2
    )
    if not registered:
        raise ValueError(
            "Requested transport and multi-group registration not confirmed"
        )
    output = Path(os.environ["RESULTS_DIR"]) / workload
    output.mkdir(parents=True, exist_ok=True)
    notes = "\n".join(
        f"Record {i:03d}: Station Cedar-{i} keeps its backup batteries in room "
        f"{100 + i}. The maintenance team checks cooling, power, and network "
        "connections each Monday. A green indicator means the last check passed."
        for i in range(16)
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = (
        "Read the following operations notes.\n"
        + notes
        + "\nSummarize the routine maintenance procedure in three sentences."
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) + output_tokens > int(os.environ.get("MAX_MODEL_LEN", "8192")):
        raise ValueError("Prompt and output exceed max model length")
    request = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": output_tokens,
            "min_new_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0,
        },
        "return_logprob": True,
        "top_logprobs_num": 100,
        "return_text_in_logprobs": False,
    }
    (output / "request.json").write_text(json.dumps(request, indent=2))
    results = []
    previous_retrieved = 0
    with requests.Session() as session:
        session.trust_env = False
        for stage in ("cold", "gpu_apc", "lmcache"):
            if stage == "gpu_apc":
                time.sleep(float(os.environ.get("STORE_DRAIN_SECONDS", "20")))
            if stage == "lmcache":
                reset = session.post(
                    f"{base_url}/flush_cache", params={"timeout": 60}, timeout=70
                )
                reset.raise_for_status()
            response = session.post(f"{base_url}/generate", json=request, timeout=600)
            response.raise_for_status()
            data = response.json()
            (output / f"{stage}.json").write_text(json.dumps(data, indent=2))
            # Engine metrics publication can lag the final HTTP response.
            time.sleep(1)
            retrieved = sum(
                int(n) for n in re.findall(r"Retrieved (\d+) tokens", log.read_text())
            )
            meta = data["meta_info"]
            details = meta["cached_tokens_details"]
            if meta["cached_tokens"] == 0:
                gpu_hits = storage_hits = 0
                if details is not None:
                    raise ValueError("Cold SGLang request reported cache sources")
            else:
                gpu_hits, storage_hits = details["device"], details["storage"]
                if details["host"] != 0:
                    raise ValueError("Unexpected SGLang host-cache hits")
            if (
                meta["cached_tokens"] != gpu_hits + storage_hits
                or storage_hits != retrieved - previous_retrieved
                or meta["completion_tokens"] != output_tokens
            ):
                raise ValueError("SGLang cache or completion accounting differs")
            selected = meta["output_token_logprobs"]
            if data["output_ids"] != [entry[1] for entry in selected]:
                raise ValueError("SGLang output and logprob token IDs differ")
            results.append(
                Result(
                    meta["prompt_tokens"],
                    data["output_ids"],
                    [entry[0] for entry in selected],
                    [
                        {str(entry[1]): entry[0] for entry in candidates}
                        for candidates in meta["output_top_logprobs"]
                    ],
                    gpu_hits,
                    storage_hits,
                )
            )
            print(
                f"{stage}: GPU hit={gpu_hits}, LMCache retrieved={storage_hits}",
                flush=True,
            )
            previous_retrieved = retrieved
    verify_results(
        results[0],
        results[1],
        results[2],
        chunk_size,
        0,
        output_tokens=output_tokens,
    )
    print(
        "PASS: GPU APC and LMCache token IDs, selected-token logprobs, "
        "and top-100 candidate logprobs match exactly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workload",
        nargs="?",
        choices=("hma_dsv4_correctness",),
        default="hma_dsv4_correctness",
    )
    parser.add_argument(
        "--transfer-mode", choices=("lmcache_driven",), default="lmcache_driven"
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=32,
        help="Number of output token IDs required at every stage (default: 32)",
    )
    args = parser.parse_args()
    main(
        args.workload,
        transfer_mode=args.transfer_mode,
        output_tokens=args.output_tokens,
    )
