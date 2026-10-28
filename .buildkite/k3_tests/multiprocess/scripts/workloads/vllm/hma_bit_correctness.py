# SPDX-License-Identifier: Apache-2.0
"""Compare hybrid-model GPU APC and LMCache results in three stages.

Cold prefill populates both caches. The second request hits GPU APC and supplies
the reference token IDs, selected-token logprobs, and top-100 candidate logprobs
at every output position. Reset only GPU APC before the third request, which
must retrieve the same prefix from LMCache and reproduce the reference exactly.
Cold prefill is not the numerical reference because full and cached
prefills can differ in BF16. Equal block/chunk sizes and measured hit lengths
ensure both cached runs recompute the same suffix.
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
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoConfig, AutoTokenizer
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
) -> None:
    """Reject incomplete generations, wrong cache paths, and numerical drift.

    All stages must contain 32 generated tokens and 100 finite candidate
    logprobs per position. GPU APC and LMCache must match exactly, including
    candidate token identities, independently of dictionary insertion order.

    Args:
        cold: Initial request, which must miss both caches.
        gpu: GPU APC reference request, without LMCache retrievals.
        restored: LMCache request after GPU APC reset.
        chunk_size: Shared positive LMCache chunk and vLLM block size.
        sliding_window: Sliding-window length, or zero for a non-SWA hybrid.

    Raises:
        ValueError: If hit lengths, output token IDs, selected-token logprobs,
            or top-100 candidate logprobs are invalid or differ.
    """
    for result in (cold, gpu, restored):
        if (
            len(result.token_ids) != 32
            or len(result.logprobs) != 32
            or not all(math.isfinite(value) for value in result.logprobs)
        ):
            raise ValueError("Expected 32 output token IDs and finite logprobs")
        if len(result.top_logprobs) != 32 or any(
            len(candidates) != 100
            or not all(math.isfinite(value) for value in candidates.values())
            for candidates in result.top_logprobs
        ):
            raise ValueError(
                "Expected 100 finite candidate logprobs at all 32 positions"
            )
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
    if gpu.token_ids != restored.token_ids or gpu.logprobs != restored.logprobs:
        raise ValueError("GPU APC and LMCache token IDs or logprobs differ")
    if gpu.top_logprobs != restored.top_logprobs:
        raise ValueError(
            "GPU APC and LMCache top-100 candidate tokens or logprobs differ"
        )


def main(workload: str = "hma_swa_correctness") -> None:
    """Run the three-stage comparison using the multiprocess harness environment.

    Args:
        workload: Either hma_swa_correctness, which also verifies a prefix longer
            than the model's sliding window, or hma_kda_mla_correctness.

    Raises:
        ValueError: If configuration, cache hits, or generated results are wrong.
        requests.HTTPError: If a vLLM endpoint fails.
        OSError: If server logs or result artifacts cannot be accessed.
    """
    model = os.environ["MODEL"]
    chunk_size = int(os.environ["CHUNK_SIZE"])
    if chunk_size <= 0 or chunk_size != int(os.environ["VLLM_BLOCK_SIZE"]):
        raise ValueError("LMCache chunk size must equal the positive vLLM block size")
    base_url = f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8000')}"
    build_id = os.environ["BUILD_ID"]
    log = Path(os.environ.get("LMCACHE_LOG", f"/tmp/build_{build_id}_lmcache.log"))
    vllm_log = Path(os.environ["ENGINE_LOG_FILE"]).read_text()
    transport = os.environ["ENGINE_DRIVEN_TRANSPORT"]
    marker = {
        "shm": "Using shm engine-driven transfer strategy",
        "inline": "Using safe inline engine-driven transfer strategy",
    }[transport]
    groups = re.findall(
        r"Worker engine-driven context registered \([^)]*groups=(\d+)", vllm_log
    )
    if marker not in log.read_text() or not groups or int(groups[-1]) < 2:
        raise ValueError(
            "Requested transport and multi-group registration not confirmed"
        )
    output = Path(os.environ["RESULTS_DIR"]) / workload
    output.mkdir(parents=True, exist_ok=True)
    notes = "\n".join(
        f"Record {i:03d}: Station Cedar-{i} keeps its backup batteries in room "
        f"{100 + i}. The maintenance team checks cooling, power, and network "
        "connections each Monday. A green indicator means the last check passed."
        for i in range(60)
    )
    sliding_window = 0
    if workload == "hma_swa_correctness":
        sliding_window = (
            AutoConfig.from_pretrained(model).get_text_config().sliding_window
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0") == "1"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Read the following operations notes.\n"
                + notes
                + "\nSummarize the routine maintenance procedure in three sentences.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) + 32 > int(os.environ.get("MAX_MODEL_LEN", "8192")):
        raise ValueError("Prompt and output exceed max model length")
    request = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 32,
        "min_tokens": 32,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 0,
        "logprobs": 100,
        "return_token_ids": True,
        # Decoded strings can collide for distinct vocabulary tokens.
        "return_tokens_as_token_ids": True,
    }
    (output / "request.json").write_text(json.dumps(request, indent=2))
    results = []
    previous_gpu = previous_retrieved = 0
    with requests.Session() as session:
        session.trust_env = False
        for stage in ("cold", "gpu_apc", "lmcache"):
            if stage == "gpu_apc":
                time.sleep(float(os.environ.get("STORE_DRAIN_SECONDS", "20")))
            if stage == "lmcache":
                reset = session.post(
                    f"{base_url}/reset_prefix_cache",
                    params={"reset_external": "false"},
                    timeout=60,
                )
                reset.raise_for_status()
                if reset.json()["success"] is not True:
                    raise ValueError("GPU prefix-cache reset failed")
            response = session.post(
                f"{base_url}/v1/completions", json=request, timeout=600
            )
            response.raise_for_status()
            data = response.json()
            (output / f"{stage}.json").write_text(json.dumps(data, indent=2))
            # Engine metrics publication can lag the final HTTP response.
            time.sleep(1)
            metrics = session.get(f"{base_url}/metrics", timeout=10)
            metrics.raise_for_status()
            hits = [
                sample.value
                for family in text_string_to_metric_families(metrics.text)
                for sample in family.samples
                if sample.name == "vllm:prefix_cache_hits_total"
            ]
            if not hits:
                raise ValueError("Missing GPU prefix-cache hit counter")
            gpu = int(sum(hits))
            retrieved = sum(
                int(n) for n in re.findall(r"Retrieved (\d+) tokens", log.read_text())
            )
            choice = data["choices"][0]
            results.append(
                Result(
                    data["usage"]["prompt_tokens"],
                    choice["token_ids"],
                    choice["logprobs"]["token_logprobs"],
                    choice["logprobs"]["top_logprobs"],
                    gpu - previous_gpu,
                    retrieved - previous_retrieved,
                )
            )
            print(
                f"{stage}: GPU hit={gpu - previous_gpu}, "
                f"LMCache retrieved={retrieved - previous_retrieved}",
                flush=True,
            )
            previous_gpu, previous_retrieved = gpu, retrieved
    verify_results(results[0], results[1], results[2], chunk_size, sliding_window)
    print(
        "PASS: GPU APC and LMCache token IDs, selected-token logprobs, "
        "and top-100 candidate logprobs match exactly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workload",
        nargs="?",
        choices=("hma_swa_correctness", "hma_kda_mla_correctness"),
        default="hma_swa_correctness",
    )
    main(parser.parse_args().workload)
