# SPDX-License-Identifier: Apache-2.0
"""End-to-end KV cache remapping driver for the SDK example."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import cast
import argparse
import json
import time

# Third Party
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import httpx

# First Party
import lmcache.sdk as lmc_sdk

SOURCE_PARAGRAPH = (
    "A systems researcher is studying how an inference cache changes the "
    "latency profile of a long language model prompt. The notes discuss "
    "attention keys, attention values, memory tiers, token chunks, and the "
    "careful measurement of cold and warm requests."
)
TARGET_PARAGRAPH = (
    "A travel writer is drafting a field guide about coastal architecture, "
    "harbor markets, lighthouse repairs, regional food, and interviews with "
    "families who maintain old boats through stormy winters."
)


@dataclass(frozen=True)
class CompletionResult:
    """Text and latency returned by one OpenAI-compatible completion call."""

    text: str
    elapsed_seconds: float


def _token_ids(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    """Tokenize a prompt the same way this example expects vLLM to tokenize it."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    return [int(token_id) for token_id in token_ids]


def _prompt_with_min_tokens(
    tokenizer: PreTrainedTokenizerBase,
    paragraph: str,
    min_tokens: int,
) -> tuple[str, list[int]]:
    """Repeat ``paragraph`` until the prompt reaches ``min_tokens`` tokens."""
    prompt = paragraph
    token_ids = _token_ids(tokenizer, prompt)
    while len(token_ids) < min_tokens:
        prompt = f"{prompt}\n\n{paragraph}"
        token_ids = _token_ids(tokenizer, prompt)
    return prompt, token_ids


def _post_completion(
    *,
    vllm_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> CompletionResult:
    """Send one non-streaming completion request to vLLM."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    start = time.perf_counter()
    response = httpx.post(
        f"{vllm_url.rstrip('/')}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"completion response missing choices: {body}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict) or not isinstance(
        first_choice.get("text"), str
    ):
        raise RuntimeError(f"completion response has invalid choice: {body}")
    return CompletionResult(text=first_choice["text"], elapsed_seconds=elapsed)


def _registered_model_name(lmcache_url: str, fallback: str, timeout: float) -> str:
    """Read the model name registered by vLLM in the LMCache MP server."""
    response = httpx.get(f"{lmcache_url.rstrip('/')}/api/status", timeout=timeout)
    response.raise_for_status()
    status = response.json()
    if not isinstance(status, dict):
        return fallback
    contexts = status.get("gpu_context_meta", {})
    if not isinstance(contexts, dict):
        return fallback
    for context in contexts.values():
        if isinstance(context, dict) and isinstance(context.get("model_name"), str):
            return context["model_name"]
    return fallback


def _wait_for_hit(
    *,
    lmcache_url: str,
    model_name: str,
    tokens: list[int],
    expected_tokens: int,
    timeout: float,
) -> lmc_sdk.LookupResult:
    """Wait until LMCache lookup reports at least ``expected_tokens`` hits."""
    deadline = time.monotonic() + timeout
    last_result = lmc_sdk.LookupResult(0, 0, 0, 0)
    while time.monotonic() < deadline:
        last_result = lmc_sdk.lookup(lmcache_url, model_name=model_name, tokens=tokens)
        if last_result.hit_tokens >= expected_tokens:
            return last_result
        time.sleep(1)
    raise TimeoutError(
        "timed out waiting for LMCache hit; "
        f"last lookup was {last_result.hit_tokens}/{last_result.total_tokens} tokens"
    )


def _short_text(text: str, limit: int = 240) -> str:
    """Return a compact single-line text preview."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def run(args: argparse.Namespace) -> None:
    """Run the full KV retrieve/remap/store demonstration."""
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    source_prompt, source_tokens = _prompt_with_min_tokens(
        tokenizer,
        SOURCE_PARAGRAPH,
        args.min_prompt_tokens,
    )
    target_prompt, target_tokens = _prompt_with_min_tokens(
        tokenizer,
        TARGET_PARAGRAPH,
        args.min_prompt_tokens,
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_kv_path = args.work_dir / "source-kv.pt"

    print("== Step 1: source inference stores KV under source token IDs ==")
    source_completion = _post_completion(
        vllm_url=args.vllm_url,
        model_name=args.vllm_model_name,
        prompt=source_prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    lmcache_model_name = _registered_model_name(
        args.lmcache_url,
        args.lmcache_model_name or args.vllm_model_name,
        args.timeout,
    )
    source_lookup = _wait_for_hit(
        lmcache_url=args.lmcache_url,
        model_name=lmcache_model_name,
        tokens=source_tokens,
        expected_tokens=args.chunk_size,
        timeout=args.store_wait_timeout,
    )

    print("== Step 2: retrieve source KV through lmcache.sdk ==")
    retrieve_result = lmc_sdk.retrieve(
        source_kv_path,
        args.lmcache_url,
        model_name=lmcache_model_name,
        tokens=source_tokens,
        timeout=args.timeout,
    )
    if retrieve_result.hit_tokens <= 0:
        raise RuntimeError("source retrieve did not return any KV cache tokens")

    if len(target_tokens) < retrieve_result.hit_tokens:
        target_prompt, target_tokens = _prompt_with_min_tokens(
            tokenizer,
            TARGET_PARAGRAPH,
            retrieve_result.hit_tokens,
        )

    target_prefix_tokens = target_tokens[: retrieve_result.hit_tokens]
    source_prefix_tokens = source_tokens[: retrieve_result.hit_tokens]
    if source_prefix_tokens == target_prefix_tokens:
        raise RuntimeError("source and target token prefixes unexpectedly match")

    target_lookup_before = lmc_sdk.lookup(
        args.lmcache_url,
        model_name=lmcache_model_name,
        tokens=target_tokens,
        timeout=args.timeout,
    )

    print("== Step 3: store source KV under different target token IDs ==")
    store_result = lmc_sdk.store(
        source_kv_path,
        args.lmcache_url,
        model_name=lmcache_model_name,
        tokens=target_prefix_tokens,
        timeout=args.timeout,
    )

    target_lookup_after = lmc_sdk.lookup(
        args.lmcache_url,
        model_name=lmcache_model_name,
        tokens=target_tokens,
        timeout=args.timeout,
    )
    if store_result.stored_tokens < retrieve_result.hit_tokens:
        raise RuntimeError(
            "remap store wrote fewer tokens than were retrieved: "
            f"{store_result.stored_tokens} < {retrieve_result.hit_tokens}"
        )
    if target_lookup_after.hit_tokens < retrieve_result.hit_tokens:
        raise RuntimeError(
            "target lookup did not hit the remapped prefix: "
            f"{target_lookup_after.hit_tokens} < {retrieve_result.hit_tokens}"
        )

    print("== Step 4: target inference reuses the remapped token IDs ==")
    target_completion = _post_completion(
        vllm_url=args.vllm_url,
        model_name=args.vllm_model_name,
        prompt=target_prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    evaluation = {
        "lmcache_model_name": lmcache_model_name,
        "vllm_model_name": args.vllm_model_name,
        "source_prompt_tokens": len(source_tokens),
        "target_prompt_tokens": len(target_tokens),
        "retrieved_hit_tokens": retrieve_result.hit_tokens,
        "retrieved_hit_chunks": retrieve_result.hit_chunks,
        "target_lookup_before_hit_tokens": target_lookup_before.hit_tokens,
        "target_lookup_after_hit_tokens": target_lookup_after.hit_tokens,
        "target_prefix_ids_differ_from_source": source_prefix_tokens
        != target_prefix_tokens,
        "store_result": store_result.__dict__,
        "source_lookup_after_inference": source_lookup.__dict__,
        "target_lookup_after_remap": target_lookup_after.__dict__,
        "source_latency_seconds": source_completion.elapsed_seconds,
        "target_latency_seconds": target_completion.elapsed_seconds,
        "source_output_preview": _short_text(source_completion.text),
        "target_output_preview": _short_text(target_completion.text),
    }
    print("== Evaluation ==")
    print(json.dumps(cast(dict[str, object], evaluation), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model path or name.")
    parser.add_argument(
        "--vllm-model-name",
        required=True,
        help="Model name to send to vLLM's OpenAI-compatible API.",
    )
    parser.add_argument(
        "--lmcache-model-name",
        default="",
        help="Optional LMCache registered model name override.",
    )
    parser.add_argument("--tokenizer", default="", help="Optional tokenizer override.")
    parser.add_argument("--lmcache-url", required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--min-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--store-wait-timeout", type=float, default=120.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    """Run the command-line driver."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
