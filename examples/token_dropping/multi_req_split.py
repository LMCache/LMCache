# SPDX-License-Identifier: Apache-2.0
"""End-to-end KV cache remapping driver for the token dropping example. Accepts multiple concurrent requests."""

# Standard
import asyncio
from dataclasses import dataclass
import argparse
import json
import time
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer

# Third Party
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import httpx
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# First Party
import lmcache.sdk.kvcache as lmc_sdk

app = FastAPI()
SEND_EXECUTOR = ThreadPoolExecutor(max_workers=256)
ctx = None

@dataclass
class AppState:
    """State of the FastAPI app."""
    
    model: str
    vllm_model_name: str
    lmcache_model_name: str
    tokenizer: PreTrainedTokenizerBase
    config: AutoConfig
    lmcache_url: str
    vllm_url: str
    chunk_size: int
    timeout: float
    trust_remote_code: bool
    head_size: int

@dataclass(frozen=True)
class CompletionResult:
    """Text and latency returned by one OpenAI-compatible completion call."""

    text: str
    elapsed_seconds: float
    ttft_seconds: float
    decode_throughput_tokens_per_second: float
    output_chunks: int

SOURCE_TOKENS = {}

@dataclass(frozen=True)
class DropResult:
    """Result of token dropping operation."""

    dropped_tokens: int
    remaining_tokens: int
    stored_tokens: bool

class CompletionRequest(BaseModel):
    """Parsed fields from the incoming JSON body of a completion request."""

    prompt: str
    id: int
    max_tokens: int = 32
    drop_algorithm: str = "random"
    drop_remap_pe: bool = False
    drop_compression: float = 0.0
    drop_chunk: int = 0


def _token_ids_without_special_tokens(
    tokenizer: PreTrainedTokenizerBase, prompt: str
) -> list[int]:
    """Tokenize text for a suffix that will be embedded in a token-ID prompt."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return [int(token_id) for token_id in token_ids]


def _post_completion(
    *,
    vllm_url: str,
    model_name: str,
    prompt: str | list[int],
    max_tokens: int,
    timeout: float,
    cache_salt: str = ""
) -> CompletionResult:
    """Send one non-streaming completion request to vLLM."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True
    }
    if cache_salt:
        payload["cache_salt"] = cache_salt
    start = time.perf_counter()
    response = httpx.post(
        f"{vllm_url.rstrip('/')}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    print(f"completion response body: {body}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"completion response missing choices: {body}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict) or not isinstance(
        first_choice.get("text"), str
    ):
        raise RuntimeError(f"completion response has invalid choice: {body}")
    return CompletionResult(text=first_choice["text"], elapsed_seconds=elapsed, ttft_seconds=0, decode_throughput_tokens_per_second=0.0, output_chunks=0)


def _post_completion_streaming(
    *,
    vllm_url: str,
    model_name: str,
    prompt: str | list[int],
    max_tokens: int,
    timeout: float,
    cache_salt: str = ""
) -> CompletionResult:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
    }
    if cache_salt:
        payload["cache_salt"] = cache_salt

    pieces: list[str] = []
    token_times: list[float] = []
    start = time.perf_counter()

    with httpx.stream(
        "POST",
        f"{vllm_url.rstrip('/')}/v1/completions",
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue

            body = json.loads(line.removeprefix("data: "))
            text = body["choices"][0].get("text", "")
            # if text:
            pieces.append(text)
            token_times.append(time.perf_counter())

    end = time.perf_counter()
    output_chunks = len(token_times)

    ttft = token_times[0] - start if token_times else 0.0
    if output_chunks <= 1:
        decode_tps = 0.0
    else:
        decode_tps = (output_chunks - 1) / (token_times[-1] - token_times[0])
    
    if output_chunks > 1:
        print(
            f"for decoding {output_chunks - 1} chunks, needs a total of "
            f"{token_times[-1] - token_times[0]:.10f} seconds between the first and last token emitted"
        )

    return CompletionResult(
        text="".join(pieces),
        elapsed_seconds=end - start,
        ttft_seconds=ttft,
        decode_throughput_tokens_per_second=decode_tps,
        output_chunks=output_chunks
    )


def _registered_model_name(lmcache_url: str, fallback: str, timeout: float) -> str:
    """Read the model name registered by vLLM in the LMCache MP server."""
    response = httpx.get(f"{lmcache_url.rstrip('/')}/status", timeout=timeout)
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


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs using GPT-NeoX/Llama-style half layout."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs using interleaved layout."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _rope_cos_sin(
    *,
    config: AutoConfig,
    positions: torch.Tensor,
    rotary_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build RoPE cos/sin for the model config, including HF rope_scaling."""
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = None
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))

    if rope_scaling is None or rope_type == "default":
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        attention_scaling = 1.0
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type is None:
            raise ValueError(f"rope_scaling is missing rope_type/type: {rope_scaling}")

        if rope_type == "default":
            rope_theta = getattr(config, "rope_theta", 1000000) # Qwen3
            inv_freq = 1.0 / (
                rope_theta
                ** (
                    torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                    / rotary_dim
                )
            )
            attention_scaling = 1.0
        else:
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                config=config,
                device=device,
                seq_len=int(positions.max().item()) + 1,
            )

    inv_freq = inv_freq[: rotary_dim // 2].to(device=device, dtype=torch.float32)
    positions = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos[:, None, :], sin[:, None, :], float(attention_scaling)


def _rerotate_k_cache(
    *,
    k_flat: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    config: AutoConfig,
    head_size: int,
    is_neox_style: bool,
) -> torch.Tensor:
    """Move key cache RoPE from old positions to new positions."""
    num_tokens = k_flat.shape[0]
    k = k_flat.view(num_tokens, -1, head_size)

    partial_factor = getattr(config, "partial_rotary_factor", 1.0)
    rotary_dim = int(head_size * partial_factor)
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    old_cos, old_sin, old_scale = _rope_cos_sin(
        config=config,
        positions=old_positions,
        rotary_dim=rotary_dim,
        device=k.device,
        dtype=k.dtype,
    )
    new_cos, new_sin, new_scale = _rope_cos_sin(
        config=config,
        positions=new_positions,
        rotary_dim=rotary_dim,
        device=k.device,
        dtype=k.dtype,
    )

    rotate_half = _rotate_half_neox if is_neox_style else _rotate_half_interleaved

    # Detach old RoPE: inverse rotation. If attention_scaling was applied,
    # divide it out before applying the inverse rotation.
    k_unscaled = k_rot / old_scale
    k_plain = (k_unscaled * old_cos) - (
        rotate_half(k_unscaled) * old_sin
    )

    # Attach new RoPE.
    k_new = new_scale * (
        (k_plain * new_cos) + (rotate_half(k_plain) * new_sin)
    )

    return torch.cat((k_new, k_pass), dim=-1).reshape_as(k_flat)


@app.post("/api/prefill")
async def prefill(request: CompletionRequest) -> CompletionResult:
    """Endpoint for performing prefill with the given prompt."""
    return await asyncio.to_thread(_prefill, request)


def _prefill(req: CompletionRequest) -> CompletionResult:
    """Perform a prefill operation with the given request."""
    source_tokens = _token_ids_without_special_tokens(app.state.tokenizer, req.prompt)

    SOURCE_TOKENS[req.id] = {}
    SOURCE_TOKENS[req.id]["source"] = source_tokens

    source_completion = _post_completion(
        vllm_url=app.state.vllm_url,
        model_name=app.state.vllm_model_name,
        prompt=source_tokens,
        max_tokens=1,
        timeout=app.state.timeout,
    )
    
    return CompletionResult(
        text="",
        elapsed_seconds=source_completion.elapsed_seconds,
        ttft_seconds=source_completion.ttft_seconds,
        decode_throughput_tokens_per_second=source_completion.decode_throughput_tokens_per_second,
        output_chunks=source_completion.output_chunks,
    )


@app.post("/api/retrieve_drop")
async def retrieve_drop(request: CompletionRequest) -> DropResult:
    """Endpoint for performing retrieve and drop with the given request."""
    return await asyncio.to_thread(_retrieve_drop, request)


def _retrieve_drop(req: CompletionRequest) -> DropResult:
    """Perform a retrieve and drop operation with the given request."""
    global ctx
    start_time = default_timer()

    retrieve_result = lmc_sdk.retrieve(
        ctx=ctx,
        tokens=SOURCE_TOKENS[req.id]["source"],
    )
    
    if retrieve_result is None:
        raise RuntimeError("source retrieve missed the expected cached prefix")

    elapsed_time = default_timer() - start_time
    print(f"[RETRIEVE] took {elapsed_time:.6f} seconds")

    start_time = default_timer()

    retrieve_result_hit_tokens = int(retrieve_result.shape[2])
    hidden_dim = retrieve_result.shape[3]
    num_kv_heads = hidden_dim // app.state.head_size

    if req.drop_compression > 0.0:
        if req.drop_compression < 1.0:
            raise ValueError("drop_compression must be 0.0 or >= 1.0")
        # prefer compression rate over chunk count
        # so if drop_compression=2.0, will keep half the tokens, if drop_compression=4.0, will keep 1/4 the tokens
        # then round down to a multiple of chunk_size
        drop_tokens = (int(retrieve_result_hit_tokens * (1.0 - 1.0 / req.drop_compression)) // app.state.chunk_size) * app.state.chunk_size
    else:
        drop_tokens = req.drop_chunk * app.state.chunk_size

    if drop_tokens >= retrieve_result_hit_tokens:
        raise ValueError(
            f"drop_chunk ({req.drop_chunk}) drops {drop_tokens} tokens, "
            f"but retrieve_result_hit_tokens is only {retrieve_result_hit_tokens}"
        )
    
    tokens_to_keep = ((retrieve_result_hit_tokens - drop_tokens) // app.state.chunk_size) * app.state.chunk_size
    if tokens_to_keep == 0:
        raise ValueError("fewer than one complete chunk to store")

    remaining_to_keep = tokens_to_keep - app.state.chunk_size
    if remaining_to_keep < 0:
        raise ValueError("tokens_to_keep must be at least one chunk")

    work_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_chunk_start = retrieve_result_hit_tokens - app.state.chunk_size
    last_chunk_indices = torch.arange(
        last_chunk_start,
        retrieve_result_hit_tokens,
        dtype=torch.long,
    )

    if req.drop_algorithm == "random":
        selected_before_last = torch.tensor(
            random.sample(range(last_chunk_start), remaining_to_keep),
            dtype=torch.long,
        )
    else:
        num_layers = retrieve_result.shape[1]

        token_scores = torch.zeros(
            retrieve_result_hit_tokens,
            dtype=torch.float32,
            device=work_device,
        )

        # only move K cache to GPU for L2 scoring
        key_cache = retrieve_result[0].to(work_device)
        for layer in range(num_layers):
            keys = key_cache[layer].view(
                retrieve_result_hit_tokens,
                num_kv_heads,
                app.state.head_size,
            )
            token_scores += torch.norm(keys.float(), p=2, dim=-1).mean(dim=-1)

        selected_before_last = torch.topk(
            token_scores[:last_chunk_start],
            k=remaining_to_keep,
        ).indices.cpu()

    keep_indices = torch.cat([selected_before_last, last_chunk_indices])
    keep_indices = keep_indices.sort().values

    edited_kv = retrieve_result[:, :, keep_indices, :].clone()
    kept_cached_tokens = [SOURCE_TOKENS[req.id]["source"][i] for i in keep_indices.tolist()]
    uncached_tail_tokens = SOURCE_TOKENS[req.id]["source"][retrieve_result_hit_tokens:]
    edited_tokens = kept_cached_tokens + uncached_tail_tokens

    if req.drop_remap_pe:
        edited_kv = edited_kv.to(work_device)
        old_positions = keep_indices.to(work_device)
        new_positions = torch.arange(
            len(kept_cached_tokens),
            device=work_device,
            dtype=torch.long,
        )

        is_neox_style = True

        for layer in range(edited_kv.shape[1]):
            edited_kv[0, layer] = _rerotate_k_cache(
                k_flat=edited_kv[0, layer],
                old_positions=old_positions,
                new_positions=new_positions,
                config=app.state.config,
                head_size=app.state.head_size,
                is_neox_style=is_neox_style,
            )

        edited_kv = edited_kv.cpu()
    
    SOURCE_TOKENS[req.id]["edited_kv"] = edited_kv
    SOURCE_TOKENS[req.id]["edited_tokens"] = edited_tokens

    elapsed_time = default_timer() - start_time

    print(
        f"[TOKEN_DROP] dropped={drop_tokens} remaining={len(edited_tokens)}"
        f" took {elapsed_time:.6f} seconds"
    )

    start_time = default_timer()

    edited_salt = f"edited-{hash(tuple(SOURCE_TOKENS[req.id]['source']))}-{req.id}"
    SOURCE_TOKENS[req.id]["edited_salt"] = edited_salt

    store_result = lmc_sdk.store(
        ctx=ctx,
        kv=edited_kv,
        tokens=kept_cached_tokens,
        # cache_salt=edited_salt,
    )
    elapsed_time = default_timer() - start_time
    print(f"[STORE] took {elapsed_time:.6f} seconds")

    if not store_result:
        raise RuntimeError(f"Failed to store tokens out of {len(edited_tokens)} edited tokens")

    return DropResult(
        dropped_tokens=drop_tokens,
        remaining_tokens=len(edited_tokens),
        stored_tokens=store_result
    )

@app.post("/api/chat_completion")
async def send(request: CompletionRequest) -> CompletionResult:
    """Endpoint for sending request for prefill, drop tokens, then inference."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(SEND_EXECUTOR, _send, request)


def _send(req: CompletionRequest) -> CompletionResult:
    """Blocking token-dropping pipeline run in a worker thread."""

    start_time = default_timer()
    target_completion = _post_completion_streaming(
        vllm_url=app.state.vllm_url,
        model_name=app.state.vllm_model_name,
        prompt=SOURCE_TOKENS[req.id]["edited_tokens"],
        max_tokens=req.max_tokens,
        timeout=app.state.timeout,
        # cache_salt=SOURCE_TOKENS[req.id]["edited_salt"]
    )

    elapsed_time = default_timer() - start_time
    print(f"[GENERATE] took {elapsed_time:.6f} seconds")

    return CompletionResult(
        text=target_completion.text,
        elapsed_seconds=target_completion.elapsed_seconds,
        ttft_seconds=target_completion.ttft_seconds,
        decode_throughput_tokens_per_second=target_completion.decode_throughput_tokens_per_second,
        output_chunks=target_completion.output_chunks,
    )


def main() -> None:
    """Run the command-line driver."""
    global app, ctx

    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    head_size = getattr(
        config, 
        "head_dim", 
        config.hidden_size // config.num_attention_heads
    )
    lmcache_model_name = _registered_model_name(
        args.lmcache_url,
        args.lmcache_model_name or args.vllm_model_name,
        args.timeout,
    )
    app.state = AppState(
        model=args.model,
        vllm_model_name=args.vllm_model_name,
        lmcache_model_name=lmcache_model_name,
        tokenizer=tokenizer,
        config=config,
        lmcache_url=args.lmcache_url,
        vllm_url=args.vllm_url,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        trust_remote_code=args.trust_remote_code,
        head_size=head_size,
    )

    server_host = args.lmcache_mp_host # "tcp://localhost"
    server_port = args.lmcache_mp_port # 6556
    server_url = f"{server_host}:{server_port}"
    ctx = lmc_sdk.connect(
        url=server_url,
        http_url=args.lmcache_url,
        model_name=app.state.lmcache_model_name,
        timeout=app.state.timeout,
    )
    uvicorn.run(app, host=args.app_host, port=args.app_port)

    lmc_sdk.close(ctx)

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
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--lmcache-url", required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--app-host", default="0.0.0.0")
    parser.add_argument("--app-port", type=int, default=9000)
    parser.add_argument("--lmcache-mp-host", default="tcp://localhost")
    parser.add_argument("--lmcache-mp-port", type=int, default=6556)
    return parser


if __name__ == "__main__":
    main()
