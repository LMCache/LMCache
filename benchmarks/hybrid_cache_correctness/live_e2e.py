# SPDX-License-Identifier: Apache-2.0
"""Capture and compare two live vLLM + LMCache exact-prefix runs."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import time

# Third Party
from transformers import AutoTokenizer
import httpx
import torch

# First Party
from benchmarks.hybrid_cache_correctness import (
    CacheGroupFrame,
    HybridCorrectnessTrace,
    LifecycleEvent,
    LifecyclePhase,
    OutputFrame,
    RequestStateFrame,
    TopKEntry,
    TraceFrame,
    compare_traces,
    sha256_digest,
    write_report,
    write_trace,
)
import lmcache.sdk as lmc_sdk


@dataclass(frozen=True)
class CapturedToken:
    token_id: int
    text: str
    logprob: float
    top_k: tuple[TopKEntry, ...]


class RecordingCompletion:
    """vLLM streaming completion adapter that retains real output evidence."""

    def __init__(self, url: str, model: str, timeout: float, top_k: int) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.top_k = top_k
        self.calls: list[list[CapturedToken]] = []

    def __call__(
        self,
        prompt_token_ids: list[int],
        sampling_params: dict[str, Any],
        cache_salt: str,
    ) -> Iterable[lmc_sdk.request.TokenEvent]:
        payload = {
            **sampling_params,
            "model": self.model,
            "prompt": prompt_token_ids,
            "stream": True,
            "logprobs": self.top_k,
        }
        if cache_salt:
            payload["cache_salt"] = cache_salt
        captured: list[CapturedToken] = []
        self.calls.append(captured)
        timeout = httpx.Timeout(
            connect=self.timeout,
            read=None,
            write=self.timeout,
            pool=self.timeout,
        )
        with httpx.stream(
            "POST",
            f"{self.url}/v1/completions",
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                body = json.loads(line.removeprefix("data: "))
                choices = body.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                logprobs = choice.get("logprobs") or {}
                token_names = logprobs.get("tokens") or []
                token_logprobs = logprobs.get("token_logprobs") or []
                top_logprobs = logprobs.get("top_logprobs") or []
                if not token_names or not token_logprobs or not top_logprobs:
                    raise RuntimeError("vLLM response omitted requested logprobs")
                token_id = parse_token_id(token_names[-1])
                logprob = float(token_logprobs[-1])
                entries = tuple(
                    TopKEntry(parse_token_id(name), float(value))
                    for name, value in sorted(top_logprobs[-1].items())
                )
                event = CapturedToken(
                    token_id=token_id,
                    text=str(choice.get("text", "")),
                    logprob=logprob,
                    top_k=entries,
                )
                captured.append(event)
                yield lmc_sdk.request.TokenEvent(token_id=token_id, text=event.text)


def parse_token_id(token: str) -> int:
    if ":" not in token:
        raise ValueError(
            f"token ids require vLLM --return-tokens-as-token-ids; received {token!r}"
        )
    return int(token.rsplit(":", 1)[-1])


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_digest(encoded)


def capture_run(
    *,
    run_id: str,
    request_id: str,
    prompt: list[int],
    cache_salt: str,
    model: str,
    vllm_url: str,
    lmcache_url: str,
    lmcache_mq_url: str,
    steps: int,
    top_k: int,
    timeout: float,
) -> tuple[HybridCorrectnessTrace, list[dict[str, float]]]:
    recorder = RecordingCompletion(vllm_url, model, timeout, top_k)
    ctx = lmc_sdk.kvcache.connect(
        url=lmcache_mq_url,
        http_url=lmcache_url,
        model_name=model,
        timeout=timeout,
    )
    frames: list[TraceFrame] = []
    events: list[LifecycleEvent] = []
    timings: list[dict[str, float]] = []
    try:
        stream = lmc_sdk.request.create_request(
            contexts=[ctx],
            post_completion=recorder,
            prompt_token_ids=prompt,
            cache_salt=cache_salt,
        )
        for step in range(steps):
            perf = stream.generate(
                {"max_tokens": 1, "temperature": 0.0, "ignore_eos": True}
            )
            captured = recorder.calls[-1]
            if len(captured) != 1:
                raise RuntimeError(f"step {step} returned {len(captured)} tokens")
            sequence = len(events)
            events.append(
                LifecycleEvent(
                    sequence=sequence,
                    step=step,
                    phase=LifecyclePhase.RETRIEVE_SUBMITTED,
                    request_generation=0,
                    operation_id=step,
                    group_id="dense-kv",
                )
            )
            kv = stream.retrieve(
                lmc_sdk.context.LMCacheSDKCacheKind.KV,
                timeout=timeout,
            )
            events.append(
                LifecycleEvent(
                    sequence=sequence + 1,
                    step=step,
                    phase=LifecyclePhase.RETRIEVE_COMPLETE,
                    request_generation=0,
                    operation_id=step,
                    group_id="dense-kv",
                    detail_digest=sha256_digest(tensor_bytes(kv)),
                )
            )
            cached_tokens = int(kv.shape[2])
            chunk_size = ctx.chunk_size
            chunks = tuple(range((cached_tokens + chunk_size - 1) // chunk_size))
            token = captured[0]
            frames.append(
                TraceFrame(
                    step=step,
                    output=OutputFrame(
                        token_id=token.token_id,
                        logprob=token.logprob,
                        top_k=token.top_k,
                    ),
                    request=RequestStateFrame(
                        request_generation=0,
                        accepted_seq_len=len(stream.tokens),
                        block_table_digest=digest_json(chunks),
                        prefix_digest=digest_json(stream.tokens),
                        drop_round_digest=digest_json({"drop_round": 0}),
                    ),
                    cache_groups=(
                        CacheGroupFrame(
                            group_id="dense-kv",
                            rank=0,
                            semantic_kind="dense_attention_sdk_contiguous",
                            logical_start=0,
                            logical_end=cached_tokens,
                            physical_page_ids=chunks,
                            dtype=str(kv.dtype),
                            shape=tuple(kv.shape),
                            stride=tuple(kv.stride()),
                            content_digest=sha256_digest(tensor_bytes(kv)),
                            revision="sdk-retrieve-v1",
                        ),
                    ),
                )
            )
            timings.append(
                {
                    "duration_s": perf.duration,
                    "ttft_s": perf.ttft,
                    "input_tokens_per_s": perf.input_tput,
                }
            )
    finally:
        ctx.close()
    return (
        HybridCorrectnessTrace(
            run_id=run_id,
            request_id=request_id,
            frames=tuple(frames),
            lifecycle_events=tuple(events),
            metadata=(
                ("cache_salt", cache_salt),
                ("capture", "live-vllm-lmcache-sdk"),
                ("model", model),
            ),
        ),
        timings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--lmcache-url", default="http://localhost:8080")
    parser.add_argument("--lmcache-mq-url", default="tcp://localhost:6555")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/lmcache-hybrid-correctness-e2e"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_tokens <= 0 or args.steps <= 0 or args.top_k <= 0:
        raise ValueError("prompt tokens, steps, and top-k must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = tokenizer.encode(
        "Hybrid cache traces localize the first correctness divergence. ",
        add_special_tokens=False,
    )
    prompt = (seed * ((args.prompt_tokens + len(seed) - 1) // len(seed)))[
        : args.prompt_tokens
    ]
    cache_salt = f"hybrid-correctness-e2e-{time.time_ns()}"
    common = {
        "request_id": "hybrid-correctness-e2e-request",
        "prompt": prompt,
        "cache_salt": cache_salt,
        "model": args.model,
        "vllm_url": args.vllm_url,
        "lmcache_url": args.lmcache_url,
        "lmcache_mq_url": args.lmcache_mq_url,
        "steps": args.steps,
        "top_k": args.top_k,
        "timeout": args.timeout,
    }
    reference, reference_timings = capture_run(run_id="reference", **common)
    candidate, candidate_timings = capture_run(run_id="candidate", **common)
    report = compare_traces(reference, candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_trace(reference, args.output_dir / "reference.json")
    write_trace(candidate, args.output_dir / "candidate.json")
    write_report(report, args.output_dir / "report.json")
    summary = {
        "matched": report.matched,
        "first_divergence": (
            asdict(report.first_divergence)
            if report.first_divergence is not None
            else None
        ),
        "frames": len(report.frame_metrics),
        "reference_timings": reference_timings,
        "candidate_timings": candidate_timings,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))
    if not report.matched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
