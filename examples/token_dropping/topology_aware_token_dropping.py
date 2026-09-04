# SPDX-License-Identifier: Apache-2.0
"""Run a topology-aware token-selection plan against a live LMCache request."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
import argparse
import json
import time

# Third Party
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from utils import make_post_completion, rerotate_k_cache
import torch

# First Party
from lmcache.sdk.token_selection import (
    CacheGroupGeometry,
    CacheSemanticKind,
    CacheTopologyDescriptor,
    GroupSelection,
    LogicalSpan,
    PlanValidationCode,
    RetentionAction,
    TokenSelectionPlan,
    expand_plan,
)
import lmcache.sdk as lmc_sdk

POLICY_REVISION = "middle-chunk-drop-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--lmcache-url", default="http://localhost:8080")
    parser.add_argument("--lmcache-mq-url", default="tcp://localhost:6555")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--prompt-chunks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def build_topology(kv_tensor: torch.Tensor, chunk_size: int) -> CacheTopologyDescriptor:
    """Describe the live dense KV layout without exposing tensor addresses."""
    bytes_per_token = (
        kv_tensor.shape[0]
        * kv_tensor.shape[1]
        * kv_tensor.shape[3]
        * kv_tensor.element_size()
    )
    dense = CacheGroupGeometry(
        group_id="dense-kv",
        semantic_kind=CacheSemanticKind.DENSE_ATTENTION,
        logical_tokens_per_block=chunk_size,
        physical_entries_per_block=chunk_size,
        compression_ratio=1,
        rank_sharding="single-rank-contiguous",
        page_stride_bytes=bytes_per_token,
        alignment_bytes=kv_tensor.element_size(),
    )
    return CacheTopologyDescriptor.create(
        model_architecture="live-vllm-dense-kv",
        backend_name="lmcache-mp",
        topology_version="v1",
        groups=(dense,),
    )


def build_plan(
    *,
    request_id: str,
    topology: CacheTopologyDescriptor,
    cached_tokens: int,
    chunk_size: int,
) -> TokenSelectionPlan:
    """Select the middle half of complete chunks for invalidation."""
    num_chunks = cached_tokens // chunk_size
    if num_chunks < 3:
        raise ValueError("at least three complete cached chunks are required")
    drop_count = max(1, num_chunks // 2)
    drop_start_chunk = max(1, (num_chunks - drop_count) // 2)
    drop_start_chunk = min(drop_start_chunk, num_chunks - drop_count)
    span = LogicalSpan(
        drop_start_chunk * chunk_size,
        (drop_start_chunk + drop_count) * chunk_size,
    )
    return TokenSelectionPlan.create(
        request_id=request_id,
        request_generation=0,
        decode_round=0,
        accepted_seq_len=cached_tokens,
        source_kv_revision=0,
        topology_fingerprint=topology.fingerprint,
        policy_revision=POLICY_REVISION,
        groups=(
            GroupSelection(
                group_id="dense-kv",
                semantic_kind=CacheSemanticKind.DENSE_ATTENTION,
                logical_span=span,
                required_siblings=(),
                action=RetentionAction.INVALIDATE,
            ),
        ),
    )


def expand_live_plan(
    plan: TokenSelectionPlan,
    topology: CacheTopologyDescriptor,
    *,
    request_generation: int = 0,
):
    return expand_plan(
        plan,
        topology,
        request_id=plan.request_id,
        request_generation=request_generation,
        decode_round=0,
        accepted_seq_len=plan.accepted_seq_len,
        source_kv_revision=0,
        policy_revision=POLICY_REVISION,
        current_step=0,
    )


def apply_topology_plan(
    request_stream: lmc_sdk.request.LMCacheRequestStream,
    model_config: PretrainedConfig,
    chunk_size: int,
    timeout: float,
) -> dict[str, object]:
    """Expand one semantic plan and use it to edit the live cached tensor."""
    evidence: dict[str, object] = {}

    def edit(
        tensors: Mapping[lmc_sdk.context.LMCacheSDKCacheKind, torch.Tensor],
        token_source: Sequence[int],
    ) -> tuple[torch.Tensor, Sequence[int]]:
        kv_tensor = tensors[lmc_sdk.context.LMCacheSDKCacheKind.KV]
        cached_tokens = int(kv_tensor.shape[2])
        topology = build_topology(kv_tensor, chunk_size)
        plan = build_plan(
            request_id=request_stream.request_stream_id,
            topology=topology,
            cached_tokens=cached_tokens,
            chunk_size=chunk_size,
        )
        expansion = expand_live_plan(plan, topology)
        if not expansion.validation.valid or len(expansion.operations) != 1:
            raise RuntimeError(f"valid plan did not expand: {expansion.validation}")
        operation = expansion.operations[0]
        if operation.requires_residual_handling:
            raise RuntimeError("this example requires a full-chunk selection")
        if operation.full_block_logical_span is None:
            raise RuntimeError("selection did not contain a full physical block")

        lo = operation.full_block_logical_span.start_token
        hi = operation.full_block_logical_span.end_token
        keep_idx = torch.cat([torch.arange(lo), torch.arange(hi, cached_tokens)])
        kept_ids = list(token_source[:lo]) + list(token_source[hi:cached_tokens])
        compacted = rerotate_k_cache(
            kv_tensor[:, :, keep_idx, :].clone(),
            old_positions=keep_idx,
            new_positions=torch.arange(keep_idx.numel(), dtype=torch.long),
            model_config=model_config,
        )

        stale = expand_live_plan(plan, topology, request_generation=1)
        stale_codes = [issue.code for issue in stale.validation.issues]
        generation_mismatch = PlanValidationCode.REQUEST_GENERATION_MISMATCH
        if stale.operations or generation_mismatch not in stale_codes:
            raise RuntimeError("stale request generation did not fail closed")

        evidence.update(
            {
                "topology_fingerprint": topology.fingerprint,
                "plan_digest": plan.plan_digest,
                "logical_drop_span": [lo, hi],
                "physical_entry_range": [
                    operation.physical_entry_start,
                    operation.physical_entry_end,
                ],
                "physical_byte_range": [operation.byte_start, operation.byte_end],
                "source_cached_tokens": cached_tokens,
                "kept_cached_tokens": len(kept_ids),
                "stale_generation_validation": [code.value for code in stale_codes],
                "stale_generation_operations": len(stale.operations),
            }
        )
        return compacted, kept_ids

    request_stream.modify_kv(edit, timeout=timeout)
    if not evidence:
        raise RuntimeError("cache edit completed without expansion evidence")
    return evidence


def main() -> None:
    args = parse_args()
    invalid_size = args.chunk_size <= 0 or args.max_tokens <= 0
    if invalid_size or args.prompt_chunks < 3:
        raise ValueError(
            "chunk size/max tokens must be positive; use >=3 prompt chunks"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    prompt_tokens = args.chunk_size * args.prompt_chunks
    seed = tokenizer.encode(
        "Topology plans keep cache policy independent from physical layout. ",
        add_special_tokens=False,
    )
    prompt = (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]
    post_completion = make_post_completion(args.vllm_url, args.model, args.timeout)
    ctx = lmc_sdk.kvcache.connect(
        url=args.lmcache_mq_url,
        http_url=args.lmcache_url,
        model_name=args.model,
        timeout=args.timeout,
    )
    try:
        stream = lmc_sdk.request.create_request(
            contexts=[ctx],
            post_completion=post_completion,
            prompt_token_ids=prompt,
            cache_salt=f"topology-aware-drop-{time.time_ns()}",
        )
        prefill = stream.generate(
            {"max_tokens": 1, "temperature": 0.0, "ignore_eos": True}
        )
        evidence = apply_topology_plan(
            stream,
            model_config=model_config,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
        )
        decode = stream.generate(
            {"max_tokens": args.max_tokens, "temperature": 0.0, "ignore_eos": True}
        )
        evidence.update(
            {
                "model": args.model,
                "request_stream_id": stream.request_stream_id,
                "prefill_output_tokens": prefill.output_tokens,
                "decode_output_tokens": decode.output_tokens,
                "decode_text_preview": stream.output_text[:160],
            }
        )
        print(json.dumps(evidence, indent=2))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
