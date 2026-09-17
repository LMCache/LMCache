# SPDX-License-Identifier: Apache-2.0
"""Factories for a real vLLM ``Scheduler`` wired to ``LMCacheMPConnector``.

Ported from vLLM's ``tests/v1/core/utils.py`` and
``tests/v1/kv_connector/unit/utils.py`` (which are not shipped in the
installed package) and trimmed to what the preemption harness needs.
"""

# Standard
from typing import Any

# Third Party
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
import torch

EOS_TOKEN_ID = 50256

_none_hash_initialized = False


def _ensure_none_hash() -> None:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True


def create_scheduler(
    *,
    num_blocks: int,
    block_size: int = 16,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 8192,
    max_model_len: int | None = None,
    enable_prefix_caching: bool = True,
    enable_chunked_prefill: bool = True,
    long_prefill_token_threshold: int = 0,
    scheduling_policy: str = "fcfs",
    async_scheduling: bool = False,
    kv_connector_extra_config: dict[str, Any] | None = None,
    kv_load_failure_policy: str = "recompute",
    model: str = "facebook/opt-125m",
) -> Scheduler:
    """Build a Scheduler whose KV connector is ``LMCacheMPConnector``.

    ``num_blocks`` includes vLLM's null block, so the usable pool is
    ``num_blocks - 1`` blocks.  Keep it tiny to force preemption.
    """
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    # vLLM requires max_num_batched_tokens >= max_num_seqs.
    max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
        # Deterministic admission / preemption mechanics.
        watermark=0.0,
        policy=scheduling_policy,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    extra = {"lmcache.mp.port": 0}
    if kv_connector_extra_config:
        extra.update(kv_connector_extra_config)
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
        kv_load_failure_policy=kv_load_failure_policy,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    if async_scheduling:
        # Third Party
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        scheduler_cls: type[Scheduler] = AsyncScheduler
    else:
        scheduler_cls = Scheduler
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
    block_size: int = 16,
    priority: int = 0,
    cache_salt: str | None = None,
) -> Request:
    """Make a deterministic request that never stops on EOS."""
    _ensure_none_hash()
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        priority=priority,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def make_prompt(
    prefix_id: int, prefix_len: int, unique_id: int, unique_len: int
) -> list[int]:
    """Prompt = a shared prefix (by ``prefix_id``) + a request-unique suffix.

    Token values are distinct per position so block hashes behave like real
    text rather than degenerate repeated tokens.
    """
    prefix = [10_000 + prefix_id * 1_000 + i for i in range(prefix_len)]
    suffix = [100_000 + unique_id * 1_000 + i for i in range(unique_len)]
    return prefix + suffix


def create_model_runner_output(
    req_ids: list[str],
    sampled_token_ids: list[list[int]],
    *,
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    kv_connector_worker_meta: Any = None,
) -> ModelRunnerOutput:
    kv_connector_output = None
    if (
        finished_sending
        or finished_recving
        or invalid_block_ids
        or kv_connector_worker_meta is not None
    ):
        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            invalid_block_ids=set(invalid_block_ids or ()),
            kv_connector_worker_meta=kv_connector_worker_meta,
        )
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )
