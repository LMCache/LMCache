# SPDX-License-Identifier: Apache-2.0
"""Pinned vLLM 0.25.1 TD request admission contracts."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

# First Party
from lmcache.integration.vllm.experimental.token_drop_scheduler import (
    admit_token_drop_request,
)


def _admission_connector(*, block_stride=False):
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False, async_scheduling=False
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(world_size=1, use_ubatching=False),
        use_v2_model_runner=False,
        attention_config=SimpleNamespace(backend=None),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )
    connector = SimpleNamespace(
        _vllm_config=cfg,
        _group_tokens_per_block=[16],
        _kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
                        {
                            "layer.0": FullAttentionSpec(
                                block_size=16,
                                num_kv_heads=2,
                                head_size=64,
                                dtype=torch.bfloat16,
                                indexes_kv_by_block_stride=block_stride,
                            )
                        }
                    )
                )
            ]
        ),
    )
    request = SimpleNamespace(skip_reading_prefix_cache=False)
    return connector, request


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("chunked", "chunked prefill"),
        ("speculative", "speculative decoding"),
        ("async", "synchronous scheduling"),
        ("parallel", "single GPU"),
        ("groups", "one KV cache group"),
    ],
)
def test_unsupported_td_is_rejected_before_installing_adaptors(
    kind, error, monkeypatch
):
    connector, request = _admission_connector()
    cfg = connector._vllm_config
    if kind == "chunked":
        cfg.scheduler_config.enable_chunked_prefill = True
    elif kind == "speculative":
        cfg.speculative_config = object()
    elif kind == "async":
        cfg.scheduler_config.async_scheduling = True
    elif kind == "parallel":
        cfg.parallel_config.world_size = 2
    else:
        connector._group_tokens_per_block = [16, 16]

    def forbidden():
        raise AssertionError("Adaptor installed before admission validated")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match=error):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_full_cudagraph_rejected_before_td_hooks(mode, monkeypatch):
    connector, request = _admission_connector()
    connector._vllm_config.compilation_config.cudagraph_mode = mode

    def forbidden():
        raise AssertionError("TD allocator/APC hook installed before mode validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        forbidden,
    )
    with pytest.raises(ValueError, match="CUDA Graph mode NONE or PIECEWISE"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize(
    "backend", [AttentionBackendEnum.FLASHINFER, AttentionBackendEnum.TRITON_ATTN]
)
def test_explicit_unsupported_attention_backend_rejected_before_hooks(
    backend, monkeypatch
):
    connector, request = _admission_connector()
    connector._vllm_config.attention_config = SimpleNamespace(backend=backend)

    def forbidden():
        raise AssertionError("TD hooks installed before backend validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_allocator_adapter",
        forbidden,
    )
    with pytest.raises(ValueError, match="requires FlashAttention backend"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize("kind", ["v2_runner", "microbatching"])
def test_td_admission_rejects_unsupported_runner_before_hooks(kind, monkeypatch):
    connector, request = _admission_connector()
    if kind == "v2_runner":
        connector._vllm_config.use_v2_model_runner = True
    else:
        connector._vllm_config.parallel_config.use_ubatching = True

    def forbidden():
        raise AssertionError("TD hooks installed before runner validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match="classic|microbatch"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False


@pytest.mark.parametrize(
    "kind",
    [
        "sliding_window",
        "sliding_alloc",
        "chunked_local",
        "non_causal",
        "sink",
        "quantized",
        "asymmetric_kv",
        "mixed",
        "missing_config",
        "missing_groups",
    ],
)
def test_td_admission_rejects_unsupported_kv_specs_before_hooks(kind, monkeypatch):
    connector, request = _admission_connector()
    base = dict(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    if kind == "sliding_window":
        spec = SlidingWindowSpec(**base, sliding_window=64)
    elif kind == "sliding_alloc":
        spec = FullAttentionSpec(**base, sliding_window=64)
    elif kind == "chunked_local":
        spec = FullAttentionSpec(**base, attention_chunk_size=64)
    elif kind == "non_causal":
        spec = FullAttentionSpec(**base, non_causal=True)
    elif kind == "sink":
        spec = SinkFullAttentionSpec(**base, sink_len=4)
    elif kind == "quantized":
        spec = FullAttentionSpec(**base, kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
    elif kind == "asymmetric_kv":
        spec = FullAttentionSpec(**base, head_size_v=32)
    else:
        spec = FullAttentionSpec(**base)
    if kind == "missing_config":
        connector._kv_cache_config = None
    elif kind == "missing_groups":
        connector._kv_cache_config.kv_cache_groups = []
    elif kind == "mixed":
        connector._kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "layer.0": spec,
                "layer.1": SlidingWindowSpec(**base, sliding_window=64),
            },
        )
    else:
        connector._kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec = UniformTypeKVCacheSpecs.from_specs({"layer.0": spec})

    def forbidden():
        raise AssertionError("TD hooks installed before topology validation")

    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter.install_token_drop_prefix_caching_publication_guard",
        forbidden,
    )
    with pytest.raises(ValueError, match="KV cache group|full-attention"):
        admit_token_drop_request(connector, request)
    assert request.skip_reading_prefix_cache is False
