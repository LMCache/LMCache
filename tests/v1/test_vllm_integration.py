# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import (
    create_lmcache_metadata,
    resolve_vllm_worker_identity,
)
from lmcache.v1.config import LMCacheEngineConfig


def _make_vllm_config(
    *,
    tp_pp_rank: int = 0,
    tp_pp_world_size: int = 1,
    tp_pp_local_world_size: int = 1,
    dp_rank: int = 0,
    dp_local_rank: int = 0,
    dp_size: int = 1,
    api_process_count: int = 1,
):
    model_config = SimpleNamespace(
        model="test-model",
        dtype=torch.bfloat16,
        use_mla=True,
        served_model_name="test-served-model",
    )
    model_config.get_num_layers = lambda parallel_config: 61
    model_config.get_num_kv_heads = lambda parallel_config: 1
    model_config.get_head_size = lambda: 576

    parallel_config = SimpleNamespace(
        rank=tp_pp_rank,
        world_size=tp_pp_world_size,
        local_world_size=tp_pp_local_world_size,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        data_parallel_rank_local=dp_local_rank,
        data_parallel_index=dp_rank,
        _api_process_count=api_process_count,
        distributed_executor_backend=None,
        nnodes=1,
    )

    cache_config = SimpleNamespace(cache_dtype="auto")
    kv_transfer_config = SimpleNamespace(
        engine_id="test-engine-id",
        kv_connector_extra_config={},
    )

    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
    )


def test_resolve_vllm_worker_identity_uses_dp_rank_and_current_device():
    """Dense-model DP workers should keep distinct LMCache identity/device."""

    vllm_config = _make_vllm_config(
        dp_rank=6,
        dp_local_rank=6,
        dp_size=1,
        api_process_count=8,
    )
    fake_torch_dev = SimpleNamespace(
        device_count=lambda: 8,
        current_device=lambda: 6,
    )

    with patch(
        "lmcache.integration.vllm.utils.get_vllm_torch_dev",
        return_value=(fake_torch_dev, "cuda"),
    ):
        identity = resolve_vllm_worker_identity(vllm_config)

    assert identity.world_size == 8
    assert identity.worker_id == 6
    assert identity.local_world_size == 8
    assert identity.local_worker_id == 6


def test_resolve_vllm_worker_identity_falls_back_to_dp_local_rank():
    """DP local rank should still work when current_device is unavailable."""

    vllm_config = _make_vllm_config(
        dp_rank=5,
        dp_local_rank=5,
        dp_size=1,
        api_process_count=8,
    )

    class FakeTorchDevice:
        def device_count(self) -> int:
            return 8

        def current_device(self) -> int:
            raise RuntimeError("device not initialized")

    with patch(
        "lmcache.integration.vllm.utils.get_vllm_torch_dev",
        return_value=(FakeTorchDevice(), "cuda"),
    ):
        identity = resolve_vllm_worker_identity(vllm_config)

    assert identity.world_size == 8
    assert identity.worker_id == 5
    assert identity.local_world_size == 8
    assert identity.local_worker_id == 5


def test_create_lmcache_metadata_is_dp_aware_for_dense_workers():
    """Metadata should not collapse all dense-model DP workers to rank 0."""

    vllm_config = _make_vllm_config(
        dp_rank=7,
        dp_local_rank=7,
        dp_size=1,
        api_process_count=8,
    )
    fake_torch_dev = SimpleNamespace(
        device_count=lambda: 8,
        current_device=lambda: 7,
    )

    with (
        patch(
            "lmcache.integration.vllm.utils.get_vllm_torch_dev",
            return_value=(fake_torch_dev, "cuda"),
        ),
        patch(
            "lmcache.integration.vllm.utils.lmcache_get_or_create_config",
            return_value=LMCacheEngineConfig.from_defaults(),
        ),
    ):
        metadata, _ = create_lmcache_metadata(vllm_config=vllm_config, role="worker")

    assert metadata.world_size == 8
    assert metadata.worker_id == 7
    assert metadata.local_world_size == 8
    assert metadata.local_worker_id == 7
    assert metadata.kv_shape == (61, 1, 256, 1, 576)
