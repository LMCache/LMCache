# SPDX-License-Identifier: Apache-2.0
"""Check that planning and worker initialization size the same staging window."""

# Standard
from types import ModuleType, SimpleNamespace
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any
import logging
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import utils
from lmcache.v1.broadcast_transfer import (
    chunk_bytes,
    chunks_per_window,
    staging_bytes,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

pytestmark = pytest.mark.no_shared_allocator

# A DeepSeek-V3 class MLA layout: 61 layers, 576-wide latent, two-byte KV.
LAYERS = 61
LATENT = 576


LogCapture = Callable[[str, int], AbstractContextManager[None]]


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Supply a fresh config and the slice of vLLM's config planning reads."""
    monkeypatch.setattr(utils, "_config_instance", LMCacheEngineConfig.from_defaults())
    torch_utils = ModuleType("vllm.utils.torch_utils")
    torch_utils.get_kv_cache_torch_dtype = (  # type: ignore[attr-defined]
        lambda cache_dtype, model_dtype: torch.bfloat16
    )
    monkeypatch.setitem(sys.modules, torch_utils.__name__, torch_utils)
    parallel_config = SimpleNamespace(world_size=2, rank=0)
    model_config = SimpleNamespace(
        model="deepseek-like",
        use_mla=True,
        dtype=torch.bfloat16,
        get_num_layers=lambda parallel_config: LAYERS,
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: LATENT,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None, cache_dtype="auto"),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        speculative_config=None,
    )


def worker_layout(chunk_size: int) -> LMCacheMetadata:
    """The metadata a worker builds for the fixture's model."""
    return LMCacheMetadata(
        model_name="deepseek-like",
        world_size=2,
        local_world_size=2,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(LAYERS, 1, chunk_size, 1, LATENT),
        use_mla=True,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize(
    ("chunk_size", "window_bytes", "expected_chunks"),
    [
        (256, 64 * 1024**2, 3),  # 17.2 MiB chunks: three fit, the target holds
        (1024, 64 * 1024**2, 1),  # 68.6 MiB chunks: raised to one chunk
        (256, 1, 1),
    ],
)
def test_planned_staging_matches_the_worker_window(
    config: Any, chunk_size: int, window_bytes: int, expected_chunks: int
) -> None:
    config.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.chunk_size": chunk_size,
        "lmcache.retrieve_window_bytes": window_bytes,
    }

    planned = utils.get_required_device_memory(config)

    one_chunk = chunk_bytes(worker_layout(chunk_size), chunk_size)
    assert one_chunk == LAYERS * LATENT * 2 * chunk_size
    assert chunks_per_window(window_bytes, one_chunk) == expected_chunks
    assert planned == staging_bytes(one_chunk, expected_chunks)
    # The target is a ceiling whenever one chunk fits; otherwise one chunk.
    assert planned <= max(window_bytes, one_chunk)


@pytest.mark.parametrize(
    "disabled_by", ["model", "layerwise", "first_rank", "single_worker"]
)
def test_inactive_retrieval_reserves_nothing(config: Any, disabled_by: str) -> None:
    if disabled_by == "model":
        config.model_config.use_mla = False
    elif disabled_by == "single_worker":
        config.parallel_config.world_size = 1
    elif disabled_by == "layerwise":
        config.kv_transfer_config.kv_connector_extra_config = {
            "lmcache.use_layerwise": True,
        }
    else:
        config.kv_transfer_config.kv_connector_extra_config = {
            "lmcache.extra_config": {"save_only_first_rank": False},
        }
    assert utils.get_required_device_memory(config) == 0
    utils.validate_retrieval_memory_budget(config)


@pytest.mark.parametrize("shortfall", ["no_hook", "nothing", "one_byte", "none"])
def test_automatic_sizing_warns_unless_fully_reserved(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    config: Any,
    shortfall: str,
    lmcache_logs: LogCapture,
) -> None:
    required = utils.get_required_device_memory(config)
    factory = SimpleNamespace()
    if shortfall != "no_hook":
        reserved = {"nothing": 0, "one_byte": required - 1, "none": required}[shortfall]
        factory.get_required_device_memory = lambda config: reserved
    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    module.KVConnectorFactory = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with lmcache_logs("lmcache.integration.vllm.utils", logging.WARNING):
        utils.validate_retrieval_memory_budget(config)

    assert ("not accounted for" in caplog.text) == (shortfall != "none")


def test_explicit_kv_budget_supports_older_vllm(config: Any) -> None:
    config.cache_config.kv_cache_memory_bytes = 1024**3
    utils.validate_retrieval_memory_budget(config)


def test_invalid_override_is_rejected_during_planning(config: Any) -> None:
    config.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.retrieve_window_bytes": 0,
    }
    with pytest.raises(ValueError, match="retrieve_window_bytes"):
        utils.get_required_device_memory(config)
