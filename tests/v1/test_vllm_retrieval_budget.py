# SPDX-License-Identifier: Apache-2.0
"""Check that planning and worker initialization use the same staging budget."""

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any
import logging
import sys

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import utils
from lmcache.v1.config import LMCacheEngineConfig

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Supply a fresh loaded config without importing or initializing vLLM."""
    monkeypatch.setattr(utils, "_config_instance", LMCacheEngineConfig.from_defaults())
    return SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )


@pytest.mark.parametrize("budget", [128, 64 * 1024**2, 1024**3])
def test_planned_budget_matches_worker_config(config: Any, budget: int) -> None:
    config.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.retrieve_buffer_size": budget,
    }
    planned = utils.get_required_device_memory(config)
    worker_config = utils.lmcache_get_or_create_config(config)
    assert planned == worker_config.retrieve_buffer_size == budget


@pytest.mark.parametrize("disabled_by", ["model", "layerwise", "first_rank"])
def test_inactive_retrieval_reserves_nothing(config: Any, disabled_by: str) -> None:
    if disabled_by == "model":
        config.model_config.use_mla = False
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


@pytest.mark.parametrize("reserved", [None, 0, 64 * 1024**2 - 1, 64 * 1024**2])
def test_automatic_sizing_warns_unless_fully_reserved(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    config: Any,
    reserved: int | None,
) -> None:
    factory = SimpleNamespace()
    if reserved is not None:
        factory.get_required_device_memory = lambda config: reserved
    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    module.KVConnectorFactory = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with caplog.at_level(logging.WARNING, logger="lmcache.integration.vllm.utils"):
        utils.validate_retrieval_memory_budget(config)
    warned = "not accounted for" in caplog.text
    assert warned == (reserved != 64 * 1024**2)


def test_explicit_kv_budget_supports_older_vllm(config: Any) -> None:
    config.cache_config.kv_cache_memory_bytes = 1024**3
    utils.validate_retrieval_memory_budget(config)


def test_invalid_override_is_rejected_during_planning(config: Any) -> None:
    config.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.retrieve_buffer_size": 0,
    }
    with pytest.raises(ValueError, match="retrieve_buffer_size"):
        utils.get_required_device_memory(config)
