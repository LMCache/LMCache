# SPDX-License-Identifier: Apache-2.0
"""``lmcache.mp.lookahead_chunk_keys`` decides whether LMCache keys use
vLLM's EAGLE / MTP lookahead tokens; the vLLM opt-in is unconditional."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

pytestmark = pytest.mark.no_shared_allocator

VLLM_LOOKAHEAD_TOKENS = 1


def _config(extra_config: dict[str, Any]) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector",
                kv_role="kv_both",
                kv_connector_extra_config=extra_config,
            ),
            num_hash_lookahead_tokens=VLLM_LOOKAHEAD_TOKENS,
        ),
    )


@pytest.fixture
def adapter_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the scheduler adapter's arguments; transport is stubbed."""
    captured: dict[str, Any] = {}

    def make_adapter(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock(lmcache_tokens_per_chunk=4)

    monkeypatch.setattr(connector_mod, "LMCacheMPSchedulerAdapter", make_adapter)
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    return captured


@pytest.mark.parametrize(
    ("extra_config", "expected"),
    [
        ({}, 0),
        ({"lmcache.mp.lookahead_chunk_keys": False}, 0),
        ({"lmcache.mp.lookahead_chunk_keys": True}, VLLM_LOOKAHEAD_TOKENS),
    ],
)
def test_lookahead_keys_follow_the_flag(
    adapter_kwargs: dict[str, Any], extra_config: dict[str, Any], expected: int
) -> None:
    """Keys carry vLLM's lookahead tokens only when the flag is on."""
    LMCacheMPConnector(_config(extra_config), KVConnectorRole.SCHEDULER)
    assert adapter_kwargs["lookahead_tokens"] == expected


def test_vllm_opt_in_does_not_depend_on_the_flag() -> None:
    """vLLM's lookahead block hashes are enabled whatever the flag says."""
    for extra_config in ({}, {"lmcache.mp.lookahead_chunk_keys": True}):
        transfer_config = KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        )
        assert LMCacheMPConnector.supports_lookahead_block_hashes(transfer_config)


def test_non_boolean_flag_is_rejected(adapter_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="lookahead_chunk_keys"):
        LMCacheMPConnector(
            _config({"lmcache.mp.lookahead_chunk_keys": "yes"}),
            KVConnectorRole.SCHEDULER,
        )
